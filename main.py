import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import clip
from collections import defaultdict
import time
from tqdm import tqdm

class SemanticVideoSearch:
    """
    Fast semantic video search using OpenAI's CLIP model.
    Leverages vectorized NumPy/PyTorch operations for efficient embedding computation.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """Initialize CLIP model and move to GPU if available."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.frame_embeddings = None
        self.timestamps = None
        self.video_path = None
    
    def extract_frames(self, video_path: str, fps_sample: int = 2) -> Tuple[np.ndarray, List[float]]:
        """
        Extract frames from video at specified sampling rate.
        Uses vectorized operations where possible for speed.
        
        Args:
            video_path: Path to video file
            fps_sample: Sample every nth frame (2 = every 0.5s at 30fps)
        
        Returns:
            frames: Array of extracted frames, timestamps: List of timestamps
        """
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        timestamps = []
        frame_idx = 0
        
        print(f"Extracting frames from {video_path}...")
        print(f"Video: {total_frames} frames @ {fps} FPS")
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % fps_sample == 0:
                    # Convert BGR to RGB for CLIP
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    timestamps.append(frame_idx / fps)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        print(f"Extracted {len(frames)} frames")
        
        return frames, timestamps
    
    def compute_frame_embeddings(self, frames: List[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """
        Compute CLIP embeddings for all frames using vectorized batch processing.
        This is where we leverage PyTorch's GPU-accelerated tensor operations.
        
        Args:
            frames: List of frame arrays
            batch_size: Process frames in batches for memory efficiency
        
        Returns:
            embeddings: Tensor of shape (num_frames, embedding_dim)
        """
        embeddings = []
        
        print("Computing frame embeddings...")
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size)):
                batch = frames[i:i+batch_size]
                
                # Preprocess all frames in batch (vectorized)
                processed_batch = torch.stack([
                    self.preprocess(frame) for frame in batch
                ]).to(self.device)
                
                # Get embeddings for entire batch at once (GPU acceleration)
                batch_embeddings = self.model.encode_image(processed_batch)
                
                # Normalize embeddings (vectorized operation)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu())
        
        # Stack all batch embeddings (vectorized)
        self.frame_embeddings = torch.cat(embeddings, dim=0)
        
        print(f"Embeddings shape: {self.frame_embeddings.shape}")
        return self.frame_embeddings
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Tuple[float, float, str]]:
        """
        Search for frames matching the text query using cosine similarity.
        Uses vectorized dot products for fast similarity computation.
        
        Args:
            query: Text description (e.g., "person wearing a red hat")
            top_k: Number of top matches to return
            threshold: Minimum cosine similarity score (0-1)
        
        Returns:
            results: List of (timestamp, similarity_score, time_formatted) tuples
        """
        if self.frame_embeddings is None:
            raise ValueError("No video loaded. Run extract_frames and compute_frame_embeddings first.")
        
        print(f"\nSearching for: '{query}'")
        
        # Encode query text
        with torch.no_grad():
            query_tokens = clip.tokenize(query).to(self.device)
            query_embedding = self.model.encode_text(query_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
        
        # Vectorized cosine similarity: dot product between normalized vectors
        # Frame embeddings shape: (num_frames, 512)
        # Query embedding shape: (1, 512)
        # Result shape: (num_frames,) using efficient matrix multiplication
        similarities = torch.mm(
            self.frame_embeddings,
            query_embedding.t()
        ).squeeze()  # Shape: (num_frames,)
        
        # Filter by threshold and get top-k
        valid_indices = torch.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            print(f"No matches found above threshold {threshold}")
            return []
        
        # Get top-k matches (vectorized indexing)
        top_similarities, top_indices = torch.topk(
            similarities[valid_indices],
            k=min(top_k, len(valid_indices))
        )
        
        # Build results
        results = []
        for sim_score, idx in zip(top_similarities, top_indices):
            actual_idx = valid_indices[idx].item()
            timestamp = self.timestamps[actual_idx]
            sim_float = sim_score.item()
            time_fmt = self._format_timestamp(timestamp)
            results.append((timestamp, sim_float, time_fmt))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 100)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"
    
    def display_results(self, results: List[Tuple[float, float, str]], query: str) -> None:
        """Pretty print search results."""
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results\n")
        
        for rank, (timestamp, similarity, time_fmt) in enumerate(results, 1):
            bar_width = int(similarity * 40)
            bar = "█" * bar_width + "░" * (40 - bar_width)
            print(f"{rank}. [{bar}] {similarity:.3f} @ {time_fmt}")
        
        print(f"{'='*70}\n")


def main():
    """Example usage of semantic video search."""
    # Initialize search engine
    search_engine = SemanticVideoSearch(device="cuda")
    
    # Example: using a sample video
    # For demo, we'll create a short dummy video or use an existing one
    video_path = "sample_video.mp4"
    
    # Check if video exists, otherwise provide instructions
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("\nUsage Example:")
        print("=" * 70)
        print("""
        # Step 1: Prepare your video
        video_path = "path/to/your/video.mp4"
        
        # Step 2: Initialize search engine
        search_engine = SemanticVideoSearch(device="cuda")
        
        # Step 3: Extract frames and compute embeddings
        frames, timestamps = search_engine.extract_frames(video_path, fps_sample=2)
        search_engine.frame_embeddings = search_engine.compute_frame_embeddings(frames)
        search_engine.timestamps = timestamps
        
        # Step 4: Search for your queries
        results = search_engine.search("person wearing a red hat", top_k=5)
        search_engine.display_results(results, "person wearing a red hat")
        
        # More searches:
        results = search_engine.search("car driving on highway", top_k=3)
        search_engine.display_results(results, "car driving on highway")
        """)
        print("=" * 70)
        return
    
    # Run the search pipeline
    try:
        # Extract frames
        frames, timestamps = search_engine.extract_frames(video_path, fps_sample=2)
        search_engine.timestamps = timestamps
        
        # Compute embeddings
        embeddings = search_engine.compute_frame_embeddings(frames, batch_size=32)
        
        # Perform searches
        queries = [
            "person wearing a red hat",
            "close-up face",
            "outdoor scene",
        ]
        
        for query in queries:
            results = search_engine.search(query, top_k=5, threshold=0.25)
            search_engine.display_results(results, query)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
