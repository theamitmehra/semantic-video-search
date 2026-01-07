from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-video-search",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Find exact timestamps in videos using natural language - powered by CLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic-video-search",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
    ],
)
