from setuptools import setup, find_packages

setup(
    name="deep-learning-project",
    version="0.1.0",
    description="Semantic segmentation based traffic collision warning system using DeepLabV3-ResNet50 and SegFormer",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
)