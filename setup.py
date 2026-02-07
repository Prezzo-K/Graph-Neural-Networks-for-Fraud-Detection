"""Setup script for GNN Fraud Detection project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gnn-fraud-detection",
    version="0.1.0",
    author="CSCI 3834 Group",
    description="Graph Neural Networks for Financial Fraud Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Prezzo-K/Graph-Neural-Networks-for-Fraud-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.0.0",
        ],
    },
)
