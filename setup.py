"""
Setup script for Taylor Expansion Learning.
"""

from setuptools import setup, find_packages

setup(
    name="taylor-expansion-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "sympy>=1.11.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.64.0",
        "python-Levenshtein>=0.20.0",
        "scikit-learn>=1.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for learning Taylor expansions using neural networks",
    keywords="taylor, expansion, neural networks, machine learning",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
