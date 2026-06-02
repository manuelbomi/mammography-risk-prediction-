"""Package setup for mammography risk prediction pipeline."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="mammography-risk-prediction",
    version="0.1.0",
    author="Manuel",
    description="Deep Learning Pipeline for Breast Cancer Risk Prediction Using Mammography Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mammography-risk-prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pydicom>=2.4.0",
        "opencv-python>=4.8.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
        ],
        "report": [
            "reportlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mammo-train=scripts.train:main",
            "mammo-evaluate=scripts.evaluate:main",
            "mammo-report=scripts.generate_report:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
