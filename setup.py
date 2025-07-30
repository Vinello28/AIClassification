#!/usr/bin/env python
"""
Setup script for AI Classification package
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-classification",
    version="1.0.0",
    author="AI Classification Team",
    author_email="contact@example.com",
    description="A text classification system for AI-related content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vinello28/AIClassification",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.0.0",
        ],
        "server": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ai-classification-server=ai_classification.api.server:main",
            "ai-classification-train=ai_classification.scripts.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)