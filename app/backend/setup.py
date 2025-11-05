# Import
from setuptools import setup, find_packages

# Setup functions
setup(
    name="Naives Bayes Models",
    version="0.1",
    description="Basic spam identification model",
    packages=find_packages(),
    install_requires=[
        # Required packages for the app.
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "pydantic",
        "uvicorn",
    ]
)