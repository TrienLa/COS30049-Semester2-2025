from setuptools import setup, find_packages
setup(
    name="Naives Bayes Models",
    version="0.1",
    description="Basic spam identification model",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn"
        "numpy"
    ]
)