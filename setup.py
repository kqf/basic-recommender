from setuptools import setup, find_packages

setup(
    name="basic-recommender",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "develop-basic-recommender=model.main:develop",
            "train-basic-recommender=model.main:train",
        ],
    },
)
