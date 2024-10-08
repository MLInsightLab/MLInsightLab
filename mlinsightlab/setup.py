from setuptools import setup, find_packages

setup(
    name="mlinsightlab",
    version="0.0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.25.0",
        "pandas"
    ],
    author="MLIL Team",
    description="Your Open Source Serverless Data Science Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jacobrenn/MLInsightLab/tree/main/mlinsightlab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
