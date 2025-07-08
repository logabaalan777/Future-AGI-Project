from setuptools import setup, find_packages

setup(
    name="future-agi-intern-project",
    version="0.1.0",
    description="Future AGI Intern Project",
    author="Future AGI",
    author_email="logabaalan2004@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "matplotlib",
    ],
)