from setuptools import setup, find_packages

setup(
    name="mtp2bbd",
    version="0.1",
    author="Xiwen Wang, Jiaxi Ying, Daniel P. Palomar",
    description="Python package for implementing bridge-block decomposition approach for learning MTP2 GGMs",
    packages=find_packages(),
    install_requires=[
    "numpy>=1.19,<2",
    "igraph>=0.8,<1",
    "scipy"
    ] 
)