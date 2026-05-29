from setuptools import setup, find_packages

setup(
    name="doc_intel",
    version="0.1",
    packages=find_packages(include=['core', 'doc_intel', 'utils']),
)