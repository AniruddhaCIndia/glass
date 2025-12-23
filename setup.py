from setuptools import setup, find_packages

setup(
    name="glass",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
    description="A Python package for GLASS",
    author="Aniruddha",
    author_email="aniruddha.chakraborty@tifr.res.in",
)
