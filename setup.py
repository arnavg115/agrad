from setuptools import setup, find_packages

setup(
    name="agrad",
    version="1.0.2",
    description="A homecooked autograd library built w/ numpy.",
    author="arnavg115",
    packages=["agrad", "agrad.nn"],  # same as name
    install_requires=["numpy"],  # external packages as dependencies
)
