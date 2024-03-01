from setuptools import setup, find_packages

setup(
   name='agrad',
   version='1.0.1',
   description='A homecooked autograd library built w/ numpy.',
   author='arnavg115',
   packages=find_packages(),  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
