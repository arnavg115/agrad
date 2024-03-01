from setuptools import setup

setup(
   name='agrad',
   version='1.0',
   description='A homecooked autograd library built w/ numpy.',
   author='arnavg115',
   packages=['agrad'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)