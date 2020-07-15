#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(name='RICE',
      version='1.0',
      description='RICE algorithm',
      author='Vincent Margot',
      author_email='vincent.margot@hotmail.fr',
      url='',
      packages=['RICE'],
      install_requires=requirements,
      )
