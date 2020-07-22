#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(name='CoveringAlgorithm',
      version='1.0',
      description='Covering Algorithm',
      author='Vincent Margot',
      author_email='vincent.margot@hotmail.fr',
      url='https://github.com/VMargot/CoveringAlgorithm',
      packages=['CoveringAlgorithm'],
      install_requires=requirements,
      )
