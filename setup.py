#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from harvesttext import __version__

setup(
    name='harvesttext',
    author="blmoistawinde",
    author_email="1840962220@qq.com",
    version=__version__,
    license='MIT',
    keywords='NLP, tokenizing, entity linking, sentiment analysis, text cleaning',
    url='https://github.com/blmoistawinde/HarvestText',
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages = find_packages(),
    platforms=["all"],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
      ],
    install_requires=open("requirements.txt", encoding='utf-8').read().split('\n'),
    tests_require=['pytest'],
    include_package_data = True,
    package_data ={"harvesttext":["resources/*"],}
)
