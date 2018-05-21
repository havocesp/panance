# -*- coding:utf-8 -*-
from setuptools import setup, find_packages

from panance import (__version__, __author__, __appname__, __dependencies__, __description__, __license__, __site__,
                     __email__)

setup(
    name=__appname__,
    version=__version__,
    packages=find_packages(exclude=['doc*', 'test*', 'venv*']),
    url=__site__,
    license=__license__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Python 3 Binance API wrapper built over Pandas Library.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ], install_requires=__dependencies__
)
