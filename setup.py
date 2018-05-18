# -*- coding:utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='panance',
    version='0.1.4',
    packages=find_packages(exclude=['doc*', 'test*', 'venv*']),
    url='https://github.com/havocesp/panance',
    license='MIT',
    author='Daniel J. Umpierrez',
    author_email='',
    description='Python 3 Binance API wrapper built over Pandas Library',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Python 3 Binance API wrapper built over Pandas Library.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ], install_requires=['pandas', 'ccxt']
)
