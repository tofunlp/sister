#!/usr/bin/env python
# coding: utf-8
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
        name='sister',
        use_scm_version=True,
        setup_requires=['setuptools_scm'],
        description='SImple SenTence EmbeddeR',
        long_description=open('./README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/sobamchan/sister',
        author='Sotaro Takeshita',
        author_email='oh.sore.sore.soutarou@gmail.com',
        packages=[
            'sister'
            ],
        license='MIT',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
            ],
        install_requires=[
            'fasttext', 'numpy', 'Janome', 'gensim', 'joblib'
            ],
        extras_require={
            'bert': ['transformers', 'torch', 'mecab-python3']
            },
        tests_requires=['pytest', 'importlib-resources']
        )
