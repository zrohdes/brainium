# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='brainium',
    packages=find_packages(),
    version='0.0.2',
    license='CC BY-NC-SA 4.0',
    description='Machine learning library extended from Google Tensorflow with flexible prototyping features.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hieu Tr. Pham',
    author_email='hieupt.ai@gmail.com',
    url='https://github.com/hieupth/brainium',
    download_url='https://github.com/hieupth/brainium/archive/v_01.tar.gz',
    keywords=['machine learning', 'deep learning'],
    install_requires=['tensorflow-gpu'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
