from setuptools import setup, find_packages

setup(
    name='probsamplers',
    version='0.1.0',
    description='JupyterLab version of mcmc-demo',
    author='Rohit Goswami',
    author_email='rog32@hi.is',
    url='https://rgoswami.me',
    packages=find_packages(include=['probsamplers', 'probsamplers.*']),
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
)
