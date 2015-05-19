from setuptools import setup, find_packages

setup(
    name='doger',
    version='0.1',
    packages=find_packages(),
    url='https://github.org/dominofire/doger',
    license='',
    author='Fernando Aguilar',
    author_email='fer.aguilar.reyes@gmail.com',
    description='Grid search with machine learning classification models',
    # Remember to keep requirements.txt updated
    install_requires=[
        'pandas >= 0.16.0',
        'scikit-learn', 
        'click',
        'joblib',
        'scipy >= 0.15.0',
        'matplotlib'
    ],
    entry_points='''
    [console_scripts]
    doger=doger.cli.puppet:cli
    '''
)
