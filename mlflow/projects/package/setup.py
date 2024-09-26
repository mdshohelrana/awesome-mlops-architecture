from setuptools import setup

setup(
    name='package',
    version='0.1',
    description='A useful package',
    author="Shohel Rana",
    author_email="iamshohelrana@gmail.com",
    packages=['package.feature', 'package.training', "package.utils"],
    # install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'mlflow==2.3.1']
)