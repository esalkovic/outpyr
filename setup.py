


from glob import glob
from setuptools import setup, find_packages


with open('README.rst') as f:
    _readme = f.read()

with open('LICENSE') as f:
    _license = f.read()

setup(
    name='outpyr',
    version='0.1.0',
    description='OutPyR',
    long_description=_readme,
    author='Edin Salkovic',
    author_email='',
    license=_license,
    packages=find_packages('src'),
    package_dir={'': 'src'},










    package=['outpyr'],


    entry_points={
        "console_scripts": ['outpyr = outpyr.train:main']
    },
    install_requires=[
        'matplotlib',
        'scipy',
        'pandas',
        'numpy',




        'tables',
        'seaborn',
        'mpmath',
        'numba',
        'multiprocess',
        'statsmodels',
        'tensorflow',
        'tensorflow_probability',






    ],
)
