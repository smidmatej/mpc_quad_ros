from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['mpcros'],
    package_dir={'': '.'}
)

setup(**d)