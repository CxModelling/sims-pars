from setuptools import setup, find_packages

setup(
    name='sims-pars',
    version='1.0.1',
    packages=find_packages(exclude=['notebooks', 'tests']),
    url='https://github.com/TimeWz667/sims-pars',
    license='MIT',
    author='TimeWz667',
    author_email='TimeWz667@gmail.com',
    description='Serving stochastic parameters to simulation models',
    install_requires=['pandas', 'networkx', 'astunparse', 'numpy', 'scipy', 'matplotlib']
)
