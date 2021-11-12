import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sims-pars',
    version='2.0.2',
    author="Chu-Chang Ku",
    author_email='TimeWz667@gmail.com',
    description='Serving stochastic parameters to simulation models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimeWz667/sims-pars",
    project_urls={
        "Bug Tracker": "https://github.com/TimeWz667/sims-pars/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['notebooks', 'tests']),
    install_requires=[
        'markdown',
        'joblib',
        'numpy',
        'scipy',
        'astunparse',
        'astunparse',
        'pandas',
        'tqdm'
    ],
    python_requires=">=3.8",
)
