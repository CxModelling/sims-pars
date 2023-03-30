import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


keywords = [
    "simulation",
    "probability theory",
    "Bayesian networks",
    "graph",
    "model fitting",
]


packages = [
    'sims_pars',
    'sims_pars.factory',
    'sims_pars.bayesnet',
    'sims_pars.simulation',
    'sims_pars.fit'
]


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]

    return requires


install_requires = []
extras_require = {
    dep: parse_requirements_file("requirements/" + dep + ".txt")
    for dep in ["core", "doc"]
}


setuptools.setup(
    name='sims-pars',
    version='2.5.1',
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
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9",
)
