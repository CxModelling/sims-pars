FROM jupyter/scipy-notebook:python-3.11

LABEL maintainer="TimeWz667 <timewz667@gmail.com>"
LABEL name="sims-pars-dev"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mamba install --quiet --yes \
    'networkx>=3.2' \
    'tqdm>=4.66' \
    'pydantic>=2.7' \
    'joblib>=1.4' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# CPU-only GP emulator stack for the `hme` extra
RUN pip install --no-cache-dir 'torch>=2.9' 'gpytorch>=1.15'
