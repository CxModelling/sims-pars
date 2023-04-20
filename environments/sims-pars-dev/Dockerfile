FROM jupyter/tensorflow-notebook:python-3.10.9

LABEL maintainer="TimeWz667 <timewz667@gmail.com>"
LABEL name="sims-pars-dev"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mamba install --quiet --yes \
    'astunparse>=1.6.3' \
    'networkx>=2.5' \
    'tqdm>=4.56.0' \
    'pydantic' \
    'gpflow>=2' \
    'joblib>=1.0.1' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" \
