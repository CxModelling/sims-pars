FROM pymc/pymc:4.1.5
# Adapted from pymc/pymc
LABEL name="acf-dev"
LABEL description="Environment for ACF model with PyMC version 4"


COPY /environment-dev.yml .
RUN conda env update -f environment-dev.yml -n pymc-dev

# For running from jupyter notebook
EXPOSE 1167
CMD ["conda", "run", "--no-capture-output", "-n", "pymc-dev", "jupyter","notebook","--ip=0.0.0.0","--port=1167","--no-browser"]