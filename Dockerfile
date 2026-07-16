FROM snakemake/snakemake:v8.30.0
WORKDIR /workflows
RUN micromamba install -y -n base -c conda-forge less
