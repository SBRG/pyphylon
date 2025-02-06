FROM snakemake/snakemake:v7.25.0
WORKDIR /workflows
RUN apt update
RUN apt install less