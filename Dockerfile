FROM snakemake/snakemake:v7.25.0
WORKDIR /workflow
RUN apt update
RUN apt install less