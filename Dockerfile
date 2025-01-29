FROM snakemake/snakemake:v7.25.0
WORKDIR /data
RUN apt update
RUN apt install less