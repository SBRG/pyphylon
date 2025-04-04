If you have a pre-downloaded database for BAKTA:
docker run --privileged -it   -v $(pwd)/examples:/examples   -v $(pwd)/workflow:/workflow   -v /path/to/database:/db   pyphylon

snakemake -d /examples/data --use-singularity -c 16  --singularity-args="-B /db:/db"

Otherwise, use default commnads from the README in examples. 

Additionally, the Dockerfile used to create the Docker Image for the creation of the GFF files from Panaroo is contained in this directory (Panaroo version 1.5.2). 