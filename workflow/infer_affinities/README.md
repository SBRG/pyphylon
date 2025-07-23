# Instructions for Running Infer Affinities Functionality
## Steps to complete prior to running the workflow:
Please complete all workflows and notebooks through 5b prior to running the infer affinities functionality, it relies on previous steps for creation of the phylon structure to compare against. 

## How to run:
1. From the main directory for this package, build the container from the Dockerfile with: `docker build -t pyphylon .` if it was not previously built.
2. Run the container interactively with:
   - For windows: `docker run --privileged -it -v %cd%/examples:/examples -v %cd%/workflow:/workflow pyphylon`
   - Command for WSL2/Linux: `docker run --privileged -it -v $(pwd)/examples:/examples -v $(pwd)/workflow:/workflow pyphylon`
3. From INSIDE the container cd to /workflow (`cd workflow`)
4. for the MASH workflow cd to the "infer_affinities" folder: `cd infer_affinities`
5. Run snakemake with: `snakemake -d /examples/data --use-singularity -c 10`
6. This will generate results in `/examples/data/infer_affinities/`
7. Run notebook 5f to analyze the output of this workflow and the assignment of you input strains to the phylon structure. 