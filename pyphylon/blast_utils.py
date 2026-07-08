"""
Functions for interacting with blast and blastDBs from pyphylon
"""

import pandas as pd
import subprocess
from tqdm import tqdm
import os

def make_blast_db(fasta_file, output_location, dbtype = 'prot'):
    """
    Calls makeblastdb from command line blast on a given fasta file

    Parameters:
    fasta_file - file to make database from
    output_location - location to place files for blast database and the name of the files

    Keyword parameters:
    dbtype - type of database to create (nucl or prot)
    silent - Output progress messages
    """
    command = ["makeblastdb", '-dbtype', dbtype, '-in', fasta_file, '-out', output_location]
    subprocess.run(command)

    print("Finished running, database created at " + output_location)
            

def extract_reference_sequences(WORKDIR, species, outfile, PANAROO = False):
    ''' 
    Extract a file of all of the sequences of representative alleles

    Parameters:
    WORKDIR - location of work directory
    species - name of the species (for cd-hit file paths)
    outfile - file to write output to
    PANAROO - setting to use panaroo method over cd-hit method
    '''

    if PANAROO:
        extract_reference_sequences_panaroo(os.path.join(WORKDIR + f'processed/panaroo_results/{species}/'), species, outfile)
    else:
        extract_reference_sequences_cdhit(os.path.join(WORKDIR + 'processed/cd-hit-results'), species, outfile)


def extract_reference_sequences_panaroo(panaroo_results, species, outfile):
    ''' 
    Extract a file of all of the sequences of representative alleles

    Parameters:
    panaroo_results - location of panaroo results
    species - name of the species (for cd-hit file paths)
    outfile - file to write output to
    '''
    import networkx as nx

    graph = nx.read_gml(os.path.join(panaroo_results, "final_graph.gml"))

    with open(outfile, "w") as out:
        for _, attributes in graph.nodes.data(True):
            gene_name = attributes["name"]

            # Protein sequences are stored as a semicolon-separated string
            proteins = attributes["protein"].split(";")

            # Select the longest non-fragmented protein sequence
            representative = None
            max_len = -1

            for protein in proteins:
                if "*" in protein:
                    continue

                if len(protein) > max_len:
                    representative = protein
                    max_len = len(protein)

            if representative is None:
                raise ValueError(
                    f"No valid protein sequence found for Panaroo gene cluster '{gene_name}'."
                )

            out.write(f">{gene_name}\n")
            out.write(f"{representative}\n")

    
def extract_reference_sequences_cdhit(cd_hit_results, species, outfile):
    ''' 
    Extract a file of all of the sequences of representative alleles

    Parameters:
    cd_hit_results - location of cd-hit results
    species - name of the species (for cd-hit file paths)
    outfile - file to write output to
    '''
    # create dictionary of headers to alleles
    alleles_to_headers = {}
    with open(cd_hit_results + '/' + species + '_allele_names.tsv', 'r') as f:
        for line in f:
            line = line.split()
            allele = line[0]
            for header in line[1:]:
                alleles_to_headers[header] = allele
    
    representative_alleles = []
    with open(cd_hit_results + '/' + species + '.clstr', 'r') as f:
        for line in f:
            if '*' in line:
                seq = line.split()[2][1:-3]
                representative_alleles.append(alleles_to_headers[seq])
    
    active_allele = None
    with open(cd_hit_results + '/' + species, 'r') as f:
        with open(outfile, 'w') as out:
            for line in f:
                if '>' in line:
                    active_allele = line[1:-1]
                if active_allele in representative_alleles:
                    out.write(line)


def extract_reference_dna_sequences(WORKDIR, species, outfile, PANAROO=False):
    '''
    Extract a file of representative DNA sequences.

    Parameters:
    WORKDIR - location of work directory
    species - species name
    outfile - output fasta file
    PANAROO - use Panaroo rather than CD-HIT
    '''

    if PANAROO:
        extract_reference_dna_sequences_panaroo(
            os.path.join(WORKDIR, "processed", "panaroo_results", species),
            species,
            outfile,
        )
    else:
        extract_reference_dna_sequences_cdhit(
            WORKDIR,
            species,
            outfile,
        )


import networkx as nx


def extract_reference_dna_sequences_panaroo(panaroo_results, species, outfile):
    '''
    Extract representative DNA sequences from a Panaroo graph.

    Parameters:
    panaroo_results - location of Panaroo results
    species - unused (included for API compatibility)
    outfile - output fasta file
    '''

    graph = nx.read_gml(os.path.join(panaroo_results, "final_graph.gml"))

    with open(outfile, "w") as out:
        for _, attributes in graph.nodes.data(True):
            gene_name = attributes["name"]
            dna_seq = attributes["dna"]

            if not dna_seq:
                raise ValueError(
                    f"No DNA sequence found for Panaroo gene cluster '{gene_name}'."
                )

            out.write(f">{gene_name}\n")
            out.write(f"{dna_seq}\n")


def extract_reference_dna_sequences_cdhit(data_path, species, outfile):
    ''' 
    Extract a file of all of the DNA sequences of representative alleles

    Parameters:
    data_path - location of pangenome results to access cd-hit results and bakta files
    species - name of the species (for cd-hit file paths)
    outfile - file to write output to
    '''
    # create dictionary of headers to alleles
    alleles_to_headers = {}
    with open(data_path + 'processed/cd-hit-results' + '/' + species + '_allele_names.tsv', 'r') as f:
        for line in f:
            line = line.split()
            allele = line[0]
            for header in line[1:]:
                alleles_to_headers[header] = allele

    
    representative_headers = []
    with open(data_path + 'processed/cd-hit-results' + '/' + species + '.clstr', 'r') as f:
        for line in f:
            if '*' in line:
                seq = line.split()[2][1:-3]
                representative_headers.append(seq)
    
    with open(outfile, 'w') as out:
        for seq in tqdm(representative_headers):
            genome = seq.split('_')[0]
            with open(data_path + 'processed/bakta' + '/' + genome + '/' + genome + '.ffn', 'r') as f:
                active_header = False
                for line in f:
                    if '>' in line and line.split()[0][1:] == seq:
                        active_header = True
                        out.write('>' + alleles_to_headers[line.split()[0][1:]] + '\n')
                    elif active_header:
                        if '>' in line:
                            break
                        out.write(line)
                        
                        
    
            
def blast_localdb_enrichment(blastdb, query_file, output_file, input_type = 'prot', dbtype = 'prot', e_val = 1e-2):
    '''
    Blasts all sequences from pangenome against a local blastdb. Output formmat is in format 

    Parameters:
    blastdb - location of blastdb to blast against
    query_file - file to blast against the database
    output_file - location of file to output results to 

    Keyword parameters:
    input_type - type of blast to perform on the database
    dbtype - type of data in blastdb to compare against
    '''
    command = []

    if input_type == 'prot' and  dbtype == 'prot':
        command.append('blastp')
    elif input_type == 'nucl' and  dbtype == 'nucl':
        command.append('blastn')
    elif input_type == 'nucl' and  dbtype == 'prot':
        command.append('blastx')
    elif input_type == 'prot' and  dbtype == 'nucl':
        command.append('tblastn')
    else:
        raise Exception("Provide valid input and database types (prot and nucl)")

    command.append('-query')
    command.append(query_file)

    command.append('-out')
    command.append(output_file)

    command.append('-db')
    command.append(blastdb)

    # set output format to 6 for downstream processing
    command.append('-outfmt')
    command.append('6')

    # set threshold for E-value
    command.append('-evalue')
    command.append(str(e_val))

    print('Command: ', ' '.join(command))
    
    subprocess.run(command)
    print('Completed blast')



def process_blast_results(file, e_val = 10, percent_identity = 0, unique = True):
    '''
    Filters output format 6 blast results based on several filters

    Parameters:
    file - path to file of results

    Keyword parameters:
    e_val - max evalue to include
    percent_identity - minimum percent identity to filter
    unique - used to only include best hit (lowest e-value) for each query
    '''
    blast_results = pd.read_csv(file, sep='\t', header = None)

    columns = ['query', 'target', 'identity', 'len', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'e_val', 'bitscore']    
    blast_results.columns = columns

    # filter the results
    blast_results = blast_results[(blast_results.identity > percent_identity) & (blast_results.e_val < e_val)]

    if unique:
        blast_results = blast_results.sort_values(by = ['query', 'e_val'], ascending = [True, True])
        blast_results = blast_results.drop_duplicates(subset = 'query', keep = 'first')

    return blast_results
    