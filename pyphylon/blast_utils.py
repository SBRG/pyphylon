"""
Functions for interacting with blast and blastDBs from pyphylon
"""

import pandas as pd
import subprocess

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



def extract_reference_sequences(cd_hit_results, species, outfile):
    ''' 
    Extract a file of all of the sequences of representative alleles

    Parameters:
    cd_hit_results - location of cd-hit results
    species - name of the species (for cd-hit file paths
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
    