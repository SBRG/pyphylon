'''
Functions for building a pangenome for phylon analysis.
'''

import os
import pickle

import pandas as pd
from tqdm.notebook import tqdm

###################################
#              CD-HIT             #
###################################
"""
Pan-genome construction tools including consolidating redundant sequences,
gene sequence cluster identification by CD-Hit, and constructing gene/allele tables.
Refer to build_cds_pangenome() and build_upstream_pangenome().

Pan-genome feature nomenclature:
<name>_C# = Gene/CDS cluster
<name>_C# A# = Gene/CDS sequence variant "allele"
<name>_C# U# = Gene/CDS 5'UTR variant "upstream"
<name>_C# D# = Gene/CDS 3'UTR variant "downstream"
<name>_T# = Non-coding transcript/RNA cluster
<name>_T# A# = Noncoding transcript/RNA sequence variant"""

import os, shutil, urllib.request, urllib.parse, urllib.error
import subprocess as sp
import hashlib 
import collections

import pandas as pd
import numpy as np
import scipy.sparse

from tqdm.notebook import tqdm # added for progress bar

CLUSTER_TYPES = {'cds':'C', 'noncoding':'T'}
VARIANT_TYPES = {'allele':'A', 'upstream':'U', 'downstream':'D'}
CLUSTER_TYPES_REV = {v:k for k,v in list(CLUSTER_TYPES.items())}
VARIANT_TYPES_REV = {v:k for k,v in list(VARIANT_TYPES.items())}
DNA_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
              'W': 'W', 'S': 'S', 'R': 'Y', 'Y': 'R', 
              'M': 'K', 'K': 'M', 'N': 'N'}
for bp in list(DNA_COMPLEMENT.keys()):
    DNA_COMPLEMENT[bp.lower()] = DNA_COMPLEMENT[bp].lower()

    
def build_cds_pangenome(genome_faa_paths, output_dir, name='Test', 
                        cdhit_args={'-n':5, '-c':0.8}, 
                        fastasort_path=None, save_csv=True):
    '''
    Constructs a pan-genome based on cds protein sequences.

    Constructs a pan-genome based on protein sequences with the following steps:
    1) Merge FAA files for genomes of interest into a non-redundant list
    2) Cluster CDS by sequence into putative genes using CD-Hit
    3) Rename non-redundant CDS as <name>_C# A# , referring to cluster and allele number
    4) Compile allele/gene membership into binary allele x genome and gene x genome tables
    
    Generates eight files within output_dir:
    1) <name>_strain_by_allele.pickle.gz, binary allele x genome table with SparseArray structure
    2) <name>_strain_by_gene.pickle.gz, binary gene x genome table with SparseArray structure
    1) <name>_strain_by_allele.csv.gz, binary allele x genome table as flat file (if save_csv)
    2) <name>_strain_by_gene.csv.gz, binary gene x genome table as flat file (if save_csv)
    3) <name>_nr.faa, all non-redundant CDSs observed, with headers <name>_C# A#
    4) <name>_nr.faa.cdhit.clstr, CD-Hit output file from clustering
    5) <name>_allele_names.tsv, mapping between <name>_C# A# to original CDS headers
    6) <name>_redundant_headers.tsv, lists of headers sharing the same CDS, with the
        representative header relevant to # 5 listed first for each group.
    7) <name>_missing_headers.txt, lists headers for original entries missing sequences
    
    Parameters
    ----------
    genome_faa_paths : list 
        FAA files containing CDSs for genomes of interest. Genome 
        names are inferred from these FAA file paths.
    output_dir : str
        Path to directory to generate outputs and intermediates.
    name : str
        Header to prepend to all output files and allele names (default 'Test')
    cdhit_args : dict
        Alignment arguments to pass CD-Hit, other than -i, -o, and -d
        (default {'-n':5, '-c':0.8})
    fastasort_path : str
        Path to Exonerate's fastasort binary, optionally for sorting
        final FAA files (default None)
    save_csv : bool
        If true, saves allele and gene tables as csv.gz. May be limiting
        step for very large tables (default True)
        
    Returns 
    -------
    df_alleles : pd.DataFrame
        Binary allele x genome table
    df_genes : pd.DataFrame
        Binary gene x genome table
    '''
    
    # Merge FAAs into one file with non-redundant sequences
    print('Identifying non-redundant CDS sequences...')
    output_nr_faa = output_dir + '/' + name + '_nr.faa' # final non-redundant FAA files
    output_shared_headers = output_dir + '/' + name + '_redundant_headers.tsv' # records headers that have the same sequence
    output_missing_headers = output_dir + '/' + name + '_missing_headers.txt' # records headers without any seqeunce
    output_nr_faa = output_nr_faa.replace('//','/')
    output_shared_headers = output_shared_headers.replace('//','/')
    output_missing_headers = output_missing_headers.replace('//','/')
    
    # Maps sequence hash to headers of that sequence, in order observed
    non_redundant_seq_hashes, missing_headers = consolidate_seqs(
        genome_faa_paths, output_nr_faa, output_shared_headers, output_missing_headers)
    
    # Apply CD-Hit to non-redundant CDS sequences
    output_nr_faa_copy = output_nr_faa + '.cdhit' # temporary FAA copy generated by CD-Hit
    output_nr_clstr = output_nr_faa + '.cdhit.clstr' # clusteeipxzr file generated by CD-Hit
    cluster_with_cdhit(output_nr_faa, output_nr_faa_copy, cdhit_args)
    os.remove(output_nr_faa_copy) # delete CD-hit copied sequences
    
    # Extract genes and alleles, rename unique sequences as <name>_C# A#
    output_allele_names = output_dir + '/' + name + '_allele_names.tsv' # allele names vs non-redundant headers
    output_allele_names = output_allele_names.replace('//','/')
    
    # Maps original headers to short names <name>_C# A# 
    header_to_allele = rename_genes_and_alleles(
        output_nr_clstr, output_nr_faa, output_nr_faa, 
        output_allele_names, name=name, cluster_type='cds',
        shared_headers_file=output_shared_headers,
        fastasort_path=fastasort_path)
    
    # Process gene/allele membership into binary tables
    df_alleles, df_genes = build_genetic_feature_tables(
        output_nr_clstr, genome_faa_paths, name,
        cluster_type='cds', header_to_allele=header_to_allele)
    
    # Save tables as PICKLE.GZ (preserve SparseArrays) and CSV.GZ (backup flat file)
    output_allele_table = output_dir + '/' + name + '_strain_by_allele'
    output_gene_table = output_dir + '/' + name + '_strain_by_gene'
    output_allele_table = output_allele_table.replace('//','/')
    output_gene_table = output_gene_table.replace('//','/')
    output_allele_csv = output_allele_table + '.csv.gz'
    output_gene_csv = output_gene_table + '.csv.gz'
    output_allele_pickle = output_allele_table + '.pickle.gz'
    output_gene_pickle = output_gene_table + '.pickle.gz'
    
    print('Saving', output_allele_pickle, '...')
    
    df_alleles.to_pickle(output_allele_pickle)
    
    print('Saving', output_gene_pickle, '...')
    df_genes.to_pickle(output_gene_pickle)
    
    if save_csv:
        print('Saving', output_allele_csv, '...')
        df_alleles.to_csv(output_allele_csv)
        print('Saving', output_gene_csv, '...')
        df_genes.to_csv(output_gene_csv)

    return df_alleles, df_genes, header_to_allele


def build_cds_nucl_pangenome(genome_data, output_dir, name='Test',
                             allowed_features=['CDS', 'tRNA'],
                             cdhit_args={'-n': 5, '-c':0.8}, fastasort_path=None,
                             save_csv=True):
    '''
    Constructs a pan-genome based on coding nucleic acid sequences.

    Constructs a pan-genome based on coding nucleic acid sequences with the following steps:
    1) Extract coding transcripts based on FNA/GFF pairs
    2) Cluster CDS by sequence into putative transcripts using CD-HIT-EST
    3) Rename non-redundant transcript as <name>_T# A# , referring to transcript cluster and allele number
    4) Compile allele/transcript membership into binary transcript allele x genome and transcript x genome tables

    Generates eight files within output_dir:
    1) <name>_strain_by_coding_nuc_allele.pickle.gz, binary allele x genome table with SparseArray structure
    2) <name>_strain_by_coding_nuc.pickle.gz, binary gene x genome table with SparseArray structure
    1) <name>_strain_by_coding_nuc_allele.csv.gz, binary allele x genome table as flat file (if save_csv)
    2) <name>_strain_by_coding_nuc.csv.gz, binary gene x genome table as flat file (if save_csv)
    3) <name>_coding_nuc_nr.fna, all non-redundant coding seqs observed, with headers <name>_T# A# 4) <name>_coding_nuc_nr.fna.cdhit.clstr, CD-HIT-EST output file from clustering
    5) <name>_coding_nuc_allele_names.tsv, mapping between <name>_T# A# to original transcript headers
    6) <name>_coding_nuc_redundant_headers.tsv, lists of headers sharing the same sequences, with the
        representative header relevant to # 5 listed first for each group.
    7) <name>_coding_missing_headers.txt, lists headers for original entries missing sequences

    Parameters
    ----------
    genome_data : list
        List of 2-tuples (genome_gff, genome_fna) for use by extract_coding()
    output_dir : str
        Path to directory to generate outputs and intermediates.
    name : str
        Header to prepend to all output files and allele names (default 'Test')
    flanking : tuple
        (X,Y) where X = number of nts to include from 5' end of feature,
        and Y = number of nts to include from 3' end feature. Features
        may be truncated by contig boundaries (default (0,0))
    allowed_features : list
        List of GFF feature types to extract. Default includes
        features labeled "CDS" and "tRNA"
    cdhit_args : dict
        Alignment arguments to pass CD-HIT-EST, other than -i, -o, and -d
        (default {'-n':5, '-c':0.8})
    fastasort_path : str
        Path to Exonerate's fastasort binary, optionally for sorting
        final FAA files (default None)
    save_csv : bool
        If true, saves allele and gene tables as csv.gz. May be limiting
        step for very large tables (default True)

    Returns
    -------
    df_nuc_alleles : pd.DataFrame
        Binary non-coding allele x genome table
    df_nuc_genes : pd.DataFrame
        Binary non-coding gene x genome table
    '''

    ''' Extract coding nucleid acid sequences from all genomes '''
    print('Identifying non-redundant CDS sequences...')
    output_nr_nuc_fna = output_dir + '/' + name + '_coding_nuc_nr.fna' # final non-redundant
    output_shared_headers = output_dir + '/' + name + '_coding_nuc_redundant_headers.tsv' # records headers that have the same sequence
    output_missing_headers = output_dir + '/' + name + '_coding_nuc_missing_headers.txt' # records headers without any seqeunce
    output_nr_nuc_fna = output_nr_nuc_fna.replace('//','/')
    output_shared_headers = output_shared_headers.replace('//','/')
    output_missing_headers = output_missing_headers.replace('//','/')

    # before we consolidate the sequences, we need to generate the coding fna files
    print('Extracting coding-sequences...')
    genome_coding_paths = []
    for i, gff_fna in enumerate(genome_data):
        '''Prepare output path'''
        genome_gff, genome_fna = gff_fna
        genome = __get_genome_from_filename__(genome_gff)
        genome_dir = '/'.join(genome_gff.split('/')[:-1]) + '/' if '/' in genome_gff else ''
        genome_nuc_dir = genome_dir + 'derived/'  # output coding sequences here
        if not os.path.exists(genome_nuc_dir):
            os.mkdir(genome_nuc_dir)
        genome_nuc = genome_nuc_dir + genome + '_nuc_coding.fna'
        '''Extract non-coding sequnces'''
        print(i+1, genome)
        genome_coding_paths.append(genome_nuc)
        extract_coding_fna(genome_gff, genome_fna, genome_nuc,
                          allowed_features=allowed_features)

    # consolidate redundant sequences
    non_redundant_seq_hashes, missing_headers = consolidate_seqs(
    genome_coding_paths, output_nr_nuc_fna, output_shared_headers, output_missing_headers)

    ''' Apply CD-Hit to non-redundant CDS sequences '''
    output_nr_faa_copy = output_nr_nuc_fna + '.cdhit'  # temporary FAA copy generated by CD-Hit
    output_nr_clstr = output_nr_nuc_fna + '.cdhit.clstr'  # cluster file generated by CD-Hit
    cluster_with_cdhit(output_nr_nuc_fna, output_nr_faa_copy, cdhit_args)
    os.remove(output_nr_faa_copy)  # delete CD-hit copied sequences

    ''' Extract genes and alleles, rename unique sequences as <name>_C# A# '''
    output_allele_names = output_dir + '/' + name + '_allele_names.tsv'  # allele names vs non-redundant headers
    output_allele_names = output_allele_names.replace('//', '/')
    header_to_allele = rename_genes_and_alleles(
        output_nr_clstr, output_nr_nuc_fna, output_nr_nuc_fna,
        output_allele_names, name=name, cluster_type='cds',
        shared_headers_file=output_shared_headers,
        fastasort_path=fastasort_path)
    # maps original headers to short names <name>_C# A# ''' Process gene/allele membership into binary tables '''
    df_alleles, df_genes = build_genetic_feature_tables(
        output_nr_clstr, genome_coding_paths, name,
        cluster_type='cds', header_to_allele=header_to_allele)

    ''' Save tables as PICKLE.GZ (preserve SparseArrays) and CSV.GZ (backup flat file) '''
    output_allele_table = output_dir + '/' + name + '_strain_by_allele'
    output_gene_table = output_dir + '/' + name + '_strain_by_gene'
    output_allele_table = output_allele_table.replace('//', '/')
    output_gene_table = output_gene_table.replace('//', '/')
    output_allele_csv = output_allele_table + '.csv.gz'
    output_gene_csv = output_gene_table + '.csv.gz'
    output_allele_pickle = output_allele_table + '.pickle.gz'
    output_gene_pickle = output_gene_table + '.pickle.gz'
    
    print('Saving', output_allele_pickle, '...')
    df_alleles.to_pickle(output_allele_pickle)
    
    print('Saving', output_gene_pickle, '...')
    df_genes.to_pickle(output_gene_pickle)
    
    if save_csv:
        print('Saving', output_allele_csv, '...')
        df_alleles.to_csv(output_allele_csv)
        print('Saving', output_gene_csv, '...')
        df_genes.to_csv(output_gene_csv)

    return df_alleles, df_genes, header_to_allele


def build_noncoding_pangenome(genome_data, output_dir, name='Test', flanking=(0,0),
                              allowed_features=['transcript', 'tRNA', 'rRNA', 'misc_binding'],
                              cdhit_args={'-n':5, '-c':0.8}, fastasort_path=None, save_csv=True):
    ''' 
    Constructs a pan-genome based on noncoding sequences.

    Constructs a pan-genome based on noncoding sequences with the following steps:
    1) Extract non-coding transcripts (optionally with flanking NTs) based on FNA/GFF pairs
    2) Cluster CDS by sequence into putative transcripts using CD-HIT-EST
    3) Rename non-redundant transcript as <name>_T# A# , referring to transcript cluster and allele number
    4) Compile allele/transcript membership into binary transcript allele x genome and transcript x genome tables
    
    Generates eight files within output_dir:
    1) <name>_strain_by_noncoding_allele.pickle.gz, binary allele x genome table with SparseArray structure
    2) <name>_strain_by_noncoding.pickle.gz, binary gene x genome table with SparseArray structure
    1) <name>_strain_by_noncoding_allele.csv.gz, binary allele x genome table as flat file (if save_csv)
    2) <name>_strain_by_noncoding.csv.gz, binary gene x genome table as flat file (if save_csv)
    3) <name>_noncoding_nr.fna, all non-redundant non-coding seqs observed, with headers <name>_T# A# 4) <name>_noncoding_nr.fna.cdhit.clstr, CD-HIT-EST output file from clustering
    5) <name>_noncoding_allele_names.tsv, mapping between <name>_T# A# to original transcript headers
    6) <name>_noncoding_redundant_headers.tsv, lists of headers sharing the same sequences, with the
        representative header relevant to # 5 listed first for each group.
    7) <name>_noncoding_missing_headers.txt, lists headers for original entries missing sequences
    
    Parameters
    ----------
    genome_data : list
        List of 2-tuples (genome_gff, genome_fna) for use by extract_noncoding()
    output_dir : str
        Path to directory to generate outputs and intermediates.
    name : str
        Header to prepend to all output files and allele names (default 'Test')
    flanking : tuple
        (X,Y) where X = number of nts to include from 5' end of feature,
        and Y = number of nts to include from 3' end feature. Features
        may be truncated by contig boundaries (default (0,0))
    allowed_features : list
        List of GFF feature types to extract. Default excludes 
        features labeled "CDS" or "repeat_region" 
        (default ['transcript', 'tRNA', 'rRNA', 'misc_binding'])
    cdhit_args : dict
        Alignment arguments to pass CD-HIT-EST, other than -i, -o, and -d
        (default {'-n':5, '-c':0.8})
    fastasort_path : str
        Path to Exonerate's fastasort binary, optionally for sorting
        final FAA files (default None)
    save_csv : bool
        If true, saves allele and gene tables as csv.gz. May be limiting
        step for very large tables (default True)
        
    Returns 
    -------
    df_nc_alleles : pd.DataFrame
        Binary non-coding allele x genome table
    df_nc_genes : pd.DataFrame
        Binary non-coding gene x genome table
    '''
    
    ''' Extract non-coding sequences from all genomes '''
    print('Extracting non-coding sequences...')
    genome_noncoding_paths = []
    for i, gff_fna in enumerate(genome_data):
        ''' Prepare output path '''
        genome_gff, genome_fna = gff_fna
        genome = __get_genome_from_filename__(genome_gff)
        genome_dir = '/'.join(genome_gff.split('/')[:-1]) + '/' if '/' in genome_gff else ''
        genome_nc_dir = genome_dir + 'derived/' # output noncoding sequences here
        if not os.path.exists(genome_nc_dir):
            os.mkdir(genome_nc_dir)
        genome_nc = genome_nc_dir + genome + '_noncoding.fna'
            
        ''' Extract non-coding sequences '''
        print(i+1, genome)
        genome_noncoding_paths.append(genome_nc)
        extract_noncoding(genome_gff, genome_fna, genome_nc, 
            flanking=flanking, allowed_features=allowed_features)
        
    ''' Reduce to non-redundant sequence set '''
    print('Identifying non-redundant non-coding sequences...')
    output_nr_fna = output_dir + '/' + name + '_noncoding_nr.fna' # final non-redundant FNA files
    output_shared_headers = output_dir + '/' + name + '_noncoding_redundant_headers.tsv' 
        # records headers that have the same sequence
    output_missing_headers = output_dir + '/' + name + '_noncoding_missing_headers.txt' 
        # records headers without any seqeunce
    output_nr_fna = output_nr_fna.replace('//','/')
    output_shared_headers = output_shared_headers.replace('//','/')
    output_missing_headers = output_missing_headers.replace('//','/')
    nr_seq_hashes, missing_headers = consolidate_seqs(
        genome_noncoding_paths, output_nr_fna, 
        output_shared_headers, output_missing_headers)
    # maps sequence hash to headers of that sequence, in order observed
    ''' Apply CD-Hit to non-redundant non-coding sequences '''
    output_nr_fna_copy = output_nr_fna + '.cdhit' # temporary FNA copy generated by CD-HIT-EST
    output_nr_clstr = output_nr_fna + '.cdhit.clstr' # cluster file generated by CD-HIT-EST
    cluster_with_cdhit(output_nr_fna, output_nr_fna_copy, cdhit_args)
    os.remove(output_nr_fna_copy) # delete CD-HIT-EST copied sequences
    
    ''' Extract genes and alleles, rename unique sequences as <name>_T# A# '''
    output_allele_names = output_dir + '/' + name + '_noncoding_allele_names.tsv' # allele names vs non-redundant headers
    output_allele_names = output_allele_names.replace('//','/')
    header_to_allele = rename_genes_and_alleles(
        output_nr_clstr, output_nr_fna, output_nr_fna, 
        output_allele_names, name=name, cluster_type='noncoding',
        shared_headers_file=output_shared_headers,
        fastasort_path=fastasort_path)
    # maps original headers to short names <name>_T# A# ''' Process gene/allele membership into binary tables '''    
    df_nc_alleles, df_nc_genes = build_genetic_feature_tables(
        output_nr_clstr, genome_noncoding_paths, name, 
        cluster_type='noncoding', header_to_allele=header_to_allele)
    df_nc_alleles.columns = df_nc_alleles.columns.map(lambda x: x.replace('_noncoding',''))
    df_nc_genes.columns = df_nc_genes.columns.map(lambda x: x.replace('_noncoding','')) 
    
    ''' Save tables as PICKLE.GZ (preserve SparseArrays) and CSV.GZ (backup flat file) '''
    output_allele_table = output_dir + '/' + name + '_strain_by_noncoding_allele'
    output_gene_table = output_dir + '/' + name + '_strain_by_noncoding_gene'
    output_allele_table = output_allele_table.replace('//','/')
    output_gene_table = output_gene_table.replace('//','/')
    output_allele_csv = output_allele_table + '.csv.gz'
    output_gene_csv = output_gene_table + '.csv.gz'
    output_allele_pickle = output_allele_table + '.pickle.gz'
    output_gene_pickle = output_gene_table + '.pickle.gz'
    print('Saving', output_allele_pickle, '...')
    df_nc_alleles.to_pickle(output_allele_pickle)
    print('Saving', output_gene_pickle, '...')
    df_nc_genes.to_pickle(output_gene_pickle)
    if save_csv:
        print('Saving', output_allele_csv, '...')
        df_nc_alleles.to_csv(output_allele_csv)
        print('Saving', output_gene_csv, '...')
        df_nc_genes.to_csv(output_gene_csv)
    
    return df_nc_alleles, df_nc_genes, header_to_allele
    

def consolidate_seqs(genome_paths, nr_out, shared_headers_out, missing_headers_out=None):
    ''' 
    Combines sequences for many genomes into a single file without duplicate
    sequences to be clustered using CD-Hit, i.e. with cluster_with_cdhit(). Tracks
    headers that share the same sequence, and optionally headers without sequences.
    
    Parameters
    ----------
    genome_paths : list
        Paths to genome FAA/FNA files to combine
    nr_out : str
        Output path for combined non-redundant FAA/FNA file
    shared_headers_out : str
        Output path for shared headers TSV file
    missing_headers_out : str
        Output path for headers without sequences TXT file (default None)

    Returns
    -------
    non_redundant_seq_hashes : dict
        Maps non-redundant sequence hashes to a list of headers, in order observed
    missing_headers : list
        List of headers without any associated sequence
    '''
    non_redundant_seq_hashes = {} # maps sequence hash to headers of that sequence, in order observed
    encounter_order = [] # stores sequence hashes in order encountered
    missing_headers = [] # stores headers without sequences
    
    def process_header_and_seq(header, seq_blocks, output_file):
        ''' Processes a header/sequence pair against the running list of non-redundant sequences '''
        seq = ''.join(seq_blocks)
        seq = seq.encode('utf-8')
        if len(header) > 0 and len(seq) > 0: # valid header-sequence record
            seqhash = __hash_sequence__(seq)
            if seqhash in non_redundant_seq_hashes: # record repeated appearances of sequence
                non_redundant_seq_hashes[seqhash].append(header)
            else: # first encounter of a sequence, record to non-redundant file
                encounter_order.append(seqhash)
                non_redundant_seq_hashes[seqhash] = [header]
                output_file.write('>' + header + '\n')
                output_file.write('\n'.join(seq_blocks) + '\n')
        elif len(header) > 0 and len(seq) == 0: # header without sequence
            missing_headers.append(header)
    
    ''' Scan for redundant sequences across all files, build non-redundant file '''
    with open(nr_out, 'w+') as f_nr_out:
        for genome_path in genome_paths:
            with open(genome_path, 'r') as f:
                header = ''; seq_blocks = []
                for line in f:
                    if line[0] == '>': # header encountered
                        process_header_and_seq(header, seq_blocks, f_nr_out)
                        header = __get_header_from_fasta_line__(line)
                        seq_blocks = []
                    else: # sequence line encountered
                        seq_blocks.append(line.strip())
                process_header_and_seq(header, seq_blocks, f_nr_out) # process last record
                
    ''' Save shared and missing headers to file '''
    with open(shared_headers_out, 'w+') as f_header_out:
        for seqhash in encounter_order:
            headers = non_redundant_seq_hashes[seqhash]
            if len(headers) > 1:
                f_header_out.write('\t'.join(headers) + '\n')
    if missing_headers_out:
        print('Headers without sequences:', len(missing_headers))
        with open(missing_headers_out, 'w+') as f_header_out:
            for header in missing_headers:
                f_header_out.write(header + '\n')
                
    return non_redundant_seq_hashes, missing_headers
        
                
def cluster_with_cdhit(fasta_file, cdhit_out, cdhit_args={'-n':5, '-c':0.8}):
    '''
    Runs CD-Hit on a fasta file, i.e. one generated by consolidate_seqs().
    Requires CD-Hit to be available in PATH. Uses CD-HIT for FAA files (default),
    and CD-HIT-EST for FNA files. 
    
    For CD-HIT-EST, word size "-n" has a more direct impact of similarity threshold
    See https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide# CDHITEST
    
    Parameters
    ----------
    fasta_file : str
        Path to FAA/FNA file to be clustered, i.e. from consolidate_seqs()
    cdhit_out : str
        Path to be provided to CD-Hit output argument
    cdhit_args : dict
        Dictionary of alignment arguments to be provided to CD-Hit, other than
        -i, -o, and -d. Default is for FAA files. (default {'-n':5, '-c':0.8})
    ''' 
    cdhit_prog = 'cd-hit-est' if fasta_file[-4:].lower() == '.fna' else 'cd-hit'
    args = [cdhit_prog, '-i', fasta_file, '-o', cdhit_out, '-d', '0']
    for arg in cdhit_args:
        args += [arg, str(cdhit_args[arg])]
    print('Running:', args)
    for line in __stream_stdout__(' '.join(args)):
        print(line)

        
def rename_genes_and_alleles(clstr_file, nr_fasta_in, nr_fasta_out, 
                             feature_names_out, name='Test', cluster_type='cds',
                             shared_headers_file=None, fastasort_path=None):
    '''
    Processes a CD-Hit CLSTR file (clstr_file) to rename headers in the orignal
    fasta as <name>_C# A# for CDS, or <name>_T# A# for non-coding features,
    based on cluster membership and stores header-name mappings as a TSV.
    
    Can optionally sort final fasta file if fastasort_path is specified, from Exonerate
    https://www.ebi.ac.uk/about/vertebrate-genomics/software/exonerate
    
    Parameters
    ----------
    clstr_file : str
        Path to CLSTR file generated by CD-Hit
    nr_fasta_in : str
        Path to FAA/FNA file corresponding to clstr_file
    nr_fasta_out : str
        Output path for renamed FAA/FNA, will overwrite if equal to nr_fasta_in
    feature_names_out : str
        Output path for header-allele name mapping TSV file
    name : str
        Header to append output files and allele names (default 'Test')
    cluster_type : str
        If 'cds', features are named <name>_C# A# for gene clusters/alleles. 
        If 'noncoding', features are named <name>_T# A# for transcripts (default 'cds')
    shared_headers_file : str
        Path to shared headers. If provided, will expand the header-allele
        mapping to include headers that map to the same sequence/allele (default None)
    fastasort_path : str
        Path to Exonerate fastasort, used to optionally sort nr_faa (default None)
        
    Returns
    -------
    header_to_allele : dict
        Maps original headers to new allele names
    '''
    
    ''' Optionally, load up shared headers '''
    shared_headers = {} # maps representative header to synonym headers
    if shared_headers_file:
        with open(shared_headers_file, 'r') as f_share:
            for line in f_share:
                headers = line.strip().split('\t')
                representative_header = headers[0]
                synonym_headers = headers[1:]
                shared_headers[representative_header] = synonym_headers
    
    ''' Read through CLSTR file to map original headers to C# A# /T# A# names '''
    header_to_allele = {} # maps headers to allele name (name_C# A# or name_T# A# )
    max_cluster = 0 
    with open(feature_names_out, 'w+') as f_naming:
        with open(clstr_file, 'r') as f_clstr:
            for line in tqdm(f_clstr):
                if line[0] == '>': # starting new gene cluster
                    cluster_num = line.split()[-1].strip() # cluster number as string
                    max_cluster = cluster_num
                else: # adding allele to cluster
                    data = line.split()
                    allele_num = int(data[0]) # allele number as string, convert to int
                    allele_header = data[2][1:-3] # old allele header
                    allele_name = create_feature_name(name, cluster_type, cluster_num, 'allele', allele_num)
                    header_to_allele[allele_header] = allele_name
                    mapped_headers = [allele_header]
                    if allele_header in shared_headers: # if synonym headers are available
                        for synonym_header in shared_headers[allele_header]:
                            header_to_allele[synonym_header] = allele_name
                        mapped_headers += shared_headers[allele_header]
                    f_naming.write(allele_name + '\t' + ('\t'.join(mapped_headers)).strip() + '\n')
                    
    ''' Create the fasta file with renamed features '''
    with open(nr_fasta_in, 'r') as f_fasta_old:
        with open(nr_fasta_out + '.tmp', 'w+') as f_fasta_new:
            ''' Iterate through alleles in cluster/allele order '''
            missing = True # if currently in a sequence without a header
            for line in tqdm(f_fasta_old):
                if line[0] == '>': # writing updated header line
                    allele_header = line[1:].strip()
                    if allele_header in header_to_allele:
                        allele_name = header_to_allele[allele_header]
                        f_fasta_new.write('>' + allele_name + '\n')
                        missing = False
                    else:
                        print('MISSING:', allele_header)
                        missing = True
                elif not missing: # writing sequence line
                    f_fasta_new.write(line)
    
    ''' Move fasta file to desired output path '''
    if nr_fasta_out == nr_fasta_in: # if overwriting, remove old faa file
        os.remove(nr_fasta_in) 
    os.rename(nr_fasta_out + '.tmp', nr_fasta_out)
    
    ''' If available, use exonerate.fastasort to sort entries in fasta file '''
    if fastasort_path:
        print('Sorting sequences by header...')
        args = [fastasort_path, nr_fasta_out]
        with open(nr_fasta_out + '.tmp', 'w+') as f_sort:
            # sp.call(args, stdout=f_sort)
            p = sp.Popen(args, stdout=f_sort, stderr=sp.PIPE)
            stdout, stderr = p.communicate()
            print(stderr)
        if p.returncode == 1: # sorting failed
            print('Aborting sort, exitcode', p.returncode)
            os.remove(nr_fasta_out + '.tmp')
        else: # sorting passed (probably)
            os.rename(nr_fasta_out + '.tmp', nr_fasta_out)
    return header_to_allele


def build_genetic_feature_tables(clstr_file, genome_fasta_paths, name='Test', cluster_type='cds',
                                 shared_header_file=None, header_to_allele=None):
    '''
    Builds two binary tables based on the presence/absence of genetic features, 
    allele x genome (allele_table_out) and gene x genome (gene_table_out).
    Works for both CDS clusters and non-coding transcript clusters.
    Uses a CD-Hit CLSTR file, the corresponding original fasta files, 
    and shared header mappings.
    
    Parameters
    ----------
    clstr_file : str
        Path to CD-Hit CLSTR file used to build header-allele mappings
    genome_fasta_paths : list
        Paths to genome fasta files originally combined and clustered (see consolidate_seqs)
    name : str
        Name to attach to features and files (default 'Test')
    cluster_type : str
        If 'cds', features are named <name>_C# A# for gene clusters/alleles. 
        If 'noncoding', features are named <name>_T# A# for transcripts (default 'cds')
    shared_header_file : str
        Path to shared header TSV file, if synonym headers are not mapped
        in header_to_allele or header_to_allele is not provided (default None)
    header_to_allele : dict
        Pre-calculated header-allele mappings corresponding to clstr_file,
        if available from rename_genes_and_alleles() (default None)

    Returns 
    -------
    df_alleles : pd.DataFrame
        Binary allele x genome table
    df_genes : pd.DataFrame
        Binary gene x genome table
    '''
    
    ''' Load header-allele mappings '''
    print('Loadings header-allele mappings...')
    header_to_allele = load_header_to_allele(clstr_file, 
        shared_header_file, header_to_allele, cluster_type)
                    
    ''' Initialize gene and allele tables '''
    genome_order = sorted([__get_genome_from_filename__(x) for x in genome_fasta_paths]) 
        # for genome names, trim .faa from filenames
    print('Sorting alleles...')
    allele_order = sorted(list(set(header_to_allele.values())))
    
    print('Sorting clusters...')
    gene_order = []; last_gene = None
    for allele in tqdm(allele_order):
        gene = __get_gene_from_allele__(allele)
        if gene != last_gene:
            gene_order.append(gene)
            last_gene = gene
    print('Genomes:', len(genome_order))
    print('Clusters:', len(gene_order))
    print('Alleles:', len(allele_order))
    
    ''' To use sparse matrix, map genomes, alleles, and genes to positions '''
    allele_indices = {allele_order[i]:i for i in range(len(allele_order))}
    gene_indices = {gene_order[i]:i for i in range(len(gene_order))}   
    allele_arrays = {} # maps genome:allele vectors as SparseArrays
    gene_arrays = {} # maps genome:gene vector as SparseArrays

    ''' Scan original genome file for allele and gene membership '''
    for i, genome_fasta in enumerate(sorted(genome_fasta_paths)):
        genome = __get_genome_from_filename__(genome_fasta)
        genome_i = genome_order.index(genome)
        allele_arrays[genome] = np.zeros(shape=len(allele_order), dtype='int64')
        gene_arrays[genome] = np.zeros(shape=len(gene_order), dtype='int64')
        with open(genome_fasta, 'r') as f_fasta:
            header = ''; seq = '' # track the sequence to skip over empty sequences
            for line in f_fasta.readlines(): # pre-load, slight speed-up
                ''' Load all alleles and genes per genome '''
                if line[0] == '>': # new header line encountered
                    if len(seq) > 0:
                        if header in header_to_allele:
                            allele_name = header_to_allele[header]
                            allele_i = allele_indices[allele_name]
                            allele_arrays[genome][allele_i] = 1
                            gene = __get_gene_from_allele__(allele_name)
                            gene_i = gene_indices[gene]
                            gene_arrays[genome][gene_i] = 1
                        else:
                            print('MISSING:', header)
                    header = __get_header_from_fasta_line__(line)
                    seq = '' # reset sequence
                else: # sequence line encountered
                    seq += line.strip()
            if len(seq) > 0: # process last record
                if header in header_to_allele:
                    allele_name = header_to_allele[header]
                    allele_i = allele_indices[allele_name]
                    allele_arrays[genome][allele_i] = 1
                    gene = __get_gene_from_allele__(allele_name)
                    gene_i = gene_indices[gene]
                    gene_arrays[genome][gene_i] = 1
                else:
                    print('MISSING:', header)
                
        allele_arrays[genome] = pd.arrays.SparseArray(allele_arrays[genome])
        gene_arrays[genome] = pd.arrays.SparseArray(gene_arrays[genome])
        allele_arrays[genome].fill_value = np.nan
        gene_arrays[genome].fill_value = np.nan
        print('Updating genome', i+1, ':', genome, end=' ') 
        print('\tAlleles:', allele_arrays[genome].sum(), '\tClusters:', gene_arrays[genome].sum())
        
    ''' Construct DataFrame '''
    print('Building DataFrame...')
    df_alleles = pd.DataFrame(data=allele_arrays, index=allele_order)
    df_genes = pd.DataFrame(data=gene_arrays, index=gene_order)
    return df_alleles, df_genes


def load_header_to_allele(clstr_file=None, shared_header_file=None, 
                          header_to_allele=None, name='Test', cluster_type='cds'):
    '''
    Loads a mapping from original fasta headers to allele names format 
    <name>_C# A# for genes or <name>_T# A# for noncoding features.
    
    Parameters
    ----------
    clstr_file : str
        Path to CD-Hit CLSTR file used to build header-allele mappings,
        only needed if header_to_allele is None (default None)
    shared_header_file : str
        Path to shared header TSV file, if synonym headers are not mapped
        in header_to_allele or header_to_allele is not provided (default None)
    header_to_allele : dict
        Pre-calculated header-allele mappings corresponding to clstr_file,
        if available from rename_genes_and_alleles (default None)
    name : str
        Name to attach to features and files (default 'Test')
    cluster_type : str
        If 'cds', features are named <name>_C# A# for gene clusters/alleles. 
        If 'noncoding', features are named <name>_T# A# for transcripts (default 'cds')
        
    Returns
    -------
    full_header_to_allele : dict
        Full header-allele mappings combining contents of both header_to_allele 
        (copied or built from clstr_file) and shared_header_file.
    '''
    
    ''' Load header to allele mapping from CLSTR, if not provided '''
    if header_to_allele is None:
        full_header_to_allele = {} # maps representative header to allele name (name_C# A# )
        with open(clstr_file, 'r') as f_clstr:
            for line in f_clstr:
                if line[0] == '>': # starting new gene cluster
                    cluster_num = line.split()[-1].strip() # cluster number as string
                    max_cluster = cluster_num
                else: # adding allele to cluster
                    data = line.split()
                    allele_num = data[0] # allele number as string
                    allele_header = data[2][1:-3] # old allele header
                    allele_name = create_feature_name(name, cluster_type, cluster_num, 'allele', allele_num)
                    full_header_to_allele[allele_header] = allele_name
    elif type(header_to_allele) == dict:
        full_header_to_allele = header_to_allele.copy()
    
    ''' Load headers that share the same sequence '''
    if shared_header_file:
        with open(shared_header_file, 'r') as f_header:
            for line in f_header:
                headers = [x.strip() for x in line.split('\t')]
                if len(headers) > 1:
                    repr_header = headers[0]
                    repr_allele = full_header_to_allele[repr_header]
                    for alt_header in headers[1:]:
                        full_header_to_allele[alt_header] = repr_allele
    return full_header_to_allele


def build_upstream_pangenome(genome_data, allele_names, output_dir, limits=(-50,3), 
                             name='Test', include_fragments=False, max_overlap=-1, 
                             fastasort_path=None, save_csv=True):
    '''
    Extracts nucleotides upstream of coding sequences for multiple genomes, 
    create <genome>_upstream.fna files in the same directory for each genome.
    Then, classifies/names them relative to gene clusters identified by coding sequence,  
    i.e. after build_cds_pangenome(). See build_proximal_pangenome() for parameters.
    '''
    return build_proximal_pangenome(
        genome_data, allele_names, output_dir, limits, 
        side='upstream', name=name, include_fragments=include_fragments, 
        max_overlap=max_overlap, fastasort_path=fastasort_path, save_csv=save_csv)
    

def build_downstream_pangenome(genome_data, allele_names, output_dir, limits=(-3,50), 
                               name='Test', include_fragments=False, max_overlap=-1, 
                               fastasort_path=None, save_csv=True):
    '''
    Extracts nucleotides downstream of coding sequences for multiple genomes, 
    create <genome>_downstream.fna files in the same directory for each genome.
    Then, classifies/names them relative to gene clusters identified by coding sequence,  
    i.e. after build_cds_pangenome(). See build_proximal_pangenome() for parameters.
    '''
    return build_proximal_pangenome(
        genome_data, allele_names, output_dir, limits, 
        side='downstream', name=name, include_fragments=include_fragments, 
        max_overlap=max_overlap, fastasort_path=fastasort_path, save_csv=save_csv)

    
def build_proximal_pangenome(genome_data, allele_names, output_dir, limits, side, name='Test', 
                             include_fragments=False, max_overlap=-1, fastasort_path=None, save_csv=True):
    '''
    Extracts nucleotides proximal to coding sequences for multiple genomes, 
    create genome-specific proximal sequence fna files in the same directory for each genome.
    Then, classifies/names them relative to gene clusters identified by coding sequence,  
    i.e. after build_cds_pangenome(). See extract_proximal_sequences() and
    consolidate_proximal() for more details.
    
    Parameters
    ----------
    genome_data : list
        List of 2-tuples (genome_gff, genome_fna) for use by extract_proximal_sequences()
    allele_names : str
        Path to allele names file, should be named <name>_allele_names.tsv
    output_dir : str
        Path to directory to generate summary outputs.
    limits : 2-tuple
        Length of proximal region to extract, formatted (-X,Y). For upstream, extracts X 
        upstream bases (up to but excluding first base of start codon) and first Y coding 
        bases (including first base of start codon). For downstream, extracts the last X
        coding bases and Y downstream bases. In both cases, the total length is X+Y bases.
    side : str
        'upstream' or 'downstream' for 5'UTR or 3'UTR
    name : str
        Short header to prepend output summary files, recommendated to be same as what
        was used in the build_cds_pangenome() (default 'Test')
    include_fragments : bool
        If true, include proximal sequences that are not fully available 
        due to contig boundaries (default False)
    max_overlap : int
        If non-negative, truncates UTRs that cross over into coding sequences
        of other genes such that the overlap is no more than <max_overlap> nts.
        If negative, does not truncate UTRs s.t. all UTRs same length (default -1)
    fastasort_path : str
        Path to Exonerate's fastasort binary, optionally for sorting
        final FNA files (default None)
    save_csv : bool
        If true, saves allele and gene tables as csv.gz. May be limiting
        step for very large tables (default True)
        
    Returns
    -------
    df_proximal : pd.DataFrame
        Binary proximal x genome table
    '''
    
    ''' Load header-allele name mapping '''
    print('Loading header-allele mapping...')
    feature_to_allele = __load_feature_to_allele__(allele_names)
        
    ''' Generate proximal sequences '''
    print('Extracting', side, 'sequences...')
    genome_proximals = []
    for i, gff_fna in enumerate(genome_data):
        ''' Prepare output path '''
        genome_gff, genome_fna = gff_fna
        genome = __get_genome_from_filename__(genome_gff)
        genome_dir = '/'.join(genome_gff.split('/')[:-1]) + '/' if '/' in genome_gff else ''
        genome_prox_dir = genome_dir + 'derived/'
        if not os.path.exists(genome_prox_dir):
            os.mkdir(genome_prox_dir)
        genome_prox = genome_prox_dir + genome + '_' + side + '.fna'
            
        ''' Extract proximal sequences '''
        print(i+1, genome)
        genome_proximals.append(genome_prox)
        extract_proximal_sequences(genome_gff, genome_fna, genome_prox, 
                                   limits=limits, side=side,
                                   feature_to_allele=feature_to_allele,
                                   include_fragments=include_fragments,
                                   max_overlap=max_overlap)
        
    ''' Consolidate non-redundant proximal sequences per gene '''
    print('Identifying non-redundant', side, 'sequences per gene...')
    nr_prox_out = output_dir + '/' + name + '_nr_' + side + '.fna'
    nr_prox_out = nr_prox_out.replace('//','/')
    df_proximal = consolidate_proximal(genome_proximals, nr_prox_out, feature_to_allele, side)
    
    ''' Optionally sort non-redundant proximal sequences file '''
    if fastasort_path:
        print('Sorting sequences by header...')
        args = ['./' + fastasort_path, nr_prox_out]
        with open(nr_prox_out + '.tmp', 'w+') as f_sort:
            sp.call(args, stdout=f_sort)
        os.rename(nr_prox_out + '.tmp', nr_prox_out)
        
    ''' Save proximal x genome table '''
    prox_table_out = output_dir + '/' + name + '_strain_by_' + side
    prox_table_out = prox_table_out.replace('//','/')
    prox_table_pickle = prox_table_out + '.pickle.gz'
    prox_table_csv = prox_table_out + '.csv.gz'
    print('Saving', prox_table_pickle, '...')
    df_proximal.to_pickle(prox_table_pickle)
    if save_csv:
        print('Saving', prox_table_csv, '...')
        df_proximal.to_csv(prox_table_csv)
    return df_proximal

    
def consolidate_proximal(genome_proximals, nr_proximal_out, feature_to_allele, side):
    ''' 
    Consolidates proximal sequences to a non-redudnant set with respect to each
    gene described by feature_to_allele (maps_features to <name>_C# A# ), then
    creates a proximal x genome binary table. For use with fixed-length 3' or 5'UTRs.
    Upstream features are <name>_C# U# , downstream features are <name>_C# D# .
    
    Parameters
    ----------
    genome_proximals : list
        List of paths to proximal sequences FNA to combine
    nr_proximal_out : str
        Path to output non-redundant proximal sequences as FNA
    feature_to_allele : dict
        Dictionary mapping headers to <name>_C# A# alleles
    side : str
        'upstream' or 'downstream' for 5'UTR or 3'UTR
    
    Returns
    -------
    df_proximal : pd.DataFrame
         Binary proximal x genome table
    '''
    
    ftype_abb = VARIANT_TYPES[side]
    gene_to_unique_proximal = {} # maps gene:prox_seq:prox_seq_id (int)
    genome_to_proximal = {} # maps genome:proximal_name:1 if present (<name>_C# U# or <name>_C# D# )
    unique_proximal_ids = set() # record non-redundant list of proximal sequence IDs <name>_C# U# or <name>_C# D# genome_order = [] # sorted list of genomes inferred from proximal sequence file names
    
    with open(nr_proximal_out, 'w+') as f_nr_prox:
        for genome_proximal in sorted(genome_proximals):
            ''' Infer genome name from genome filename '''
            genome = genome_proximal.split('/')[-1] # trim off full path
            genome = genome.split('_' + side)[0] # remove .fna footer
            genome_to_proximal[genome] = {}
            genome_order.append(genome)
            
            ''' Process genome's proximal record '''
            with open(genome_proximal, 'r') as f_prox: # reading current proximal seq file
                header = ''; prox_seq = ''; new_sequence = False
                for line in f_prox.readlines(): # slight speed up reading whole file at once, should only be few MBs
                    if line[0] == '>': # header line
                        if len(prox_seq) > 0:
                            ''' Process header-seq to non-redundant <name>_C# <U/D># proximal allele '''
                            feature = header.split('_' + side + '(')[0] # trim off "_<up/down>stream" footer
                            allele = feature_to_allele[feature] # get <name>_C# A# allele
                            gene = __get_gene_from_allele__(allele) # gene <name>_C# gene
                            if not gene in gene_to_unique_proximal:
                                gene_to_unique_proximal[gene] = {}
                            if not prox_seq in gene_to_unique_proximal[gene]:
                                gene_to_unique_proximal[gene][prox_seq] = len(gene_to_unique_proximal[gene])
                                prox_id = gene + ftype_abb + str(gene_to_unique_proximal[gene][prox_seq])
                                unique_proximal_ids.add(prox_id)
                                new_sequence = True
                            prox_id = gene + ftype_abb + str(gene_to_unique_proximal[gene][prox_seq])
                            genome_to_proximal[genome][prox_id] = 1
                            
                            ''' Write renamed sequence to running file '''
                            if new_sequence:
                                f_nr_prox.write('>' + prox_id + '\n')
                                f_nr_prox.write(prox_seq + '\n')
                                new_sequence = False

                        header = line[1:].strip(); prox_seq = ''
                    else: # sequence line
                        prox_seq += line.strip()
            
                ''' Process last record'''
                feature = header.split('_' + side + '(')[0] # trim off "_<up/down>stream" footer
                allele = feature_to_allele[feature] # get <name>_C# A# allele
                gene = __get_gene_from_allele__(allele) # gene <name>_C# gene
                if not gene in gene_to_unique_proximal:
                    gene_to_unique_proximal[gene] = {}
                if not prox_seq in gene_to_unique_proximal[gene]:
                    gene_to_unique_proximal[gene][prox_seq] = len(gene_to_unique_proximal[gene])
                    prox_id = gene + ftype_abb + str(gene_to_unique_proximal[gene][prox_seq])
                    unique_proximal_ids.add(prox_id)
                    new_sequence = True
                prox_id = gene + ftype_abb + str(gene_to_unique_proximal[gene][prox_seq])
                genome_to_proximal[genome][prox_id] = 1

                ''' Write renamed sequence to running file '''
                if new_sequence:
                    f_nr_prox.write('>' + prox_id + '\n')
                    f_nr_prox.write(prox_seq + '\n')
                    new_sequence = False
                    
    ''' Convert nested dict to dict of genome:SparseArrays once all proximal sequences are known '''
    print('Sparsifying', side, 'table...')
    prox_order = sorted(list(unique_proximal_ids))
    del unique_proximal_ids
    prox_indices = {prox_order[i]:i for i in range(len(prox_order))} # map proximal ID to index
    for g,genome in enumerate(genome_order):
        prox_array = np.zeros(shape=len(prox_order), dtype='int64')
        for genome_prox in list(genome_to_proximal[genome].keys()):
            prox_i = prox_indices[genome_prox]
            prox_array[prox_i] = 1
        genome_to_proximal[genome] = pd.arrays.SparseArray(prox_array)
        genome_to_proximal[genome].fill_value = np.nan
        
    print('Constructing DataFrame...')
    df_proximal = pd.DataFrame(data=genome_to_proximal, index=prox_order)
    return df_proximal


def extract_upstream_sequences(genome_gff, genome_fna, upstream_out, limits=(-50,3), max_overlap=-1,
                               feature_to_allele=None, allele_names=None, include_fragments=False):
    '''
    Extracts nucleotides upstream of coding sequences to file, default 50bp + start codon.
    Refer to extract_proximal_sequences() for parameters.
    '''
    extract_proximal_sequences(genome_gff, genome_fna, proximal_out=upstream_out, 
                               limits=limits, max_overlap=max_overlap, side='upstream', 
                               feature_to_allele=feature_to_allele, allele_names=allele_names,
                               include_fragments=include_fragments)

    
def extract_downstream_sequences(genome_gff, genome_fna, downstream_out, limits=(-3,50), max_overlap=-1,
                                 feature_to_allele=None, allele_names=None, include_fragments=False):
    '''
    Extracts nucleotides downstream of coding sequences to file, default stop codon + 50 bp.
    Refer to extract_proximal_sequences() for parameters.
    '''
    extract_proximal_sequences(genome_gff, genome_fna, proximal_out=downstream_out, 
                               limits=limits, max_overlap=max_overlap, side='downstream', 
                               feature_to_allele=feature_to_allele, allele_names=allele_names,
                               include_fragments=include_fragments)
    
                
def extract_proximal_sequences(genome_gff, genome_fna, proximal_out, limits, max_overlap, side,
                               feature_to_allele=None, allele_names=None, include_fragments=False):
    '''
    Extracts nucleotides upstream or downstream of coding sequences. 
    Interprets GFFs as formatted by PATRIC:
        1) Assumes contigs are labeled "accn|<contig>". 
        2) Assumes protein features have ".peg." in the ID
        3) Assumes ID = fig|<genome>.peg.# Output features are named "<feature header>_<up/down>stream(<limit1>,<limit2>)" 
    if overlap is not restricted, otherwise features are named:
    "<feature header>_<up/down>stream(<limit1>,<limit2>,<max_overlap>)"
    Excludes features that do not have any UTR bases.
        
    Parameters
    ----------
    genome_gff : str
        Path to genome GFF file with CDS coordinates
    genome_fna : str
        Path to genome FNA file with contig nucleotides
    proximal_out : str
        Path to output upstream/downstream sequences FNA files
    limits : 2-tuple
        Length of proximal region to extract, formatted (-X,Y). For upstream, extracts X 
        upstream bases (up to but excluding first base of start codon) and first Y coding 
        bases (including first base of start codon). For downstream, extracts the last X
        coding bases and Y downstream bases. In both cases, the total length is X+Y bases.
    max_overlap : int
        If non-negative, truncates UTRs that cross over into coding sequences
        of other genes such that the overlap is no more than <max_overlap> nts.
        If negative, does not truncate UTRs s.t. all UTRs same length. 
        Note: Requires that GFF has features sorted by position.
    side : str
        'upstream' or 'downstream' for 5'UTR or 3'UTR
    feature_to_allele : dict, str
        Dictionary mapping original feature headers to <name>_C# A# short names,
        alternatively, the allele_names file can be provided (default None)
    allele_names : str
        Path to allele names file if feature_to_allele is not provided,
        should be named <name>_allele_names.tsv. If neither are provided,
        simply processes all features present in the GFF (default None)
    include_fragments : bool
        If true, include upstream sequences that are not fully available 
        due to contig boundaries (default False)
    '''

    def extract_utr(start, stop, strand, contig, contig_seq, occupancy):
        ''' Calculate ideal bounds for UTR, without accounting for overlap '''
        pos = (side, strand)
        utr_side = start if pos in [('upstream','+'),('downstream','-')] else stop # side UTR effectively starts from
        utr_limits = limits if strand == '+' else (-limits[1], -limits[0]) # how to extend UTR bounds
        utr_start = utr_side + utr_limits[0]
        utr_stop = utr_side + utr_limits[1]
        
        ''' Optionally account for overlap '''
        if max_overlap >= 0: # checking CDS-UTR overlaps
            leftbound, rightbound = strand_occupancy[contig][strand][(start, stop)]
            leftbound -= max_overlap
            rightbound += max_overlap
            if utr_start < leftbound: # 5' overlap exceeds limit
                utr_start = leftbound
                # print 'UTR start overlap for', start, stop
            if utr_stop > rightbound: # 3' overlap exceeds limit
                utr_stop = rightbound
                # print 'UTR stop overlap for', start, stop
            
        ''' Extract UTR from computed bounds, RC if negative strand '''
        proximal = contig_seq[utr_start:utr_stop].strip()
        proximal = reverse_complement(proximal) if strand == '-' else proximal
        is_fragment = (utr_start < 0) or (utr_stop > len(contig_seq)) # if cut-off by contig bounds
        return proximal, is_fragment
    
    
    strand_occupancy = {} # maps contig:strand:(start,stop):[(left start,stop), (right start,stop)]
    if max_overlap >= 0:   
        ''' Load feature order on each contig and each strand '''
        occupancies = {}
        with open(genome_gff, 'r') as f_gff:
            for line in f_gff:
                line = line.strip()
                if len(line) > 0 and line[0] != '# ':
                    contig, src, feat_type, start, stop, score, \
                        strand, phase, attr_raw = line.split('\t')
                    if feat_type == 'CDS': # only consider CDS-UTR overlaps
                        contig = contig.split('|')[-1] # accn|<contig> to just <contig>
                        if not contig in occupancies:
                            occupancies[contig] = {'+':[], '-':[]}
                        start = int(start) - 1 # starts are 1-indexed
                        stop = int(stop) # stops 1-indexed, inclusive = correct exclusive bound
                        occupancies[contig][strand].append( (start, stop) )
                         
        ''' Convert feature order to 5'/3' neighbors pairs '''
        for contig in occupancies:
            strand_occupancy[contig] = {'+':{}, '-':{}}
            for strand in occupancies[contig]:
                n_features = len(occupancies[contig][strand])
                for i, feature in enumerate(occupancies[contig][strand]): # features in order on strand
                    leftbound = (-np.inf,-np.inf) if i == 0 else occupancies[contig][strand][i-1]
                    rightbound = (np.inf,np.inf) if i == n_features-1 else occupancies[contig][strand][i+1]
                    leftbound = leftbound[1] # take 3' end of nearest CDS to the 5' side
                    rightbound = rightbound[0] # take 5' end of nearest CDS to the 3' end
                    strand_occupancy[contig][strand][feature] = (leftbound, rightbound) 
        del occupancies
                        
    ''' Load contig sequences '''
    contigs = load_sequences_from_fasta(genome_fna, header_fxn=lambda x: x.split()[0])
            
    ''' Load header-allele name mapping '''
    if feature_to_allele: # dictionary provided directly
        feat_to_allele = feature_to_allele
    elif allele_names: # allele map file provided
        feat_to_allele = __load_feature_to_allele__(allele_names)
    else: # no allele mapping, process everything
        feat_to_allele = None
                    
    ''' Parse GFF file for CDS coordinates '''
    feature_footer = '_' + side
    params = (limits[0], limits[1], max_overlap) if max_overlap >= 0 else limits
    feature_footer += str(params).replace(' ','')
    proximal_count = 0 # total UTRs extracted
    coding_length = limits[1] if side == 'upstream' else -limits[0] # bases of UTR that overlap with reference gene CDS
    with open(proximal_out, 'w+') as f_prox:
        with open(genome_gff, 'r') as f_gff:
            for line in f_gff:
                line = line.strip()
                if len(line) > 0 and line[0] != '# ':
                    contig, src, feat_type, start, stop, score, \
                        strand, phase, attr_raw = line.split('\t')
                    contig = contig.split('|')[-1] # accn|<contig> to just <contig>
                    start = int(start) - 1 # starts are 1-indexed, inclusive
                    stop = int(stop) # stops 1-indexed, inclusive = correct exclusive bound
                    attrs = {} # key:value
                    for entry in attr_raw.split(';'):
                        k,v = entry.split('='); attrs[k] = v
                    gffid = attrs['ID']

                    ''' Verify allele has been mapped, and contig has been identified '''
                    if contig in contigs: 
                        if gffid in feat_to_allele or feat_to_allele is None:
                            try:
                                contig_seq = contigs[contig]
                            except KeyError:
                                continue
                            proximal, is_fragment = extract_utr(start, stop, strand, contig, contig_seq, strand_occupancy)
                                
                            ''' Save proximal sequence '''
                            if len(proximal) > coding_length and (not is_fragment or include_fragments):
                                feat_name = gffid + feature_footer
                                f_prox.write('>' + feat_name + '\n')
                                f_prox.write(proximal + '\n')
                                proximal_count += 1
                                
    print('Loaded', side, 'sequences:', proximal_count)

    
def extract_noncoding(genome_gff, genome_fna, noncoding_out, flanking=(0,0),
                      allowed_features=['transcript', 'tRNA', 'rRNA', 'misc_binding']):
    '''
    Extracts nucleotides for non-coding sequences. 
    Interprets GFFs as formatted by PATRIC:
        1) Assumes contigs are labeled "accn|<contig>". 
        2) Assumes protein features have ".peg." in the ID
        3) Assumes ID = fig|<genome>.peg.# Parameters
    ----------
    genome_gff : str
        Path to genome GFF file with CDS coordinates
    genome_fna : str
        Path to genome FNA file with contig nucleotides
    noncoding_out : str
        Path to output transcript sequences FNA files
    flanking : tuple
        (X,Y) where X = number of nts to include from 5' end of feature,
        and Y = number of nts to include from 3' end feature. Features
        may be truncated by contig boundaries (default (0,0))
    allowed_features : list
        List of GFF feature types to extract. Default excludes 
        features labeled "CDS" or "repeat_region" 
        (default ['transcript', 'tRNA', 'rRNA', 'misc_binding'])
    '''
    contigs = load_sequences_from_fasta(genome_fna, header_fxn=lambda x:x.split()[0])
    with open(noncoding_out, 'w+') as f_noncoding:
        with open(genome_gff, 'r') as f_gff:
            for line in f_gff:
                ''' Check for non-comment and non-empty line '''
                if not line[0] == '# ' and not len(line.strip()) == 0: 
                    contig, src, feature_type, start, stop, \
                        score, strand, phase, meta = line.split('\t')
                    contig = contig[5:] # trim off "accn|" header
                    start = int(start)
                    stop = int(stop)
                    
                    if feature_type in allowed_features: 
                        ''' Get noncoding feature sequence and ID '''
                        contig_seq = contigs[contig]
                        fstart = start - 1 - flanking[0]
                        fstart = max(0,fstart) # avoid looping due to contig boundaries
                        fstop = stop + flanking[1]
                        feature_seq = contig_seq[fstart:fstop]
                        if strand == '-': # negative strand
                            feature_seq = reverse_complement(feature_seq)
                        meta_key_vals = [x.split('=') for x in meta.split(';')]
                        metadata = {x[0]:x[1] for x in meta_key_vals}
                        feature_id = metadata['ID']
                        
                        ''' Save to output file '''
                        feature_seq = '\n'.join(feature_seq[i:i+70] for i in range(0, len(feature_seq), 70))
                        f_noncoding.write('>' + feature_id + '\n')
                        f_noncoding.write(feature_seq + '\n') 

    
def validate_gene_table(df_genes, df_alleles, log_group=1):
    '''
    Verifies that the gene x genome table is consistent with the
    corresponding allele x genome table. Optimized to run column-by-column
    rather than gene-by-gene for sparse tables.
    
    Parameters
    ----------
    df_genes : pd.DataFrame or str
        Either the gene x genome table, or path to the table
    df_alleles : pd.DataFrame or str
        Either the allele x genome table, or path to the table
    log_group : int
        Print message per this many genomes 
    '''
    dfg = load_feature_table(df_genes)
    dfa = load_feature_table(df_alleles)
    print('Validating gene clusters...')
    num_inconsistencies = 0
    for g,genome in enumerate(df_genes.columns):
        if (g+1) % log_group == 0:
            print(g+1, 'Testing', genome)
        genes = set(dfg[genome].dropna().index)
        alleles = dfa[genome].dropna().index
        allele_genes = set(alleles.map(__get_gene_from_allele__))
        if genes != allele_genes:
            inconsistencies = genes.symmetric_difference(allele_genes)
            print('\tInconsistent:', inconsistencies)
            num_inconsistencies += len(inconsistencies)
    print('Gene Table Inconsistencies:', num_inconsistencies) 


def validate_gene_table_dense(df_genes, df_alleles):
    '''
    Verifies that the gene x genome table is consistent with the
    corresponding allele x genome table. Original approach for
    when df_genes and df_alleles are dense DataFrames.
    
    Parameters
    ----------
    df_genes : pd.DataFrame or str
        Either the gene x genome table, or path to the table
    df_alleles : pd.DataFrame or str
        Either the allele x genome table, or path to the table
    '''
    dfg = load_feature_table(df_genes)
    dfa = load_feature_table(df_alleles)
    print('Validating gene clusters...')
    
    current_cluster = None; allele_data = []; 
    clusters_tested = 0; inconsistencies = 0
    for allele_row in dfa.fillna(0).itertuples(name=None):
        cluster = __get_gene_from_allele__(allele_row[0])
        if current_cluster is None: # initializing
            current_cluster = cluster
        elif current_cluster != cluster: # end of gene cluster
            alleles_all = np.array(allele_data)
            has_gene = alleles_all.sum(axis=0) > 0
            is_consistent = np.array_equal(has_gene, dfg.loc[current_cluster,:].fillna(0).values)
            clusters_tested += 1
            if not is_consistent:
                print('Inconsistent', cluster)
                print(has_gene)
                print(dfg.loc[current_cluster,:].fillna(0).values)
                inconsistencies += 1
            if clusters_tested % 1000 == 0:
                print('\tTested', clusters_tested, 'clusters')
            allele_data = []
            current_cluster = cluster
        allele_data.append(np.array(allele_row[1:]))
    
    ''' Process final line '''
    alleles_all = np.array(allele_data) 
    has_gene = alleles_all.sum(axis=0) > 0
    is_consistent = np.array_equal(has_gene, dfg.loc[current_cluster,:].fillna(0).values)
    if not is_consistent:
        print('Inconsistent', cluster)
        print(has_gene)
        print(dfg.loc[current_cluster,:].fillna(0).values)
        inconsistencies += 1
    print('Gene Table Inconsistencies:', inconsistencies)
    

def validate_upstream_table(df_upstream, upstream_fna_paths, nr_upstream_fna,
                            allele_names, log_group=1):
    '''
    Verifies that the upstream x genome table is consistent with
    the corresponding extracted upstream sequences. See 
    validate_table_against_fasta() for details.
    
    Parameters
    ----------
    df_upstream : pd.DataFrame
        Binary upstream x genome table
    upstream_fna_paths : list
        List containing all paths to FNA files containing 
        upstream sequences for each genome
    nr_upstream_fna : str
        Path to FNA with non-redundant upstream sequences
    allele_names : str
        Path to allele names file generated by build_cds_pangenome()
    log_group : int
        Print message per this many genomes (default 1)
    '''
    return validate_table_against_fasta(
        df_features=df_upstream, genome_fasta_paths=upstream_fna_paths, 
        features_fasta=nr_upstream_fna, allele_names=allele_names,
        log_group=log_group)

    
def validate_downstream_table(df_downstream, downstream_fna_paths, nr_downstream_fna, 
                              allele_names, log_group=1):
    '''
    Verifies that the downstream x genome table is consistent with
    the corresponding extracted downstream sequences. See 
    validate_table_against_fasta() for details.
    
    Parameters
    ----------
    df_downstream : pd.DataFrame
        Binary downstream x genome table
    downstream_fna_paths : list
        List containing all paths to FNA files containing 
        downstream sequences for each genome
    nr_downstream_fna : str
        Path to FNA with non-redundant upstream sequences
    allele_names : str
        Path to allele names file generated by build_cds_pangenome()
    log_group : int
        Print message per this many genomes (default 1)
    '''
    return validate_table_against_fasta(
        df_features=df_downstream, genome_fasta_paths=downstream_fna_paths, 
        features_fasta=nr_downstream_fna, allele_names=allele_names, 
        log_group=log_group)
    
    
def validate_allele_table(df_alleles, genome_fasta_paths, 
                          alleles_fasta, log_group=1):
    ''' 
    Verifies that the allele x genome table is consistent with the
    the corresponding fasta files. Originally validate_table_against_fasta().
    
    Parameters
    ----------
    df_alleles : pd.DataFrame
        Binary allele x genome table, for either CDS or non-coding genes
    genome_fasta_paths : list
        List containing all paths to fastas containing feature sequences 
        per genome. FAAs for CDS, or FNAs for non-coding features.
    alleles_fasta : str
        Path to fasta with all non-redundant sequences
    cluster_type : str
        Either 'cds' or 'noncoding' depending on feature (default 'cds')
    log_group : int
        Print message per this many genomes (default 1)
    '''
    return validate_table_against_fasta(
        df_features=df_alleles, genome_fasta_paths=genome_fasta_paths, 
        features_fasta=alleles_fasta, allele_names=None, log_group=log_group)
    

def validate_table_against_fasta(df_features, genome_fasta_paths, 
                                 features_fasta, allele_names=None, 
                                 log_group=1):
    '''
    Verifies that a table x genome table is consistent with the original 
    fasta files. Works for the following cases:
    - CDS allele table vs original CDS FAA files
    - CDS upstream table vs original upstream FNA files
    - CDS downstream table vs original upstream FNA files
    - Non-coding allele table vs original non-coding FNA files
    
    Parameters
    ----------
    df_features : pd.DataFrame or str
        Either the feature x genome table, or path to the table
    genome_fasta_paths : list
        Paths to genome fasta files originally combined and clustered
    features_fasta : str
        Path to non-redundant sequences corresponding to df_features
    allele_names : str
        Path to allele names file. If provided, verifies both sequence
        and feature name at the cluster level. Required for upstream and 
        downstream sequence validation due to conserved UTRs (default None).
    log_group : int
        Print message per this many genomes (default 1)
    '''
    dfa = load_feature_table(df_features)
    inconsistencies = 0 # number of genomes with table-genome inconsistencies
    
    ''' Pre-load allele names if available '''
    if allele_names: 
        print('Loading feature names...')
        feathash_to_allele = {}
        with open(allele_names, 'r') as f:
            for line in f:
                data = line.strip().split('\t')
                allele = data[0]; features = data[1:]
                for feature in features:
                    ''' Workaround for PATRIC files: 
                        PATRIC gff files store features as fig|genome.peg.# , but
                        PATRIC faa files store features as fig|genome.peg.# |locus_tag, 
                        whenever locus_tags are available. Will trim off locus_tag, 
                        but may possibly break compatibility with non-PATRIC files. '''
                    if feature.count('|') == 2:
                        feature = feature[:feature.rindex('|')]
                    feature_hash = __hash_sequence__(feature)
                    feathash_to_allele[feature_hash] = allele

    ''' Pre-load hashes for non-redundant protein sequences '''
    print('Loading non-redundant sequences...')
    seqhash_to_feature = {}
    def load_sequence_entry(seq_blocks, header):
        if len(seq_blocks) > 0:
            seq = ''.join(seq_blocks)
            seq = seq if (allele_names is None) else seq + trim_variant(header)
            seqhash = __hash_sequence__(seq)
            if seqhash in seqhash_to_feature:
                print('COLLISION:' , header)
            seqhash_to_feature[seqhash] = header

    with open(features_fasta, 'r') as f_fasta:
        header = ''; seq_blocks = []
        for line in f_fasta:
            if line[0] == '>': # new sequence encountered
                load_sequence_entry(seq_blocks, header)
                header = line[1:].strip()
                seq_blocks = []
            else: # sequence encountered
                seq_blocks.append(line.strip())
        load_sequence_entry(seq_blocks, header) # process last record
    print('Non-redundant sequences:', len(seqhash_to_feature))

    ''' Validate individual genomes against table '''
    
    def check_genome_sequence(seq_blocks, genome_features, feature_name, num_missing):
        if len(seq_blocks) > 0:
            seq = ''.join(seq_blocks)
            if not (allele_names is None):
                ''' Also validating allele name '''
                feature_name = feature_name.split('_upstream(')[0]
                feature_name = feature_name.split('_downstream(')[0]
                feature_hash = __hash_sequence__(feature_name)
                if feature_hash in feathash_to_allele:
                    seq += trim_variant(feathash_to_allele[feature_hash])
            seqhash = __hash_sequence__(seq)
            if seqhash in seqhash_to_feature:
                ''' Note: Sequence hashes may be missing if any original
                    sequences were excluded intentionally, i.e. too short '''
                feature = seqhash_to_feature[seqhash] # original name to NR name
                genome_features.add(feature)
            else:
                num_missing += 1
        return genome_features
    
    missing_features = 0
    feature_counts = dfa.sum() # genome x total features
    for i, genome_fasta in enumerate(sorted(genome_fasta_paths)):
        if (i+1) % log_group == 0:
            print('Validating genome', i+1, ':', genome_fasta)
        ''' Load all features present in the genome '''
        genome_features = set()
        with open(genome_fasta, 'r') as f_fasta:
            feature_header = ''; seq_blocks = []
            for line in f_fasta:
                if line[0] == '>': # new sequence encountered
                    genome_features = check_genome_sequence(seq_blocks, genome_features, feature_header, missing_features)
                    feature_header = line[1:].strip()
                    seq_blocks = []
                else: # sequence encountered
                    seq_blocks.append(line.strip())
            genome_features = check_genome_sequence(seq_blocks, genome_features, feature_header, missing_features) 
            # process last record
            
        ''' Check that identified features are consistent with the table '''
        genome = __get_genome_from_filename__(genome_fasta) # trim off full path and .fna/.faa
        if not genome in dfa.columns: # possible footer
            genome = '_'.join(genome.split('_')[:-1])
        df_ga = dfa.loc[:,genome]
        table_features = set(df_ga.index[df_ga == 1]) # features from df_features
        test = table_features == genome_features # features from original fasta
        inconsistencies += (1 - int(test))
        if not test:
            table_only = len(table_features.difference(genome_features))
            genome_only = len(genome_features.difference(table_features))
            print(genome, '\t', 'Table only:', table_only, '\t', 'Genome only:', genome_only)
    print('Missing Features:', missing_features)
    print('Feature Table Inconsistencies:', inconsistencies)


def validate_upstream_table_direct(df_upstream, genome_fna_paths, nr_upstream_fna,
                                  limits=(-50,3), log_group=1):
    '''
    Does a partial validation of the upstream x genome table by checking that
    the recorded upstream sequences are present in the corresponding genome, 
    and counts start codons observed. DOES NOT check the exact location of the
    upstream sequences. See validate_proximal_table() for parameters.
    '''
    validate_proximal_table(df_upstream, genome_fna_paths, nr_upstream_fna, 
                            limits, 'upstream', log_group)

    
def validate_downstream_table_direct(df_downstream, genome_fna_paths, nr_downstream_fna,
                              limits=(-3,50), log_group=1):
    '''
    Does a partial validation of the downstream x genome table by checking that
    the recorded downstream downstream are present in the corresponding genome, 
    and counts stop codons observed. DOES NOT check the exact location of the
    downstream sequences. See validate_proximal_table() for parameters.
    '''
    validate_proximal_table(df_downstream, genome_fna_paths, nr_downstream_fna, 
                            limits, 'downstream', log_group)
    

def validate_proximal_table_direct(df_prox, genome_fna_paths, nr_prox_fna, limits, side, log_group=1):
    '''
    Does a partial validation of the proximal x genome table by checking that
    the recorded proximal sequences are present in the corresponding genome, 
    and counts start codons observed. DOES NOT check the exact location of the
    proximal sequences. TODO: Does not yet support non-fixed length UTRs.
    
    Parameters
    ----------
    df_prox : pd.DataFrame or str
        Either the proximal x genome table, or path to the table as CSV or CSV.GZ
    genome_fna_paths : list
        Paths to FNAs for each genome's contigs
    nr_prox_fna : str
        Path to FNA with non-redundant proximal sequences per gene
    limits : 2-tuple
        Proximal sequence limits specified when extracting upstream 
        or downstream sequences (default (-50,3))
    side : str
        Either "upstream" or "downstream"
    log_group : int
        Print message per this many genomes 
    '''
    dfp = load_feature_table(df_prox)
    
    ''' Load non-redundant proximal sequences '''
    print('Loading', side, 'sequences...')
    nr_prox = load_sequences_from_fasta(nr_prox_fna)
            
    ''' Verify present of each proximal sequence within each genome '''
    window = limits[1] - limits[0]
    for g, genome_fna in enumerate(genome_fna_paths):
        genome_contigs = load_sequences_from_fasta(genome_fna)
        genome = __get_genome_from_filename__(genome_fna)
        if (g+1) % log_group == 0:
            print(g+1, 'Evaluating', genome, genome_fna)
        dfp_strain = dfp.loc[:,genome]
        table_prox = dfp_strain.index[pd.notnull(dfp_strain)] # proximal sequences as defined by the table
        table_prox_seqs = {nr_prox[x]:x for x in table_prox} # maps sequences to names
        
        ''' Scan all contigs for detected fixed-length proximal sequences '''
        for contig in list(genome_contigs.values()):
            for i in range(len(contig)):
                segment = contig[i:i+window]
                if segment in table_prox_seqs: 
                    table_prox_seqs.pop(segment)
            rc_contig = reverse_complement(contig)
            for i in range(len(rc_contig)):
                segment = rc_contig[i:i+window]
                if segment in table_prox_seqs: 
                    table_prox_seqs.pop(segment)
                    
        ''' Report undetected proximal sequences '''
        for prox in table_prox_seqs:
            print('\tMissing', table_prox_seqs[prox], 'from', genome)
        
    ''' Count start/stop codons among non-redundant proximal sequences '''
    if limits[1] >= 3 and side == 'upstream':
        print('Computing start codon distribution...')
        if limits[1] == 3:
            get_start = lambda x: x[-3:]
        else:
            get_start = lambda x: x[-limits[1]:-limits[1]+3]
        start_codons = list(map(get_start, list(nr_prox.values())))
        print(collections.Counter(start_codons))
    elif limits[0] <= -3 and side == 'downstream':
        print('Computing stop codon distribution...')
        if limits[0] == -3:
            get_stop = lambda x: x[:3]
        else:
            get_stop = lambda x: x[-limits[0]-3:-limits[0]]
        stop_codons = list(map(get_stop, list(nr_prox.values())))
        print(collections.Counter(stop_codons))

        
def generate_annotations(features, annotation_files):
    '''
    For a set of features, creates a pd.Series with PATRIC annotations
    for each feature, or np.nan if no annotation was found.
    1) For cluster-level features, returns the consensus annotation
    2) For variant-level features, returns the variant-specific 
        annotation if recorded as different from consensus. Otherwise,
        returns the consensus annotation.
    3) For proximal UTR features, returns the consensus annotation
        of the parent cluster-level feature.
        
    Parameters
    ----------
    features : iterable
        List of features to extract annotations.
    annotations_files : iterable
        Paths to files containing annotations, from extract_annotations().
    '''
    relevant_features = list(features) + []
    for feature in features:
        name, cluster_type, cluster_num, variant_type, variant_num = \
            breakdown_feature_name(feature)
        if variant_type: # variant_level feature
            feature_cluster = name + '_' + cluster_type + str(cluster_num)
            relevant_features.append(feature_cluster)
            
    ''' Load relevant features from annotations file '''
    relevant_features = set(relevant_features)
    relevant_annotations = {}
    for annot_file in annotation_files:
        with open(annot_file, 'r') as f:
            for line in f:
                data = line.strip().split('\t'); feature = data[0]
                if feature in relevant_features:
                    relevant_annotations[feature] = ';'.join(data[1:])
    
    ''' Process loaded annotations '''
    feature_to_annot = {}
    for feature in features:
        if feature in relevant_annotations: # direct annotation exists
            feature_to_annot[feature] = relevant_annotations[feature]
        else: # possible cluster-level annotation exists
            name, cluster_type, cluster_num, variant_type, variant_num = \
                breakdown_feature_name(feature)
            feature_cluster = name + '_' + cluster_type + str(cluster_num)
            if not variant_type is None and feature_cluster in relevant_annotations: 
                feature_to_annot[feature] = relevant_annotations[feature_cluster]
            else: # cluster-level annotation is missing
                feature_to_annot[feature] = np.nan
    return pd.Series(feature_to_annot, index=features)



def extract_coding_fna(genome_gff, genome_fna, coding_out,
                       allowed_features=['CDS', 'tRNA']):
    '''
        Extracts nucleotides for coding sequences.
        Interprets GFFs as formatted by PATRIC:
            1) Assumes contigs are labeled "accn|<contig>".
            2) Assumes protein features have ".peg." in the ID
            3) Assumes ID = fig|<genome>.peg.# Parameters
        ----------
        genome_gff : str
            Path to genome GFF file with CDS coordinates
        genome_fna : str
            Path to genome FNA file with contig nucleotides
        coding_out : str
            Path to output transcript sequences FNA files
        allowed_features : list
            List of GFF feature types to extract. Default includes
            features labeled "CDS" and "tRNA"
            (default ['CDS', 'tRNA'])
    '''

    contigs = load_sequences_from_fasta(genome_fna, header_fxn=lambda x: x.split()[0])
    with open(coding_out, 'w+') as f_coding:
        with open(genome_gff, 'r') as f_gff:
            for line in f_gff:
                ''' Check for non-comment and non-empty line '''
                if not line[0] == '# ' and not len(line.strip()) == 0:
                    contig, src, feature_type, start, stop, \
                    score, strand, phase, meta = line.split('\t')
                    contig = contig[5:]  # trim off "accn|" header
                    fstart = int(start) - 1
                    fstop = int(stop)

                    if feature_type in allowed_features:
                        ''' Get noncoding feature sequence and ID '''
                        try:
                            contig_seq = contigs[contig]
                        except KeyError:
                            print(contigs.keys())
                            break

                        fstart = max(0, fstart)  # avoid looping due to contig boundaries
                        feature_seq = contig_seq[fstart:fstop]
                        if strand == '-':  # negative strand
                            feature_seq = reverse_complement(feature_seq)
                        meta_key_vals = [x.split('=') for x in meta.split(';')]
                        metadata = {x[0]: x[1] for x in meta_key_vals}
                        feature_id = metadata['ID']

                        ''' Save to output file '''
                        feature_seq = '\n'.join(feature_seq[i:i + 70] for i in range(0, len(feature_seq), 70))
                        f_coding.write('>' + feature_id + '\n')
                        f_coding.write(feature_seq + '\n')



def extract_annotations(genome_gffs, allele_name_file, annotations_out, 
                        batch=100, collapse_alleles=True, flexible_locus_tag=False,
                        allowed_features=None):
    '''
    For a given allele name file (usually <org>_allele_names.tsv),
    extracts the corresponding PATRIC annotations from original gff files.
    Can load gff files in batches to limit memory usage.
    
    Parameters
    ----------
    genome_gffs : list
        List of paths to GFF files
    allele_name_file : str
        Path to file with allele-protein ID map, usually <org>_allele_names.tsv
    annotations_out : str
        Path to output annotations. Will also create a .tmp intermediate.
    batch : int
        Maximum GFF files to load into at once
    collapse_alleles : bool
        If True, reports annotations at the gene level, using the most common allele
        annotation. Alleles with non-plurality annotation are reported separately.
        If False, reports all unique annotations for all alleles. (default True)
    flexible_locus_tag : bool
        For PATRIC annotations, CDS features map to fig|<ID>|<locus_tag> "3-term" 
        when tags are available, while RNA features map to fig|<ID> only "2-term". 
        If flexible_locus_tag is True, then all annotations are mapped to both 3-term
        and 2-term names when possible for better mapping at the cost of 2x memory use.
        Otherwise, only one name will be mapped, preferring 3-term names (default False)
    allowed_features : list or None
        List of GFF feature types to extract. If None, all feature
        types are extracted (default None)
    '''
    
    ''' Copy allele name table '''
    tmp_out = annotations_out + '.tmp'
    shutil.copyfile(allele_name_file, tmp_out)
    
    ''' Iteratively replace protein IDs with annotations from GFFs (in batches) '''
    n_gffs = len(genome_gffs)
    for g in range(0,n_gffs,batch):
        ''' Load annotations for batch of GFFs '''
        annotations = {}
        for gff in genome_gffs[g:g+batch]:
            with open(gff,'r') as f_gff:
                for line in f_gff:
                    data = line.strip().split('\t')
                    if len(data) == 9: 
                        feature_type = data[2]
                        if allowed_features is None or feature_type in allowed_features:
                            line_annots = data[-1].split(';')
                            line_annots = dict([x.split('=') for x in line_annots])
                            if 'ID' in line_annots and 'product' in line_annots:
                                product = line_annots['product']
                                product = urllib.parse.unquote(product) # replace % hex characters
                                fid2 = line_annots['ID']; fid3 = None
                                if 'locus_tag' in line_annots:
                                    fid3 = fid2 + '|' + line_annots['locus_tag']
                                if flexible_locus_tag: # save both names when possible
                                    annotations[fid2] = product
                                    if not fid3 is None:
                                        annotations[fid3] = product
                                else: # save only 3-term, or only 2-term if no locus_tag 
                                    fid = fid2 if (fid3 is None) else fid3
                                    annotations[fid] = product
        print('Loaded', len(annotations), 'annotations from batch', g+1, '-', min(n_gffs,g+batch))
        
        ''' Incorporate newly loaded annotations '''
        with open(tmp_out, 'r') as f_last:
            with open(tmp_out+'2', 'w+') as f_next:
                for line in f_last:
                    data = line.strip().split('\t')
                    allele = data[0]; fids = data[1:]
                    fids = [annotations[x] if x in annotations else x for x in fids]
                    fids = list(collections.OrderedDict.fromkeys(fids)) # remove duplicate annotations
                    output = allele + '\t' + '\t'.join(fids)
                    f_next.write(output + '\n')
        shutil.move(tmp_out+'2', tmp_out) # remove last round

    ''' Optionally collapse allele-level annotations to gene-level '''
    if collapse_alleles:
        with open(tmp_out, 'r') as f_last:
            current_cluster = None
            with open(annotations_out, 'w+') as f_next:
                for line in f_last:
                    data = line.strip().split('\t')
                    allele = data[0]
                    cluster =  __get_gene_from_allele__(allele)
                    allele_annots = '\t'.join(data[1:])
                    if current_cluster is None: # initialize first cluster
                        current_cluster = cluster
                        cluster_alleles = [allele]
                        cluster_annots = [allele_annots]
                    elif cluster == current_cluster: # continuing current cluster
                        cluster_alleles.append(allele)
                        cluster_annots.append(allele_annots)
                    else: # start of new cluster
                        most_common_annot, count = collections.Counter(cluster_annots).most_common(1)[0]
                        f_next.write(current_cluster + '\t' + most_common_annot + '\n')
                        for i, annots in enumerate(cluster_annots):
                            if annots != most_common_annot:
                                f_next.write(cluster_alleles[i] + '\t' + annots + '\n')
                        # Initialize next cluster
                        current_cluster = cluster
                        cluster_alleles = [allele]
                        cluster_annots = [allele_annots]
        os.remove(tmp_out)
    else: 
        shutil.move(tmp_out, annotations_out)
    

def load_sequences_from_fasta(fasta, header_fxn=None, seq_fxn=None, filter_fxn=None):
    ''' Loads sequences from a FAA or FNA file into a dict
        mapping headers to sequences. Can optionally apply a 
        function to all header (header_fxn) or to all
        sequences (seq_fxn). Drops the ">" from all headers,
        and removes line breaks from sequences. '''
    header_to_seq = {}
    with open(fasta, 'r') as f:
        header = ''; seq = ''
        for line in f:
            if line[0] == '>': # header line
                if len(header) > 0 and len(seq) > 0:
                    if filter_fxn is None or filter_fxn(header):
                        seq = seq_fxn(seq) if seq_fxn else seq
                        header_to_seq[header] = seq
                header = line.strip()[1:]
                header = header_fxn(header) if header_fxn else header
                seq = ''
            else: # sequence line
                seq += line.strip()
        if len(header) > 0 and len(seq) > 0:
            if filter_fxn is None or filter_fxn(header):
                seq = seq_fxn(seq) if seq_fxn else seq
                header_to_seq[header] = seq
    return header_to_seq


def load_feature_table(feature_table):
    ''' 
    Loads DataFrames from CSV, CSV.GZ, PICKLE, or PICKLE.GZ.
    Uses index_col=0 for CSVs. Returns feature_table if provided 
    with anything other than a string.
    '''
    if type(feature_table) == str: # path provided
        if feature_table[-4:].lower() == '.csv' or feature_table[-7:].lower() == '.csv.gz':
            return pd.read_csv(feature_table, index_col=0)
        elif feature_table[-7:].lower() == '.pickle' or feature_table[-10:].lower() == '.pickle.gz':
            return pd.read_pickle(feature_table)
        else:
            return feature_table
    else: # non-string input
        return feature_table
    
    
def reverse_complement(seq):
    ''' Returns the reverse complement of a DNA sequence.
        Supports lower/uppercase and ambiguous bases'''
    return ''.join([DNA_COMPLEMENT[base] for base in list(seq)])[::-1]


def create_feature_name(name, cluster_type, cluster_num, variant_type=None, variant_num=-1):
    ''' 
    Creates the short name of a given feature, defined as 
    <name>_<cluster_type_short><cluster_num>_<variant_type_short><variant_num>
    
    Parameters
    ----------
    name : str
        Header to precede short name
    cluster_type : str
        "cds" or "noncoding", abbreviated as C or T, respectively
    cluster_num : int
        ID of cluster
    variant_type : str
        "allele" or "upstream" or "downstream", abbreviated as A, U, or D,
        respectively. If None, assumes feature refers to whole cluster (default None)
    variant_num : int
        ID of variant. If negative, assumes feature refers to whole cluster (default -1)
    '''
    short_name = name + '_'
    cluster_name_short = CLUSTER_TYPES[cluster_type]
    short_name += cluster_name_short + str(cluster_num)
    if (not variant_type is None) and (variant_num >= 0):
        variant_type_short = VARIANT_TYPES[variant_type]
        short_name += variant_type_short + str(variant_num)
    return short_name


def breakdown_feature_name(feature_name):
    '''
    Separates a feature name into the name, cluster-type, 
    cluster-number, variant-type, and variant-number.
    Example 1: EsC_A123U56 -> ['EsC', 'A', 123, 'U', 56]
    Example 2: PsA_T789 -> ['PsA', 'T', 89, None, None]
    '''
    data = feature_name.split('_')
    name = '_'.join(data[:-1]); footer = data[-1]
    cluster_type = footer[0]
    for i in range(1,len(footer)): # identify variant type
        if footer[i] in VARIANT_TYPES_REV:
            variant_type = footer[i]
            cluster_num = int(footer[1:i])
            variant_num = int(footer[i+1:])
            return name, cluster_type, cluster_num, variant_type, variant_num
    cluster_num = int(footer[1:])
    return name, cluster_type, cluster_num, None, None


def trim_variant(feature_name):
    ''' Removes allele/upstream/downstream variant label
        to yield a cluster-level feature name. Trims off
        from the right to the right-most alphabetic character.
        Returns the input if no alphabetic characters are
        encountered. '''
    for i in range(1,len(feature_name)):
        if feature_name[-i].isalpha():
            return feature_name[:-i]
    return feature_name


def __create_sparse_data_frame__(sparse_array, index, columns):
    ''' 
    Creates a SparseDataFrame from a scipy.sparse matrix by initializing 
    individual columns as SparseSeries, then concatenating them. 
    Sometimes faster than native SparseDataFrame initialization? Known issues: 
    
    - Direct initialization is slow in v0.24.2, see https://github.com/pandas-dev/pandas/issues/16773
    - Empty SparseDataFrame initialization time scales quadratically with number of columns
    - Initializing as dense and converting sparse uses more memory than directly making sparse
    '''
    n_row, n_col = sparse_array.shape
    X = sparse_array.T if n_row < n_col else sparse_array # make n_col < n_row
    sparse_cols = []
    for i in range(X.shape[1]): # create columns individually
        sparse_col = pd.SparseSeries.from_coo(scipy.sparse.coo_matrix(X[:,i]), dense_index=True)
        sparse_cols.append(sparse_col)
    df = pd.concat(sparse_cols, axis=1)
    df = df.T if n_row < n_col else df # transpose back if n_col > n_row originally
    df.index = pd.Index(index)
    df.columns = pd.Index(columns)
    return df


def __load_feature_to_allele__(allele_names):
    ''' Loads feature-to-allele mapping from file, usually <name>_allele_names.tsv. '''
    map_feature_to_gffid = lambda x: '|'.join(x.split('|')[:2])
    feat_to_allele = {}
    with open(allele_names, 'r') as f_all:
        for line in f_all:
            data = line.strip().split('\t')
            allele = data[0]; synonyms = data[1:]
            for synonym in synonyms:
                gff_synonym = map_feature_to_gffid(synonym)
                feat_to_allele[gff_synonym] = allele
    return feat_to_allele
                          
def __get_gene_from_allele__(allele):
    ''' Converts <name>_C# A# or <name>_T# A# allele to 
        <name>_C# gene or <name>_T# transcript. '''
    splitter = VARIANT_TYPES['allele']
    return splitter.join(allele.split(splitter)[:-1])

def __get_genome_from_filename__(filepath):
    ''' Extracts genome from a filepath by removing the full 
        path and the extension '''
    # return filepath.split('/')[-1][:-4] # old version
    filename = os.path.split(filepath)[1] # remove full path
    return os.path.splitext(filename)[0] # remove extension

def __get_header_from_fasta_line__(line):
    ''' Extracts a short header from a full header line in a fasta'''
    return line.split()[0][1:].strip()

def __hash_sequence__(seq):
    ''' Hashes arbitary length strings/sequences to bytestrings '''
    return hashlib.sha256(seq).digest()

def __stream_stdout__(command):
    ''' Hopefully Jupyter-safe method for streaming process stdout '''
    process = sp.Popen(command, stdout=sp.PIPE, shell=True)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        yield line.rstrip()
    