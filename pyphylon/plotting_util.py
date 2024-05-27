import logging
import re
import urllib
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm.notebook import tqdm, trange
import multiprocessing
from IPython.display import display, HTML
import itertools
import plotly.graph_objects as go



def _get_attr(attributes, attr_id, ignore=False):
    """
    Helper function for parsing GFF annotations

    Parameters
    ----------
    attributes : str
        Attribute string
    attr_id : str
        Attribute ID
    ignore : bool
        If true, ignore errors if ID is not in attributes (default: False)

    Returns
    -------
    str, optional
        Value of attribute
    """

    try:
        return re.search(attr_id + "=(.*?)(;|$)", attributes).group(1)
    except AttributeError:
        if ignore:
            return None
        else:
            raise ValueError("{} not in attributes: {}".format(attr_id, attributes))

# Need to be updated for seperation of plasmid/chromosome

def gff2pandas(gff_file, feature=["CDS"], index=None):
    """
    Converts GFF file(s) to a Pandas DataFrame
    Parameters
    ----------
    gff_file : str or list
        Path(s) to GFF file
    feature: str or list
        Name(s) of features to keep (default = "CDS")
    index : str, optional
        Column or attribute to use as index

    Returns
    -------
    df_gff: ~pandas.DataFrame
        GFF formatted as a DataFrame
    """

    # Argument checking
    if isinstance(gff_file, str):
        gff_file = [gff_file]

    if isinstance(feature, str):
        feature = [feature]

    result = []

    for gff in gff_file:
        with open(gff, "r") as f:
            lines = f.readlines()

        # Get lines to skip
        skiprow = sum([line.startswith("#") for line in lines]) - 2

        # Read GFF
        names = [
            "accession",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "attributes",
        ]
        DF_gff = pd.read_csv(gff, sep="\t", skiprows=skiprow, names=names, header=None, low_memory=False)

        region = DF_gff[DF_gff.feature == 'region']
        region_len = int(region.iloc[0].end)

        oric = 0
        # try:
        #     oric = list(DF_gff[DF_gff.feature == 'oriC'].start)[0]
        # except:
        #     oric = [0]
        
        # Filter for CDSs
        DF_cds = DF_gff[DF_gff.feature.isin(feature)]

        # Sort by start position
        DF_cds = DF_cds.sort_values("start")

        # Extract attribute information
        DF_cds["locus_tag"] = DF_cds.attributes.apply(_get_attr, attr_id="locus_tag")

        result.append(DF_cds)

    DF_gff = pd.concat(result)

    if index:
        if DF_gff[index].duplicated().any():
            logging.warning("Duplicate {} detected. Dropping duplicates.".format(index))
            DF_gff = DF_gff.drop_duplicates(index)
        DF_gff.set_index("locus_tag", drop=True, inplace=True)

    return DF_gff[['start', 'end', 'locus_tag']], region_len, oric

def h2a(x, header_to_allele):
    """
    Transforms a given locus tag using the header_to_allele dictionary.

    Parameters:
    x (str): The locus tag to be transformed.
    header_to_allele (dict): A dictionary mapping locus tags to allele strings.

    Returns:
    str or None: Transformed locus tag prefixed with 'A', or None if an error occurs.
    """
    try:
        return 'A' + header_to_allele[x].split('A')[1]
    except:
        return None
    
def generate_strain_vectors(path_to_data, metadata):
    """
    Generates a dictionary of gene orders for each strain based on GFF3 files.

    Parameters:
    path_to_data (str): The base directory path where the data is stored.
    metadata (DataFrame): A DataFrame containing metadata, which includes the genome_id of each strain.
    header_to_allele (dict): A dictionary mapping locus tags to allele strings.

    Returns:
    dict: A dictionary where keys are strain names and values are lists of genes in order.
    """

    strain_vectors = {}
    
    for strain in tqdm(metadata.genome_id):
        try:
            DF_gff, size, oric = gff2pandas(f'{path_to_data}/processed/bakta/{strain}/{strain}.gff3')

            DF_gff['gene'] = DF_gff.locus_tag.apply(lambda x: h2a(x))
            DF_gff = DF_gff[['gene', 'start']]
            gene_order = DF_gff.sort_values('start').gene.to_list()

            strain_vectors[strain] = gene_order
        except Exception as e:
            print(f"Error processing strain {strain}: {e}")

    return strain_vectors

def plot_gene_length_distribution(strain_vectors):
    """
    Plots a histogram showing the distribution of gene lengths for given strain vectors.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    """
    # Collect lengths of gene lists
    gene_lengths = [len(genes) for genes in strain_vectors.values()]

    # Creating the histogram
    plt.hist(gene_lengths, bins=10, color='blue', edgecolor='black')

    # Adding titles and labels
    plt.title('Distribution of Gene Lengths')
    plt.xlabel('Gene Length')
    plt.ylabel('Frequency')

    # Display the histogram
    plt.show()

def count_common_gene_appearances(strain_vectors):
    """
    Counts the occurrences of common genes across different strains.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    DataFrame: A DataFrame where rows are strains and columns are common genes,
               with each cell representing the count of a gene in the respective strain.
    """

    # Create a set of all genes in the first strain
    common_genes = set(strain_vectors[next(iter(strain_vectors))])

    # Find intersection of genes in all strains to get common genes
    for genes in strain_vectors.values():
        common_genes.intersection_update(genes)
    
    # Prepare data for DataFrame: count occurrences of each common gene in each strain
    data = {gene: [] for gene in common_genes}
    strains = []

    for strain, genes in strain_vectors.items():
        strains.append(strain)
        gene_count = {gene: genes.count(gene) for gene in common_genes}
        for gene in common_genes:
            data[gene].append(gene_count[gene])
    
    # Create the DataFrame
    df = pd.DataFrame(data, index=strains)
    
    return df

def find_once_genes(strain_vectors):
    """
    Finds genes that appear exactly once in each strain and returns the common and once-only genes.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    tuple: The number of common genes, the number of genes appearing exactly once in each strain, and a set of those genes.
    """
    # Finding intersection of all strains
    common_genes = set(strain_vectors[next(iter(strain_vectors))])  # Start with the first strain's genes
    for genes in strain_vectors.values():
        common_genes.intersection_update(genes)
    
    # Check for genes that appear exactly once in each strain
    once_genes = set()
    all_strains_genes = list(strain_vectors.values())
    first_strain_genes = all_strains_genes[0]

    # Only add genes to the consistent set if they appear exactly once in every strain
    for gene in common_genes:
        if all(genes.count(gene) == 1 for genes in all_strains_genes):
            once_genes.add(gene)
        
    return len(common_genes), len(once_genes), once_genes

def reorder_genes_by_strain(strain_vectors, genes, strain_name):
    """
    Reorders a list of genes based on their appearance in a specified strain.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.
    genes (list): A list of genes to be reordered.
    strain_name (str): The identifier of the strain to use for ordering.

    Returns:
    list: The reordered list of genes or an error message if the strain is not found.
    """
    # Check if the specified strain exists in the strain_vectors dictionary
    if strain_name not in strain_vectors:
        return f"Strain '{strain_name}' not found."

    # Retrieve the gene list for the specified strain
    gene_list = strain_vectors[strain_name]

    # Create a dictionary to find the index of each gene in the strain
    gene_index_map = {gene: gene_list.index(gene) for gene in gene_list if gene in genes}

    # Sort the genes by their index in the strain using the gene_index_map
    ordered_genes = sorted(genes, key=lambda gene: gene_index_map.get(gene, float('inf')))

    return ordered_genes

def rearrange_genes(gene_list, target_gene):
    """
    Rearranges a list of genes such that the target gene is the first element.

    Parameters:
    gene_list (list): The list of genes to be rearranged.
    target_gene (str): The gene to position as the first element in the rearranged list.

    Returns:
    list: The rearranged list of genes, or the original list if the target gene is not found.
    """
    # Check if the target_gene exists in the list
    if target_gene in gene_list:
        # Find the index of the target_gene
        index = gene_list.index(target_gene)
        # Rearrange: genes after the target_gene come first, then target_gene, then genes before target_gene
        rearranged_list = gene_list[index:] + gene_list[:index]
        return rearranged_list
    else:
        # Return the original list if target_gene is not found
        return gene_list

def standardize_strain_orders(strain_vectors, consistent_order_genes, reference_strain_name):
    """
    Standardizes the gene order across strains to match a reference strain.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.
    consistent_order_genes (list): A list of genes to use for ordering.
    reference_strain_name (str): The identifier of the reference strain.

    Returns:
    tuple: The updated strain vectors, the number of strains flipped, the list of problem strains, and the list of updated strains.
    """
    # Get the reference strain's name and gene list
    reference_strain_name = reference_strain_name
    reference_strain_genes = strain_vectors[reference_strain_name]

    # Order genes in the reference strain according to the consistent_order_genes
    reference_ordered_genes = reorder_genes_by_strain(strain_vectors, consistent_order_genes, reference_strain_name)

    # Initialize a counter for the number of strains flipped
    count = 0
    strain_vectors_updated = {}
    problem_strains = []
    
    # Adjust each strain to match the reference order
    for strain_name, genes in strain_vectors.items():
        # Reorder genes in the current strain
        current_ordered_genes = reorder_genes_by_strain(strain_vectors, consistent_order_genes, strain_name)
        current_ordered_genes_1 = rearrange_genes(reorder_genes_by_strain(strain_vectors, consistent_order_genes, strain_name), reference_ordered_genes[0])
        current_ordered_genes_2 = rearrange_genes(reorder_genes_by_strain(strain_vectors, consistent_order_genes, strain_name), reference_ordered_genes[-1])
        
        # Check if current order matches the reference order or its reverse
        if current_ordered_genes_1 == reference_ordered_genes or current_ordered_genes_2 == reference_ordered_genes:
            strain_vectors_updated[strain_name] = genes
            count += 1
            continue  # This strain is already correctly ordered
        elif current_ordered_genes_1 == reference_ordered_genes[::-1] or current_ordered_genes_2 == reference_ordered_genes[::-1]:
            strain_vectors_updated[strain_name] = genes[::-1]
            count += 1
            continue
        else:
            problem_strains.append(strain_name)
            continue

    return strain_vectors_updated, count, problem_strains, list(strain_vectors_updated.keys())

def create_strain_groups(strain_vectors_filtered, once_genes, starting_strain):
    """
    Groups strains based on consistent gene orders starting from a specified strain.

    Parameters:
    strain_vectors_filtered (dict): A dictionary where keys are strain identifiers and values are lists of genes.
    once_genes (list): A list of genes that appear exactly once in each strain.
    starting_strain (str): The identifier of the strain to start the grouping process.

    Returns:
    dict: A dictionary where keys are group identifiers and values are lists of strains in each group.
    """
    # Initialize variables
    groups = {}
    all_consistent_strains = set()
    
    # Start with the first strain
    current_strain = starting_strain
    group_number = 1
    
    while True:
        # Run the standardization function
        _, _, problem_strains, consistent_strains = standardize_strain_orders(
            strain_vectors_filtered, once_genes, current_strain)
        
        # Add the group to the dictionary
        group_key = f'strain_group_{group_number}'
        groups[group_key] = consistent_strains
        all_consistent_strains.update(consistent_strains)
        
        # Print the current group and the number of strains it contains
        print(f" {group_key}: {len(consistent_strains)} strains.")

        # Find a new strain from those not yet in all_consistent_strains
        remaining_strains = set(strain_vectors_filtered.keys()) - all_consistent_strains
        if not remaining_strains:
            break  # Exit if there are no more strains to process
        
        # Pick a new strain to use as the next starting point
        next_strain = next(iter(remaining_strains), None)
        if next_strain is None:
            break  # No new strain found to differentiate the strains

        current_strain = next_strain
        group_number += 1
    
    return groups

def update_strain_vector(reference_ordered_genes, strain_vectors_filtered):
    """
    Updates strain vectors by mapping genes to their positions in a reference ordered gene list.

    Parameters:
    reference_ordered_genes (list): A list of genes in the reference order.
    strain_vectors_filtered (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    dict: A dictionary with updated strain vectors where genes are replaced by their positions in the reference list.
    """
    gene_mapping = {gene: idx for idx, gene in enumerate(reference_ordered_genes, start=1)}
    
    # Apply the mapping to strain_vectors_filtered, keep unmapped genes unchanged
    updated_strain_vectors = {}
    
    for strain, genes in strain_vectors_filtered.items():
        updated_genes = [gene_mapping.get(gene, gene) for gene in genes]  # Use .get() to return the gene itself if not found
        updated_strain_vectors[strain] = updated_genes

    return updated_strain_vectors

def adjust_gene_order(strain_vectors):
    """
    Adjusts the gene order in strain vectors by reversing lists that are generally decreasing.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    tuple: A dictionary with adjusted gene orders and a count of how many lists were reversed.
    """
    # Function to determine if a list is generally decreasing
    def is_generally_decreasing(numbers):
        decreasing_count = sum(x > y for x, y in zip(numbers, numbers[1:] + [numbers[0]]))
        # Consider it decreasing if more than half of the comparisons are decreasing
        return decreasing_count > len(numbers) / 2

    final_strain_vectors = {}
    reversed_count = 0  # Counter for how many lists are reversed

    for strain, genes in strain_vectors.items():
        # Extract numbers and ignore non-numerical entries
        numbers = [x for x in genes if isinstance(x, int)]
        if numbers:  # Check if there are any numbers
            if is_generally_decreasing(numbers):
                genes.reverse()  # Reverse the whole list if numbers are generally decreasing
                reversed_count += 1  # Increment counter if reversed
        final_strain_vectors[strain] = genes

    return final_strain_vectors, reversed_count

def reorder_to_start_with_one(strain_vectors):
    """
    Reorders genes in strain vectors so that the gene '1' starts first if it is present.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    tuple: A dictionary with reordered strain vectors and a count of how many lists were changed.
    """
    strain_vectors_final = {}
    count_changed = 0  # Initialize counter for changed lists

    for strain, genes in strain_vectors.items():
        if 1 in genes:
            index_of_one = genes.index(1)
            if index_of_one != 0:  # Check if '1' is not already the first element
                # Rotate the list so that '1' starts first, and the part before '1' goes to the end
                reordered_genes = genes[index_of_one:] + genes[:index_of_one]
                strain_vectors_final[strain] = reordered_genes
                count_changed += 1  # Increment the counter as the list is changed
            else:
                strain_vectors_final[strain] = genes  # '1' is already the first, no change needed
        else:
            # If '1' is not in the list, keep it unchanged
            strain_vectors_final[strain] = genes

    return strain_vectors_final, count_changed

def check_strict_sequence(strain_vectors):
    """
    Checks if the gene numbers in each strain vector follow a strict sequence [1, 2, 3, ..., max] without any gaps.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    tuple: A dictionary with boolean values indicating whether each strain follows the strict sequence,
           the count of strains that follow the strict sequence, and the count of strains that do not.
    """
    results = {}
    count_true = 0
    count_false = 0
    
    for strain, genes in strain_vectors.items():
        # Extract only integer entries from the genes list
        numbers = [x for x in genes if isinstance(x, int)]
        # Check if the numbers are exactly [1, 2, 3, ..., max(numbers)] in that order
        if numbers and numbers == list(range(1, max(numbers) + 1)):
            results[strain] = True
            count_true += 1
        else:
            results[strain] = False
            count_false += 1
            
    return results, count_true, count_false

def generate_gene_names(strain_vectors):
    """
    Generates descriptive names for genes based on their positions relative to numerical markers in strain vectors.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    DataFrame: A DataFrame where rows are genes and columns are strains, 
               with cells containing the descriptive gene names.
    """
    gene_names = {}
    
    for strain, genes in strain_vectors.items():
        # Find indices and values of numerical markers
        number_indices = [i for i, g in enumerate(genes) if isinstance(g, int)]
        number_values = [g for g in genes if isinstance(g, int)]
        # Assume circular nature
        number_indices = number_indices + [len(genes) + ni for ni in number_indices]
        number_values = number_values + number_values
        
        # Temporary storage for gene names of the current strain
        current_names = {}
        
        for i in range(len(genes)):
            if not isinstance(genes[i], int):  # Process if it's a gene identifier
                # Find the closest previous and next numbers
                previous_number_index = max([ni for ni in number_indices if ni < i])
                next_number_index = min([ni for ni in number_indices if ni > i])
                
                previous_number = genes[previous_number_index % len(genes)]
                next_number = genes[next_number_index % len(genes)]
                
                # Count the genes between the numbers including this one
                count_before = i - previous_number_index
                count_after = next_number_index - i
                
                # Form the new gene name
                gene_name = f"{previous_number}_{count_before}_{count_after}_{next_number}"
                current_names[genes[i]] = gene_name
        
        # Store names with respect to their original gene identifier
        gene_names[strain] = current_names
    
    # Create a DataFrame from the dictionary
    all_genes = sorted(set(g for names in gene_names.values() for g in names if isinstance(g, str)))
    df = pd.DataFrame(index=all_genes, columns=strain_vectors.keys())
    
    for strain, names in gene_names.items():
        for gene, name in names.items():
            if gene in df.index:  # Ensure the gene is part of the index
                df.at[gene, strain] = name
    
    return df.fillna('NA')

def count_genes_between_anchor_genes(df, strain):
    """
    Counts the genes between numerical anchor genes for a given strain.

    Parameters:
    df (DataFrame): A DataFrame where rows are genes and columns are strains, with cells containing descriptive gene names.
    strain (str): The identifier of the strain to process.

    Returns:
    DataFrame: A DataFrame where rows are anchor gene pairs and columns are the counts of genes between them.
    """
    # Extract the column for the strain and remove NA values
    column_data = pd.DataFrame(df[strain][df[strain] != 'NA']).reset_index().rename(columns={'index': 'Gene'})

    # Extract number pairs and initialize count dictionary
    counts = {}
    
    for entry in column_data[strain]:
        parts = entry.split('_')
        if len(parts) == 4:
            number_before = parts[0]
            number_after = parts[3]
            
            key = f"{number_before}-{number_after}"
            if key not in counts:
                counts[key] = 0
            counts[key] += 1
  
    # Convert the dictionary to a DataFrame sorted by the number pairs
    result_data = [(key, value) for key, value in sorted(counts.items(), key=lambda x: x[0])]
    result_df = pd.DataFrame(result_data, columns=['Anchor Genes', 'Total Genes Between'])

    return result_df

def create_gene_count_between_anchor_genes_for_all(df):
    """
    Creates a dictionary of DataFrames counting genes between anchor genes for all strains.

    Parameters:
    df (DataFrame): A DataFrame where rows are genes and columns are strains, with cells containing descriptive gene names.

    Returns:
    dict: A dictionary where keys are strain identifiers and values are DataFrames 
          containing the counts of genes between anchor genes for each strain.
    """
    result_dict = {}
    for column in df.columns:
        gene_count_between_anchor_genes = count_genes_between_anchor_genes(df, column)
        result_dict[column] = gene_count_between_anchor_genes
    return result_dict

def identify_variation(numbers, ordered_numbers):
    """
    Identifies the type of genetic variation by comparing two lists of numbers.

    Parameters:
    numbers (list): A list of integers representing gene positions in the original order.
    ordered_numbers (list): A list of integers representing gene positions in the expected order.

    Returns:
    str: The type of variation ('no variation', 'inversion', 'translocation', 'others').
    """
    if numbers == ordered_numbers:
        return 'no variation'
    
    n = len(numbers)
    visited = [False] * n
    for i in range(n):
        if visited[i] or numbers[i] == ordered_numbers[i]:
            visited[i] = True
            continue
        
        # Start of a segment
        segment_original = []
        segment_ordered = []
        pos = i
        while pos < n and not visited[pos]:
            segment_original.append(numbers[pos])
            segment_ordered.append(ordered_numbers[pos])
            visited[pos] = True
            pos = numbers.index(ordered_numbers[pos])
        
        if segment_original == segment_ordered[::-1]:
            return 'inversion'
        elif sorted(segment_original) != segment_ordered:
            return 'translocation'
    
    return 'others'

def identify_genetic_variation(strain_vectors):
    """
    Identifies the type of genetic variation for each strain by comparing gene positions.

    Parameters:
    strain_vectors (dict): A dictionary where keys are strain identifiers and values are lists of genes.

    Returns:
    DataFrame: A DataFrame where each row represents a strain and its identified variation type.
    """
    results = []
    
    for strain, genes in strain_vectors.items():
        numbers = [x for x in genes if isinstance(x, int)]
        if not numbers:
            continue
        ordered_numbers = list(range(min(numbers), max(numbers) + 1))
        
        variation_type = identify_variation(numbers, ordered_numbers)
        results.append([strain, variation_type])
    
    # Create DataFrame
    result_df = pd.DataFrame(results, columns=['Strain', 'Variation'])

    # Print the count of each category
    print(result_df['Variation'].value_counts())
    
    return result_df

def filter_genes_and_strains(gene_mapping_to_anchor_genes, L_binarized, A_binarized, phylon):
    gene_list = list(L_binarized[phylon][L_binarized[phylon] == 1].index)
    strain_list = list(A_binarized.loc[phylon][A_binarized.loc[phylon] == 1].index)
    # Filter the DataFrame to only include specified genes and strains
    filtered_df = gene_mapping_to_anchor_genes.loc[gene_list, strain_list]
    return filtered_df