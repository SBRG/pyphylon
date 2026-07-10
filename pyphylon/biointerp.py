import os
import pandas as pd
import networkx as nx
from pyphylon.plotting_util import gff2pandas, update_df_gff_panaroo


def collect_functions(WORKDIR, anno_path):
    '''
    Collects functions from BAKTA or PROKKA annotation files
    WORKDIR: str, path to the working directory
    anno_path: str, path to the annotation files (e.g. processed/bakta)
    Retuns a pandas DataFrame with columns: locus, genome_id, product, go annotation
    '''
    
    from glob import glob
    anno_folders = glob(os.path.join(WORKDIR, anno_path, "*"))
    all_data = pd.DataFrame()
    for anno_folder in anno_folders:
        genome_id = anno_folder.split(os.sep)[-1]
        functions = os.path.join(anno_folder, f'{genome_id}.tsv')
        data = pd.read_csv(functions, sep='\t', skiprows=5)
        data['genome_id'] = genome_id
        data = data[['Product', 'DbXrefs', 'Locus Tag', 'genome_id']]
        all_data = pd.concat([all_data, data])
    all_data = all_data.rename(columns={'Product':'product','DbXrefs':'go', 'Locus Tag': 'locus'})
    print(f'loaded {len(all_data)} functions from {len(anno_folders)} BAKTA tsv files')
    return all_data[['locus','genome_id','product','go']].drop_duplicates()


def get_pg_to_locus_map(WORKDIR, SPECIES, PANAROO = False):
    '''
    This function reads the allele names file and returns a dataframe with the mapping of pg cluster to locus in each strain for cdhit pangenomes
    '''
    if PANAROO: # Process map for panaroo data layout
        gene_information = pd.read_csv(os.path.join(WORKDIR, f'processed/panaroo_results/{SPECIES}/gene_data.csv'), dtype = object)
    
        annotation_to_clustering_dict = dict(
            zip(gene_information["annotation_id"], gene_information["clustering_id"])
        )
        
        panaroo_tag_dict = generate_tags_to_clusters_panaroo(WORKDIR, SPECIES)

        out = []
        for locus in annotation_to_clustering_dict.keys():
            panaroo_cluster = annotation_to_clustering_dict[locus]
            if panaroo_cluster in panaroo_tag_dict.keys():
                out.append({'cluster':panaroo_tag_dict[panaroo_cluster], 'gene_id':locus})
        out = pd.DataFrame(out)
        out.drop_duplicates(inplace=True)
        return pd.DataFrame(out)

    # CD-HIT Default Functionality
    fin = open(os.path.join(WORKDIR, f'processed/cd-hit-results/{SPECIES}_allele_names.tsv'))
    out = []
    for line in fin:
        allele = line.strip().split('\t')[0]
        cluster = allele.split('A')[0]
        loci = line.strip().split('\t')[1:]
        for locus in loci:
            out.append({'cluster': cluster, 'gene_id': locus})
    out = pd.DataFrame(out)
    out.drop_duplicates(inplace=True)
    return pd.DataFrame(out)


def generate_strain_vectors(WORKDIR, SPECIES, STRAIN_IDS, PANAROO = False, chromosome_only = False):
    '''
        Function to generate strain vectors (the physical order of genes in a strain in relation to the pangenomoe) for either CD-HIT (default) or PANAROO.

        This function will call downstream methods in order to create the vectors based on if Panaroo or CD-HIT is being used to construct the pangenome
    '''
    if PANAROO:
        return generate_strain_vectors_panaroo(WORKDIR, SPECIES, STRAIN_IDS, chromosome_only)
    return generate_strain_vectors_cdhit(WORKDIR, SPECIES, STRAIN_IDS, chromosome_only)

    
def generate_strain_vectors_panaroo(WORKDIR, SPECIES, STRAIN_IDS, chromosome_only = False):
    '''
        Function to generate strain vectors (the physical order of genes in a strain in relation to the pangenomoe) for Panaroo.

        This accounts for the refound genes using the gene information supplied by Panaroo.
    '''
    strain_vectors = {}

    gene_information = pd.read_csv(os.path.join(WORKDIR, f'processed/panaroo_results/{SPECIES}/gene_data.csv'), dtype = object)
    gene_data_panaroo_refound = gene_information[gene_information.clustering_id.str.contains('refound')]

    annotation_to_clustering_dict = dict(
        zip(gene_information["annotation_id"], gene_information["clustering_id"])
    )
    
    panaroo_tag_dict = generate_tags_to_clusters_panaroo(WORKDIR, SPECIES)

    def locus_to_gene(x):
        if 'refound' not in x:
            x = annotation_to_clustering_dict.get(x, None)
        return panaroo_tag_dict.get(x, None)
    
    for strain in STRAIN_IDS:
        DF_gff, size, oric = gff2pandas(os.path.join(WORKDIR, f'processed/bakta/{strain}/{strain}.gff3'))
        DF_gff = update_df_gff_panaroo(DF_gff, gene_data_panaroo_refound, strain)
        DF_gff['gene'] = DF_gff.locus_tag.apply(lambda x: locus_to_gene(x))
        DF_gff = DF_gff.dropna(subset=['gene'])

        if chromosome_only:   
            DF_gff = DF_gff[DF_gff.accession == DF_gff.accession.value_counts().index[0]]
        
        gene_order = DF_gff.sort_values(by = ['accession', 'start']).drop_duplicates(subset='gene', keep = 'first')
        strain_vectors[strain] = gene_order
    return strain_vectors
        



def generate_tags_to_clusters_panaroo(WORKDIR, SPECIES):
    '''
        Function to go through Panaroo graph in order to create a dictionary matching each node in the graph to its respective genome tags as a dictionary
    '''
    graph = nx.read_gml(os.path.join(WORKDIR, f'processed/panaroo_results/{SPECIES}/final_graph.gml'))

    panaroo_dict = {}
    for node, attributes in graph.nodes.data(True):
        name = attributes['name']
        seq_ids = attributes['seqIDs']
        for seq_id in seq_ids:
            panaroo_dict[seq_id] = name
    return panaroo_dict


def generate_strain_vectors_cdhit(WORKDIR, SPECIES, STRAIN_IDS, chromosome_only = False):
    '''
        Function to generate strain vectors (the physical order of genes in a strain in relation to the pangenomoe) for CD-HIT.

        This accounts for the header_to_allele file generated by cd-hit in order to generate the vectors for each strain
    '''
    strain_vectors = {}
    
    pg2locus_map = get_pg_to_locus_map(WORKDIR, SPECIES)
    
    for strain in STRAIN_IDS:
        DF_gff, size, oric = gff2pandas(os.path.join(WORKDIR, f'processed/bakta/{strain}/{strain}.gff3'))
        DF_gff = pd.merge(DF_gff, pg2locus_map, left_on='locus_tag', right_on='gene_id', how='left')
        DF_gff.rename(columns={'cluster':'gene'}, inplace=True)
        
        if chromosome_only:   
            DF_gff = DF_gff[DF_gff.accession == DF_gff.accession.value_counts().index[0]]
        
        gene_order = DF_gff.sort_values(by = ['accession', 'start']).drop_duplicates(subset='gene', keep = 'first')
        
        strain_vectors[strain] = gene_order

    return strain_vectors

    
def calc_enrichment(L, cluster_to_go_functions, function, functions2genes, phylon, phylon_contribution_cutoff=0):
    '''
    calculates the enrichment of a function in a phylon
    input: 
        L = L matrix from NMF
        cluster_to_go_functions dataframe from get_cluster_to_go_functions function
        function = function of interest
        phylon = phylon of interest
        cutoff = cutoff for the phylon contribution
        
    output: p-value of the enrichment

    '''
    from scipy import stats

    clusters_in_phylon = L[L[phylon]>phylon_contribution_cutoff][phylon].index # TODO consider other ways to generate this cutoff (e.g. quantile, mass %, etc)
    
    clusters_w_function= cluster_to_go_functions[cluster_to_go_functions['go'] == function]['cluster'].values.tolist()
    len(clusters_w_function)
    overlap = set(clusters_in_phylon).intersection(set(clusters_w_function))
    common_proteins = '; '.join(functions2genes[functions2genes['cluster'].isin(overlap)]['product'].values.tolist())
    N = len(cluster_to_go_functions['cluster'].drop_duplicates()) # Total number of clusters
    K = len(clusters_w_function) # clusters with function in poi
    n = len(clusters_in_phylon) # Number of clusters in the phylon
    k = len(overlap) # Number of genes in the phylon with function in poi
    p_value = stats.hypergeom.sf(k - 1, N, K, n)
    
    return {'function': function, 'N':N, 'K':K, 'n':n, 'k':k, 'p_value':p_value, 'products':common_proteins}


def calc_all_phylon_go_enrichments(L, all_functions, cluster_to_go_functions, go_functions, phylon_contribution_cutoff=0):
    '''
    calculates the enrichment of all functions in all phylons
    input: 
        L = L matrix from NMF
        cluster_to_go_functions dataframe from get_cluster_to_go_functions function
        go_functions = go functions dataframe
        cutoff = cutoff for the phylon contribution
    '''
    import numpy as np
    out = []
    for phylon in L.columns:
        for function in go_functions.index:
            res = calc_enrichment(L, cluster_to_go_functions, function, all_functions, phylon, phylon_contribution_cutoff=phylon_contribution_cutoff)
            # print(function, res['p_value'])
            out.append({
                'phylon': phylon,
                'function': function,
                'p_value': res['p_value'],
                'genes in phylon': res['n'],
                'genes w function': res['K'],
                'overlap': res['k'],
                'products': res['products']
            })
    out = pd.DataFrame(out).sort_values('p_value')
    out['logp'] = -1 * out['p_value'].apply(np.log10)

    #TODO: add FDR

    return out

def get_go_mapping():
    '''
    returns a dataframe of go terms
    loads from the data/go_terms.csv file - need to keep this updated
    '''
    
    # go_mapping = os.path.join('../pyphylon/data', 'go_terms.csv')
    go_mapping = os.path.join(os.sep.join(__file__.split(os.sep)[:-2]), 'pyphylon', 'data', 'go_terms.csv')
    print(f'loaded go terms from {go_mapping}')
    go_mapping = pd.read_csv(go_mapping, index_col=0)
    return go_mapping


def explode_go_annos(functions):
    '''
    generates a dataframe with the cluster, phylon, and function
    input: L matrix from NMF and functions dataframe and a function dataframe
    
    output: dataframe with cluster, phylon, and function
    '''
    acc_functions = functions # don't filter by accessory genome...

    cluster_to_go_functions = acc_functions[['cluster','go']]
    cluster_to_go_functions.loc[:,'go'] = cluster_to_go_functions.loc[:,'go'].str.split(', ')
    cluster_to_go_functions = cluster_to_go_functions.explode('go')
    cluster_to_go_functions
    
    return cluster_to_go_functions.reset_index(drop=True)



def gen_phylon_wordcloud(L, functions, phylon, cutoff=0,  save=False, filename='phylon_wordcloud.png'):
    '''
    generates a wordcloud of the functions in a phylon
    input: 
        L = L matrix from NMF
        functions = functions dataframe
        phylon = phylon of interest
        cutoff = cutoff for the phylon contribution
    '''
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    coi = L[L[phylon]>cutoff][phylon].index
    
    cluster_functions = functions[functions['cluster'].isin(coi)]['product']
    # print(cluster_functions.tolist())

    # Combine all terms into a single string
    terms_string = ' '.join(cluster_functions).lower() \
    .replace('protein','') \
    .replace('hypothetical', '') \
    .replace('uncharacterized','') \
    .replace('putative','') \
    .replace('domain','') \
    .replace('containing','') \
    .replace('family','') \
    .replace('like','') \
    .replace('protein','') \
    .replace('related','') \
    .replace('conserved','') \
    .replace('possible','') \
    .replace('unknown','') \
    .replace('transcriptional regulator','') \
    .replace('subunit','') \
    .replace('type','') \
    .replace('system','')
        
    

    # Generate the word cloud
    wordcloud = WordCloud(width=1200, height=1200, background_color='white').generate(terms_string)

    # Display the word cloud
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(phylon)
    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()