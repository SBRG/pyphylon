
from IPython import embed
import pandas as pd
from Bio import SeqIO
import json

def main():

    ''' for local testing '''
    # KLEBS: 
    genome_id = '1049565p4'         
    
    
    L_file = '/media/data/Studies/UCSD/Klebsiella_NMF/processed/nmf-outputs/L.csv' # from NMF pipeline
    A_file = '/media/data/Studies/UCSD/Klebsiella_NMF/processed/nmf-outputs/A.csv'
    outfile = 'out.html'
    features_file = f'/media/data/Studies/UCSD/Klebsiella_NMF/annotation/{genome_id}/{genome_id}.tsv'
    fna_file = f'/media/data/Studies/UCSD/Klebsiella_NMF/annotation/{genome_id}/{genome_id}.fna'
    
    features = pd.DataFrame(data['cgview']['features'])

    cluster2gene_map = pd.read_csv('/media/data/Studies/UCSD/Klebsiella_NMF/pangenome/cdhit_parsed_pg.csv')
    cluster2gene_map.index = cluster2gene_map['cluster']


    create_cgview(genome_id, features_file, fna_file, A_file, L_file, cluster2gene_map, outfile)

    

def create_cgview(genome_id, features_file, fna_file, A_file, L_file, cluster2gene_map, outfile, top_phylons=2, threshold=0.5):
    cluster2gene_map.index = cluster2gene_map['cluster']
    data = create_json_from_tsv(features_file, fna_file)
    features = pd.DataFrame(data['cgview']['features'])
    A = pd.read_csv(A_file, index_col=0)    
    L = pd.read_csv(L_file, index_col=0)

    # get top X phylons in this strain:
    top_phylons = A[genome_id].sort_values()[-top_phylons:]
    clusters_in_phylons = L[top_phylons.index.astype(str)]
    for c in clusters_in_phylons.columns: # write the outer tracks

        tmp = clusters_in_phylons[clusters_in_phylons[c]>threshold] # only show genes with L weight > threshold

        print(c, len(tmp))
        overlap = set(tmp.index.tolist()) & set(cluster2gene_map.index.tolist())
        genes_to_show = cluster2gene_map.loc[list(overlap)]['gene_id'].tolist()
        phylon_features = features[features['gene_id'].isin(genes_to_show)] 
        phylon_features.loc[:,'source'] = 'phylon' + c
        phylon_features.loc[:,'legend'] = 'phylon' + c
        del phylon_features['strand']
        d = json.loads(phylon_features.to_json(orient='records'))
        tmp = data['cgview']['features'] + d
        data['cgview']['features'] = tmp
        data['cgview']['tracks'].append(
            {
            "name": 'phylon' + c,
            "dataType": 'feature',
            "dataMethod": 'source',
            "dataKeys": 'phylon' + c
        })
    
    write_cgview_html(data, outfile)


    

def write_cgview_html(data, outfile):
    fout = open(outfile,'w')

    import json

    fout.write(HTML)
    fout.write("json=" + json.dumps(data))
    fout.write(HTML_END)
    print(f'CGView HTML written to {outfile}')
    fout.close()



def create_json_from_tsv(features_file, fna_file):

    records = SeqIO.parse(open(fna_file), 'fasta')
    contigs = []
    for record in records:
        contigs.append(
        {
            "id": record.id,
            "name": record.id,
            "orientation": "+",
            "length": len(record),
            "seq": str(record.seq)
        })
    
    data = pd.read_csv(features_file, sep='\t', skiprows=5)
    
    data = data[data['Locus Tag'].notnull()]
    
    features = []
    for i in data.index:
        start = int(data.loc[i,'Start'])
        stop = int(data.loc[i,'Stop'])
        name = str(data.loc[i,'Gene']) + ' (' + data.loc[i,'Locus Tag'] + '): ' + data.loc[i,'Product']
        gene_id = str(data.loc[i,'Locus Tag'])
        source = 'json-feature'
        legend = data.loc[i,'Type']
        strand = data.loc[i,'Strand']
        contig = data.loc[i,'#Sequence Id']
        if strand =='-':
            strand = -1
        elif strand=='+':
            strand = 1
        feature = {
            'start':start,
            'stop':stop,
            'name':name,
            'source':'json-feature', # this is the ring where the feature appears -> in "tracks" dict
            'legend':legend, # this is the color and style of the feature -> in "legend" dict
            'strand':strand,
            'contig':contig,
            'gene_id':gene_id
        }
        # print(feature)
        features.append(feature)


    legend = {
        "items": 
            [
                {
                    "name": "cds",
                    "swatchColor": "rgb(151, 151, 151)",
                    "decoration": "arrow"
                }
            ]
    }

    tracks = [
      {
        "name": "Features",
        "dataType": 'feature',
        "dataMethod": 'source',
        "dataKeys": 'json-feature'
      },
      {"name":"CG Content","thicknessRatio":2,"position":"inside","dataType":"plot","dataMethod":"sequence","dataKeys":"gc-content"},
      {"name":"CG Skew","thicknessRatio":2,"position":"inside","dataType":"plot","dataMethod":"sequence","dataKeys":"gc-skew"}
    ]

    data = {
        "cgview": {
            "version": "1.6.0",
            "sequence": {
                "contigs": contigs
            },
            "created":'date',
            "id":'test',
            "name":'test',
        'features': features,
        'legend': legend,
        'tracks': tracks
        }
    }

    return data



HTML = '''
<html>
<!-- Javscript -->
<!-- Source D3 before CGView -->
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cgview/dist/cgview.min.js"></script>
<!-- CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/cgview/dist/cgview.css">


<!-- HTML -->

<body>
    <div id="cgview"></div>
</body>


<script>
    // First create the JSON describing the map
'''

HTML_END = '''
cgv = new CGV.Viewer('#cgview', {
    height: 1500,
    width: 1500,
    });
    
    cgv.io.loadJSON(json);

    cgv.draw();

</script>
</html>

'''

if __name__=='__main__':
    main()