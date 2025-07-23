import sys
import pandas as pd

def main(argv):
    input_file = argv[1]
    output_file = argv[2]
    metadata_file = argv[3]

    metadata = pd.read_csv(metadata_file, dtype = 'object').set_index('genome_id')
    
    mash_matrix = pd.DataFrame(index = metadata.index)
    with open(input_file, 'r') as f:
        for line in f:
            input_name = line.split('.txt')[0].split('/')[-1]
            infile = pd.read_csv(line.strip(), sep='\t', header=None, dtype='object')
            infile['pangenome_ids'] = infile.iloc[:,0].apply(lambda x: x.split('/')[-1][:-4])
            infile = infile[infile.pangenome_ids.isin(metadata.index)]
            mash_matrix[input_name] = infile.set_index('pangenome_ids')[2]
    mash_matrix.to_csv(output_file)

if __name__ == "__main__":
   main(sys.argv)