import sys
import pandas as pd
from collections import defaultdict

def main(argv):
    input_file = argv[1]
    output_file = argv[2]
    metadata_file = argv[3]

    metadata = pd.read_csv(metadata_file, dtype = 'object').set_index('genome_id')
    
    new_P_matrix = defaultdict(dict)
    with open(input_file, 'r') as f:
        for line in f:
            input_name = line.split('.clstr')[0].split('/')[-1]
            with  open(line.strip(), 'r') as infile:
                cluster = ''
                present = False
                
                for l in infile:
                    if l[0] == '>':
                        if cluster != '':
                                new_P_matrix[cluster][input_name] = int(present)
                        cluster = ''
                        present = False
                    else:
                        if input_name not in l:
                            cluster = l.split('>')[1].split('...')[0].split('A')[0]
                        if input_name in l:
                            present = True
                
                if cluster != '':
                    new_P_matrix[cluster][input_name] = int(present)

    new_P_matrix = pd.DataFrame(new_P_matrix).T
    new_P_matrix.to_csv(output_file)
            

if __name__ == "__main__":
   main(sys.argv)