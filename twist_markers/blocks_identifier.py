import pandas as pd
import numpy as np
import pickle

def extract_cpg_sites_dict(cpg_bed_file, cpg_pickle_file):
    """
    convert bed file to dict of chromose + ind for each cpg site, save output to a pkl
    """
    cpg_df = pd.read_csv(cpg_bed_file, sep='\t')
    chromosome = np.array(cpg_df[cpg_df.columns[0]])
    chromosome = np.append([cpg_df.columns[0]], chromosome)
    chromosome_ind = np.array(cpg_df[cpg_df.columns[1]])
    chromosome_ind = np.append([int(cpg_df.columns[1])], chromosome_ind)
    cpg_sites_dict = {'chromosome': chromosome, 'chromosome_ind': chromosome_ind}
    with open(cpg_pickle_file,'wb') as f:
        pickle.dump(cpg_sites_dict, f)

def find_relevant_sites(cpg_sites_dict, min_sites, max_length, pkl_file = None, tsv_file = None, return_sites=True):
    """
    find sites that has at least min_sites cpg sites within distance of max_length
    """
    # pick start_points as sites that has at least min_sites in the next max_length sites (in the same chromosome)
    chromosome = cpg_sites_dict['chromosome']
    chromosome_ind = cpg_sites_dict['chromosome_ind']
    relevant_sites = np.empty(chromosome_ind.shape)
    sites_diff = chromosome_ind[min_sites-1:] - chromosome_ind[:-min_sites+1]
    crowded_sites = np.logical_and(sites_diff > 0, sites_diff < max_length)
    for i in range(min_sites):
        relevant_sites[i:i + crowded_sites.size] += crowded_sites
    relevant_sites = np.sign(relevant_sites)
    sites = []
    for i, relevant_site in enumerate(relevant_sites):
        if relevant_site:
            sites.append([chromosome[i], chromosome_ind[i], chromosome_ind[i] + 2, i + 1, i + 2])
    #write results to relevant files
    if pkl_file is not None:
        with open(pkl_file, 'wb') as f:
            pickle.dump(sites, f)
    if tsv_file is not None:
        sites_data = ['\t'.join([str(s) for s in site]) + '\n' for site in sites]
        open(tsv_file, 'w').writelines(sites_data)
    if return_sites:
        return sites

##### SANDBOX #####

# cpg_bed_file = '/home/yehonatan/personal/YDTK/data/CpG.bed'
cpg_pickle_file = '/home/yehonatan/personal/YDTK/data/CpG.pkl'
# extract_cpg_sites_dict(cpg_bed_file, cpg_pickle_file)
cpg_sites_dict = pickle.load(open(cpg_pickle_file, 'rb'))
min_sites = 5
max_length = 150
sites_pkl_file = '/home/yehonatan/personal/YDTK/data/relevant_sites_150_5.pkl'
sites_tsv_file = '/home/yehonatan/personal/YDTK/data/relevant_sites_150_5.tsv'
sites = find_relevant_sites(cpg_sites_dict, min_sites, max_length, sites_pkl_file, sites_tsv_file)