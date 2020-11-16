##### IMPORTS #####
import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
import argparse

##### GLOBALS #####
MAX_READ_LENGTH = 150
MIN_SITES_PER_READ = 5
MIN_SITES_PER_OBSERVED_READ = 4
METHYLATION_THRESHOLD = 0.3
MAX_RELEVANT_SAMPLES_PER_SITE = 3
PROBABILITY_THRESHOLD = 1000.0
CHROMOSOMES = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
               'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
               'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM']


##### CLASS #####
class CpgBlocksIdentifier:

    ##### INITIALIZATION #####
    def __init__(self, max_read_length=MAX_READ_LENGTH, min_sites_per_read=MIN_SITES_PER_READ,
                 methylation_threshold=METHYLATION_THRESHOLD,
                 max_relevant_samples_per_site=MAX_RELEVANT_SAMPLES_PER_SITE,
                 probability_threshold=PROBABILITY_THRESHOLD, methylized_only=False):
        # clustering parameters
        self.max_read_length = max_read_length
        self.min_sites_per_read = min_sites_per_read
        self.methylation_threshold = methylation_threshold
        self.max_relevant_samples_per_site = max_relevant_samples_per_site
        self.probability_threshold = probability_threshold
        self.methylized_only = methylized_only
        # set input / output files structure
        self.set_files_structure()
        self.extract_samples_stats_and_groups()

    def set_files_structure(self):
        """
        set input / output files path
        """
        # input files
        self.input_folder = '/cs/cbio/yehonatan/twist/input_data'
        self.cpg_bed_file = os.path.join(self.input_folder, 'CpG.bed')
        self.beta_folder = os.path.join(self.input_folder, 'beta_files')
        self.samples_group_file = os.path.join(self.input_folder, 'groups.csv')

        # results folder
        self.output_folder = '/cs/cbio/yehonatan/twist/output_data/mrl-{mrl}_mspr-{mspr}_mt-{mt}_ms-{ms}'.format(
            mrl=self.max_read_length, mspr=self.min_sites_per_read,
            mt=self.methylation_threshold, ms=self.max_relevant_samples_per_site)
        if self.methylized_only:
            self.output_folder += '_methylized_only'
        create_folder(self.output_folder, 'output folder')
        # cpg dict files
        self.cpg_dict_folder = os.path.join(self.output_folder, 'cpg_dicts')
        create_folder(self.cpg_dict_folder, 'cpg dict folder')
        self.all_sites_cpg_dict_file = os.path.join(self.cpg_dict_folder, 'cpg_dict_all.pkl')
        self.crowded_sites_cpg_dict_file = os.path.join(self.cpg_dict_folder, 'cpg_dict_crowded.pkl')
        self.definite_sites_cpg_dict_file = os.path.join(self.cpg_dict_folder, 'cpg_dict_definite.pkl')
        self.relevant_sites_cpg_dict_file = os.path.join(self.cpg_dict_folder, 'cpg_dict_relevant.pkl')

        # group beta files
        self.groups_beta_folder = os.path.join(self.output_folder, 'groups_beta_folder')
        create_folder(self.groups_beta_folder, 'groups beta folder')

        self.groups_file = os.path.join(self.groups_beta_folder, 'group_structure.pkl')
        self.group_beta_file = os.path.join(self.groups_beta_folder, '{group_name}_beta.pkl')

        #all sites methylation folder
        self.all_sites_folder = os.path.join(self.output_folder, 'all_sites')
        create_folder(self.all_sites_folder, 'all sites folder')
        self.all_sites_chromosome_methylation_file = os.path.join(
            self.all_sites_folder, 'all_sites_methylation_array_{chromosome}.pkl')

        # definite sites folder
        self.definite_sites_folder = os.path.join(self.output_folder, 'definite_sites')
        create_folder(self.definite_sites_folder, 'definite sites folder')
        self.definite_chromosome_methylation_file = os.path.join(
            self.definite_sites_folder, 'definite_methylation_array_{chromosome}.pkl')

        # relevant sites folder
        self.relevant_sites_folder = os.path.join(self.output_folder, 'relevant_sites')
        create_folder(self.relevant_sites_folder, 'relevant sites folder')
        self.relevant_chromosome_methylation_file = os.path.join(
            self.definite_sites_folder, 'relevant_methylation_array_{chromosome}.pkl')

        # initial blocks folder
        self.blocks_folder = os.path.join(self.output_folder, 'blocks')
        create_folder(self.blocks_folder, 'initial blocks')
        self.blocks_file = os.path.join(self.blocks_folder, 'blocks_{chromosome}.pkl')
        self.blocks_bed_file = os.path.join(self.blocks_folder, 'blocks.bed')

    def extract_samples_stats_and_groups(self):
        """
        generate groups name / structure file based on input grouping file
        """
        # skip if groups file already exists
        if os.path.exists(self.groups_file):
            groups_dict = pickle.load(open(self.groups_file, 'rb'))
            self.groups = groups_dict['groups']
            self.groups_structure = groups_dict['groups_structure']
            return
        samples_data = open(self.samples_group_file, 'r').readlines()
        samples_data = [sample.split(',') for sample in samples_data[1:]]
        samples_data = [(sample_data[2], sample_data[0].replace('# ', ''), float(sample_data[5])) for sample_data \
                        in samples_data if
                        len(sample_data) == 6 and sample_data[4] == 'True' and ':' not in sample_data[2]]
        self.groups = sorted(list(set([sample_data[0] for sample_data in samples_data])))
        self.groups_structure = {group: [] for group in self.groups}
        for sample_data in samples_data:
            self.groups_structure[sample_data[0]].append((sample_data[1], sample_data[2]))
        with open(self.groups_file, 'wb') as f:
            pickle.dump({'groups': self.groups, 'groups_structure': self.groups_structure}, f)

    def generate_group_beta_files(self):
        """
        create beta files per samples group
        """
        # build beta file per group
        beta_files = os.listdir(self.beta_folder)
        for group_name in self.groups_structure.keys():
            samples = [sample[0] for sample in self.groups_structure[group_name]]
            group_beta_files = [os.path.join(self.beta_folder, beta_file) for beta_file in beta_files
                                if beta_file.replace('.beta', '') in samples]
            if len(samples) != len(group_beta_files):
                print(
                    f'missing beta_files for group {group_name}. expected samples: {samples}. existing files: {group_beta_files}')

            for i, beta_file in enumerate(group_beta_files):
                if i == 0:
                    beta_array = np.fromfile(beta_file, dtype=np.uint8).reshape((-1, 2)).astype(float)
                else:
                    beta_array += np.fromfile(beta_file, dtype=np.uint8).reshape((-1, 2)).astype(float)
            methylation_array = np.nan * np.ones(beta_array.shape[0])
            valid = np.logical_and(beta_array.sum(axis=1) > 0, np.isfinite(beta_array.sum(axis=1)))
            methylation_array[valid] = beta_array[:, 0][valid] / beta_array[:, 1][valid]
            print_methylation_stat(group_name, methylation_array, self.methylation_threshold)

            with open(self.group_beta_file.format(group_name=group_name), 'wb') as f:
                pickle.dump(methylation_array, f)

    ##### METHODS #####
    def extract_cpg_sites_dict(self):
        """
        convert bed file to dict of the structure: {chromosome: {cpg_py_ind: [], chr_ind: []},}
        result is stored at cpg_dict_file
        """
        # read csv to numpy arrays
        cpg_df = pd.read_csv(self.cpg_bed_file, sep='\t', names=['chromosome', 'chr_ind', 'cpg_ind'])
        chromosome = np.array(cpg_df.chromosome)
        chromosome_ind = np.array(cpg_df.chr_ind, dtype=int)
        cpg_ind = np.arange(chromosome_ind.size)
        unique_chromosomes = np.unique(chromosome)
        cpg_dict = {}
        for unique_chromosome in unique_chromosomes:
            ind = (chromosome == unique_chromosome)
            cpg_dict.update({
                unique_chromosome: {
                    'cpg_py_ind': cpg_ind[ind], 'chromosome_ind': chromosome_ind[ind]}})
        with open(self.all_sites_cpg_dict_file, 'wb') as f:
            pickle.dump(cpg_dict, f)
        print(f'extracted cpg_sites_dict to {self.all_sites_cpg_dict_file}')

    def extract_crowded_cpg_sites_dict(self):
        """
        find cpg sites that has at least min_sites_per_read within a read of max_read_length
        sites data is collected from cpg_sites_dict, and written to an array per chromosome and saved at crowded_cpg_dict_file
        """
        all_sites_cpg_dict = pickle.load(open(self.all_sites_cpg_dict_file, 'rb'))
        crowded_cpg_dict = {chromosome: extract_crowded_indices(indices, self.min_sites_per_read, self.max_read_length)
                            for chromosome, indices in all_sites_cpg_dict.items()}
        for chromosome in all_sites_cpg_dict.keys():
            all_sites = all_sites_cpg_dict[chromosome]['chromosome_ind'].size
            crowded_sites = crowded_cpg_dict[chromosome]['chromosome_ind'].size
            print(f'{crowded_sites} out of {all_sites} crowded sites were traced for {chromosome}, ' + \
                  f'({100 * crowded_sites / all_sites:.2f}%). ' + \
                  f'min_sites_per_read = {self.min_sites_per_read}, max_read_length = {self.max_read_length}')

        with open(self.crowded_sites_cpg_dict_file, 'wb') as f:
            pickle.dump(crowded_cpg_dict, f)
        print(f'extracted crowded_cpg_sites_dict to {self.crowded_sites_cpg_dict_file}')


    def extract_all_sites_methylation_array(self):
        all_sites_cpg_dict = pickle.load(open(self.all_sites_cpg_dict_file, 'rb'))
        group_beta_files = [self.group_beta_file.format(group_name=group_name) for group_name in self.groups]
        for (chromosome, values) in all_sites_cpg_dict.items():
            all_sites_chromosome_array_file = self.all_sites_chromosome_methylation_file.format(chromosome=chromosome)
            ind_start = values['cpg_py_ind'][0]
            ind_end = values['cpg_py_ind'][-1] + 1
            output_array = np.empty([ind_end - ind_start, len(self.groups) + 1])
            output_array[:, 0] = np.arange(ind_start, ind_end)
            for i, group_beta_file in enumerate(group_beta_files):
                group_array = pickle.load(open(group_beta_file, 'rb'))
                output_array[:, i + 1] = group_array[ind_start:ind_end]
            with open(all_sites_chromosome_array_file, 'wb') as f:
                pickle.dump(output_array, f)


    def extract_definitive_sites(self, remove_nans=True):
        """
        pick cpg_sites with a clear cutoff between different tissues
        """
        crowded_cpg_dict = pickle.load(open(self.crowded_sites_cpg_dict_file, 'rb'))
        definite_sites_cpg_dict = {}
        group_beta_files = [self.group_beta_file.format(group_name=group_name) for group_name in self.groups]
        # extract methylation array per chromosome
        for (chromosome, values) in crowded_cpg_dict.items():
            all_sites_chromosome_array_file = self.all_sites_chromosome_methylation_file.format(chromosome=chromosome)
            definite_chromosome_array_file = self.definite_chromosome_methylation_file.format(chromosome=chromosome)

            output_array = pickle.load(open(all_sites_chromosome_array_file, 'rb'))
            all_sites = output_array.shape[0]
            ind = np.in1d(output_array[:, 0], values['cpg_py_ind'])
            output_array = output_array[ind, :]
            output_array = select_relevant_lines(output_array, remove_nans, self.methylation_threshold,
                                                 self.max_relevant_samples_per_site, self.methylized_only, chromosome)
            with open(definite_chromosome_array_file, 'wb') as f:
                pickle.dump(output_array, f)
            cpg_py_ind = output_array[:, 0]
            chromosome_ind = values['chromosome_ind'][np.in1d(values['cpg_py_ind'], cpg_py_ind)]

            definite_sites_cpg_dict.update({chromosome: {'cpg_py_ind': cpg_py_ind, 'chromosome_ind': chromosome_ind}})
            print(
                f'{output_array.shape[0]} out of {all_sites} cpg sites ({100 * output_array.shape[0] / all_sites:.2f}% ' + \
                f'of {chromosome} are relevant for threshold of {self.methylation_threshold}')

        with open(self.definite_sites_cpg_dict_file, 'wb') as f:
            pickle.dump(definite_sites_cpg_dict, f)

    def extract_relevant_sites(self):
        definite_sites_cpg_dict = pickle.load(open(self.definite_sites_cpg_dict_file, 'rb'))
        relevant_cpg_dict = {chromosome: extract_crowded_indices(indices, self.min_sites_per_read, self.max_read_length)
                             for chromosome, indices in definite_sites_cpg_dict.items()}
        # extract relevant_cpg_dict
        for chromosome in relevant_cpg_dict.keys():
            definite_sites = definite_sites_cpg_dict[chromosome]['chromosome_ind'].size
            relevant_sites = relevant_cpg_dict[chromosome]['chromosome_ind'].size
            print(f'{relevant_sites} out of {definite_sites} relevant sites were traced for {chromosome}, ' + \
                  f'({100 * relevant_sites / max(1, definite_sites):.2f}%). ' + \
                  f'min_sites_per_read = {self.min_sites_per_read}, max_read_length = {self.max_read_length}')

        with open(self.relevant_sites_cpg_dict_file, 'wb') as f:
            pickle.dump(relevant_cpg_dict, f)
        print(f'extracted relevant_cpg_sites_dict to {self.relevant_sites_cpg_dict_file}')

    def extract_blocks(self):
        relevant_cpg_dict = pickle.load(open(self.relevant_sites_cpg_dict_file, 'rb'))
        total_blocks = 0
        for (chromosome, values) in relevant_cpg_dict.items():
            blocks_file = self.blocks_file.format(chromosome=chromosome)
            all_sites_chromosome_methylation_file = self.all_sites_chromosome_methylation_file.format(
                chromosome=chromosome)
            methylation_array = pickle.load(open(all_sites_chromosome_methylation_file, 'rb'))
            blocks = []
            cpg_indices = values['cpg_py_ind']
            chr_indices = values['chromosome_ind']
            ind_start = 0
            while ind_start < cpg_indices.size - self.min_sites_per_read:
                ind_end = ind_start + self.min_sites_per_read - 1
                done = False
                while not done:
                    if chr_indices[ind_end] - chr_indices[ind_end - self.min_sites_per_read + 1] < self.max_read_length:
                        ind_end += 1
                        if ind_end == cpg_indices.size:
                            done = True
                    else:
                        done = True
                ind_cpg_start = cpg_indices[ind_start]
                ind_cpg_end = cpg_indices[ind_end-1] + 1
                ind_chr_start = chr_indices[ind_start]
                ind_chr_end = chr_indices[ind_end-1] + 1
                relevant_cpg_sites = cpg_indices[ind_start:ind_end]
                methylation_indices = np.logical_and(methylation_array[:,0] >= ind_cpg_start, methylation_array[:,0] < ind_cpg_end)
                block_methylation_array = methylation_array[methylation_indices]
                blocks.append({'chr': chromosome,
                               'ind_chr_start': ind_chr_start, 'ind_chr_end': ind_chr_end,
                               'relevant_cpg_sites': relevant_cpg_sites, 'all_cpg_sites': block_methylation_array[:,0],
                               'methylation_array': block_methylation_array[:, 1:]})
                ind_start = ind_end
            with open(blocks_file, 'wb') as f:
                pickle.dump(blocks, f)
            print(f'found {len(blocks)} valid blocks for {chromosome}')
            total_blocks += len(blocks)
        print(f'found {total_blocks} valid blocks for entire genome')

    def evaluate_blocks_predictivity(self):
        relevant_samples = []
        for chromosome in CHROMOSOMES:
            blocks_file = self.blocks_file.format(chromosome=chromosome)
            blocks = pickle.load(open(blocks_file, 'rb'))
            for block in blocks:
                samples_optimal_probability, samples_optimal_ratio, relevant_sample = evaluate_optimal_stats(
                    block['methylation_array'].copy(), self.probability_threshold, self.max_relevant_samples_per_site,
                    self.methylized_only)
                block.update({'samples_optimal_probability': samples_optimal_probability,
                              'samples_optimal_ratio': samples_optimal_ratio,
                              'relevant_sample': relevant_sample})
                if np.isfinite(relevant_sample[0]):
                    relevant_samples.append(relevant_sample)
            blocks = [block for block in blocks if np.isfinite(block['relevant_sample'][0])]
            with open(blocks_file, 'wb') as f:
                pickle.dump(blocks, f)
        relevant_samples = [':'.join([self.groups[i] for i in relevant_sample]) for relevant_sample in relevant_samples]
        samples_count = Counter(relevant_samples)
        for group in self.groups:
            if group not in samples_count.keys():
                samples_count.update({group: 0})
        samples_count = [(key, value) for key, value in samples_count.items()]
        samples_count = sorted(samples_count, key=lambda x: len(x[0].split(':')))

        for sample_count in samples_count:
            print(sample_count[0], sample_count[1])

    def extract_blocks_bed_file_old(self):
        blocks_list = []
        for chromosome in CHROMOSOMES:
            blocks_file = self.blocks_file.format(chromosome=chromosome)
            blocks = pickle.load(open(blocks_file, 'rb'))
            for block in blocks:
                blocks_list.append([chromosome, block['ind_chr_start'], block['ind_chr_end']])
        blocks_str = [f'{block[0]}\t{block[1]}\t{block[2]}\n' for block in blocks_list]
        with open(self.blocks_bed_file, 'w') as f:
            f.writelines(blocks_str)

    def extract_blocks_bed_file(self, min_sites_per_observed_read):
        all_sites_cpg_dict = pickle.load(open(self.all_sites_cpg_dict_file, 'rb'))
        blocks_list = []
        for chromosome in CHROMOSOMES:
            ind = 0
            chromosome_ind = all_sites_cpg_dict[chromosome]['chromosome_ind']
            cpg_py_ind = all_sites_cpg_dict[chromosome]['cpg_py_ind']
            blocks_file = self.blocks_file.format(chromosome=chromosome)
            blocks = pickle.load(open(blocks_file, 'rb'))
            for block in blocks:
                ind_start = block['relevant_cpg_sites'][min_sites_per_observed_read - 1]
                while cpg_py_ind[ind] < ind_start:
                    ind += 1
                assert(cpg_py_ind[ind] == ind_start)
                chr_ind = chromosome_ind[ind] + 0
                while chromosome_ind[ind] > chr_ind - self.max_read_length:
                    ind -= 1
                cpg_ind_start = cpg_py_ind[ind] + 1 # py_indexing -> .bed indexing
                cpg_ind_end = block['relevant_cpg_sites'][-min_sites_per_observed_read] + 2 # py_indexing -> .bed indexing + exclusion
                blocks_list.append([chromosome, int(cpg_ind_start), int(cpg_ind_end)])
        blocks_str = [f'{block[0]}\t{block[1]}\t{block[2]}\n' for block in blocks_list]
        with open(self.blocks_bed_file, 'w') as f:
            f.writelines(blocks_str)
        print(f'extracted {len(blocks_list)} possible start indices')


def select_relevant_lines(output_array, remove_nans, methylation_threshold, max_samples, methylized_only, chromosome):
    if remove_nans:
        problematic_lines = np.sum(np.isnan(output_array[:, 1:]), axis=1)
        output_array = output_array[problematic_lines == 0]
        print('removing {0} out of {1} sites due to nan values for chromosome {2}'.format(
            problematic_lines.size - output_array.shape[0], problematic_lines.size, chromosome))
    samples = output_array.shape[1] - 1
    c_samples = np.sum(output_array[:, 1:] < methylation_threshold, axis=1)
    t_samples = np.sum(output_array[:, 1:] > 1 - methylation_threshold, axis=1)
    significant = (c_samples + t_samples == samples)
    minority = t_samples if methylized_only else np.minimum(t_samples, c_samples)
    relevant_lines = np.logical_and(significant, np.logical_and(minority <= max_samples, minority > 0))
    print(
        'found {0} significant lines out of {1} for {2}'.format(relevant_lines.sum(), relevant_lines.size, chromosome))

    return output_array[relevant_lines]


def extract_crowded_indices(indices, min_sites_per_read, max_read_length):
    chr_indices = indices['chromosome_ind']
    cpg_indices = indices['cpg_py_ind']
    relevant_sites = np.zeros(chr_indices.shape)
    sites_diff = chr_indices[min_sites_per_read - 1:] - chr_indices[:-min_sites_per_read + 1]
    crowded_sites = (sites_diff < max_read_length)
    for i in range(min_sites_per_read):
        relevant_sites[i:i + crowded_sites.size] += crowded_sites
    crowded_indices = {'cpg_py_ind': cpg_indices[relevant_sites > 0], 'chromosome_ind': chr_indices[relevant_sites > 0]}

    return crowded_indices


def evaluate_optimal_stats(methylation_array, probability_threshold, max_relevant_samples_per_site, methylized_only):
    problematic_sites = (np.sum(np.isnan(methylation_array), axis=1) > 0)
    methylation_array[problematic_sites, :] = 0.5
    methylation_array[methylation_array > 0.99] = 0.99
    methylation_array[methylation_array < 0.01] = 0.01
    done = False
    num_of_samples = 0
    while done == False:
        samples_optimal_probability = np.zeros(methylation_array.shape[1])
        samples_optimal_ratio = np.zeros(methylation_array.shape[1])
        num_of_samples += 1
        for sample in range(methylation_array.shape[1]):
            samples_optimal_probability[sample], samples_optimal_ratio[sample] = evaluate_sample_stat(
                methylation_array.copy(), sample, num_of_samples, methylized_only, np.sum(problematic_sites))
        if np.nanmax(samples_optimal_ratio) > probability_threshold:
            relevant_sample = np.where(samples_optimal_ratio > probability_threshold)[0]
            done = True
        elif num_of_samples > max_relevant_samples_per_site:
            relevant_sample = np.nan * np.ones(1)
            done = True
    return samples_optimal_probability, samples_optimal_ratio, relevant_sample


def evaluate_sample_stat(sample_array, sample, num_of_samples, methylized_only, problematic_sites):
    if not methylized_only:
        unmethylized_sites = (sample_array[:, sample] < 0.5)
        sample_array[unmethylized_sites, :] = 1 - sample_array[unmethylized_sites, :]
    samples_probability = np.prod(sample_array, axis=0)
    sample_probability = samples_probability[sample]

    valid_samples_probability = samples_probability[~np.isnan(samples_probability)]
    sorted_samples_probability = np.sort(valid_samples_probability)
    lower_threshold_probability = sorted_samples_probability[-1 - num_of_samples]
    upper_threshold_probability = sorted_samples_probability[-1]
    if upper_threshold_probability < np.nanmax(samples_probability):
        print(sample_array, sample, num_of_samples, samples_probability)
        raise
    if sample_probability > lower_threshold_probability:
        reference = lower_threshold_probability
    else:
        reference = upper_threshold_probability
    sample_optimal_ratio = sample_probability / max(1e-10, reference)
    sample_optimal_probability = sample_probability * 2**problematic_sites

    return sample_optimal_probability, sample_optimal_ratio

##### LOGGING AND MONITORING #####

def create_folder(folder_name, folder_type):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'generated {folder_type}: {folder_name}')


def print_methylation_stat(group, methylation_array, methylation_threshold):
    valid = np.sum(np.isfinite(methylation_array))
    total = methylation_array.size
    below_threshold = np.sum(methylation_array < methylation_threshold)
    above_threshold = np.sum(methylation_array > 1 - methylation_threshold)
    print(f'methylation_stat for group {group}:')
    print(f'found {valid} out of {total} valid sites ({100 * valid / total:.2f}%)')
    print(f'{100 * above_threshold / total:.2f}% of sites above upper methylation threshold')
    print(f'{100 * below_threshold / total:.2f}% of sites below lower methylation threshold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cpg_blocks_identifier argument parser')
    parser.add_argument('--max_read_length', default=MAX_READ_LENGTH, type=int)
    parser.add_argument('--min_sites_per_read', default=MIN_SITES_PER_READ, type=int)
    parser.add_argument('--min_sites_per_observed_read', default=MIN_SITES_PER_OBSERVED_READ, type=int)
    parser.add_argument('--methylation_threshold', default=METHYLATION_THRESHOLD, type=float)
    parser.add_argument('--max_relevant_samples_per_site', default=MAX_RELEVANT_SAMPLES_PER_SITE, type=int)
    parser.add_argument('--probability_threshold', default=PROBABILITY_THRESHOLD, type=float)
    parser.add_argument('--methylized_only', action='store_true')
    args = parser.parse_args()


    blocks_identifier = CpgBlocksIdentifier(args.max_read_length, args.min_sites_per_read, args.methylation_threshold,
                                            args.max_relevant_samples_per_site, args.probability_threshold,
                                            args.methylized_only)
    blocks_identifier.generate_group_beta_files()
    blocks_identifier.extract_cpg_sites_dict()
    blocks_identifier.extract_crowded_cpg_sites_dict()
    blocks_identifier.extract_all_sites_methylation_array()
    blocks_identifier.extract_definitive_sites()
    blocks_identifier.extract_relevant_sites()
    blocks_identifier.extract_blocks()
    blocks_identifier.evaluate_blocks_predictivity()
    blocks_identifier.extract_blocks_bed_file(args.min_sites_per_observed_read)
