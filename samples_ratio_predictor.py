##### IMPORTS #####
import pandas as pd
import numpy as np
import pickle
import os
import gzip
from collections import Counter
from datetime import datetime, timedelta
import subprocess
import shlex
import argparse
import random
from matplotlib import pyplot as plt

from cpg_blocks_identifier import CpgBlocksIdentifier

##### GLOBALS #####
MIN_SITES_PER_READ = 4
MAX_READ_LENGTH = 150
MIN_SITES_PER_ORIGINAL_READ = 5
METHYLATION_THRESHOLD = 0.3
MAX_RELEVANT_SAMPLES_PER_SITE = 3
METHYLIZED_ONLY = False
NUM_OF_TRAIN_MIXES = 3
TOTAL_NUM_OF_MIXES = 7
FIRST_REPETITION = 1
LAST_REPETITION = 6
MAX_CHUNK_SIZE = 10000000
PROBABILITY_THRESHOLD = 1000.0
RATIOS_CALC_METHOD = 'prob_sum' # 'argmax'
CHROMOSOMES = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
               'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
               'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM']

##### CLASS #####
class SamplesRatioPredictor:
    ##### INITIALIZATION #####
    def __init__(self, min_sites_per_read=MIN_SITES_PER_READ, max_read_length=MAX_READ_LENGTH,
                 min_sites_per_original_read=MIN_SITES_PER_ORIGINAL_READ, methylation_threshold=METHYLATION_THRESHOLD,
                 max_relevant_samples_per_site = MAX_RELEVANT_SAMPLES_PER_SITE, methylized_only=METHYLIZED_ONLY,
                 num_of_train_mixes=NUM_OF_TRAIN_MIXES, max_chunk_size=MAX_CHUNK_SIZE,
                 probability_threshold=PROBABILITY_THRESHOLD):
        # clustering parameters
        self.min_sites_per_read = min_sites_per_read
        self.num_of_train_mixes = num_of_train_mixes
        self.max_chunk_size = max_chunk_size
        self.cpg_blocks_identifier = CpgBlocksIdentifier(max_read_length, min_sites_per_original_read,
                 methylation_threshold, max_relevant_samples_per_site, probability_threshold, methylized_only)
        self.set_files_structure()
        self.set_training_params()

    def set_files_structure(self):
        self.input_folder = self.cpg_blocks_identifier.output_folder
        cpg_dict = pickle.load(open(self.cpg_blocks_identifier.relevant_sites_cpg_dict_file, 'rb'))
        self.chromosomes = cpg_dict.keys()
        self.blocks_file = self.cpg_blocks_identifier.blocks_file
        self.blocks_bed_file = self.cpg_blocks_identifier.blocks_bed_file
        self.truncated_mixing_folder = os.path.join(self.input_folder, 'truncated_mixing_mspr-{mspr}', '{sample}_{suffix}')
        self.truncated_mixing_file = 'mix_{mix_num}_rep_{rep_num}.pat'
        self.reads_per_sample_folder = os.path.join(self.input_folder, 'reads_per_sample_mspr-{mspr}', '{sample}_{suffix}')
        self.reads_per_sample_file = 'mix_{mix_num}_rep_{rep_num}.pkl'
        self.analysis_results_folder = os.path.join(self.input_folder, 'results')
        create_folder(self.analysis_results_folder, 'results folder')
        self.analysis_results_file = os.path.join(self.analysis_results_folder, 'results.pkl')

    def set_training_params(self):
        self.train_mixes = range(self.num_of_train_mixes)
        self.test_mixes = range(self.num_of_train_mixes, TOTAL_NUM_OF_MIXES)
        self.repetitions = range(FIRST_REPETITION, LAST_REPETITION)
        # set input / output files structure
        self.mixing_folder = '/cs/zbio/sapir/data/mixing'
        self.cf_suffix = 'cfDNA_mixing_800X'
        self.wbc_suffix = 'WBC_mixing_800X'
        self.mix_files_map_file = 'mix_files_map.csv'
        groups_dict = pickle.load(open(self.cpg_blocks_identifier.groups_file, 'rb'))
        self.groups = groups_dict['groups']
        self.cf_priors = extract_priors('/cs/cbio/sapir/deconv/mix/grail/UXM/prior_cfDNA.l3.csv', self.groups)
        self.wbc_priors = extract_priors('/cs/cbio/sapir/deconv/mix/grail/UXM/prior_WBC.l3.csv', self.groups)
        self.mixing_file = os.path.join(self.mixing_folder, '{sample}_{suffix}', 'mix_{mix_num}_rep_{rep_num}.pat.gz')
        self.mixing_samples = list(set([f.replace(self.wbc_suffix, '')[:-1] for f in os.listdir(self.mixing_folder) if
                                        f.endswith(self.wbc_suffix)]))

    ##### MIXING FILES TRUNCATION #####
    def generate_truncated_mixing_files(self, sample=None, suffix=None, mix_num=None, rep_num=None):
        mixing_params_list = self.extract_mixing_params(sample, suffix, mix_num, rep_num, create_folders=True)
        random.shuffle(mixing_params_list)
        for mixing_params in mixing_params_list:
            if not os.path.exists(mixing_params['truncated_mixing_file']):
                t_start = datetime.now()
                shell_args = ['tabix', '-R', self.blocks_bed_file, mixing_params['mixing_file']]
                with open(mixing_params['truncated_mixing_file'], 'w') as f:
                    subprocess.call(shell_args, stdout=f)
                    print(f"generated {mixing_params['truncated_mixing_file']} in {datetime.now() - t_start}")
            else:
                print(f"skipped {mixing_params['truncated_mixing_file']}, file already exists")

    def split_mixing_files_to_samples(self, sample=None, suffix=None, mix_num=None, rep_num=None):
        mixing_params_list = self.extract_mixing_params(sample, suffix, mix_num, rep_num, create_folders=True)
        random.shuffle(mixing_params_list)
        for mixing_params in mixing_params_list:
            t_start = datetime.now()
            if not os.path.exists(mixing_params['reads_per_sample_file']):
                self.split_mixing_file_to_samples(mixing_params['truncated_mixing_file'],
                                                  mixing_params['reads_per_sample_file'])
                print(f"generated {mixing_params['reads_per_sample_file']} in {datetime.now() - t_start}")
            else:
                print(f"skipped {mixing_params['reads_per_sample_file']}, file already exists")

    def split_mixing_file_to_samples(self, truncated_mixing_file, reads_per_sample_file):
        reads_per_sample = {}
        mixing_data = open(truncated_mixing_file, 'r').readlines()
        read_lines = [read_line.replace('\n', '').split('\t') for read_line in mixing_data]
        truncated_mixing_ratio = extract_truncated_mixing_ratio(read_lines)
        read_lines = [read_line for read_line in read_lines if len(read_line[2]) >= self.min_sites_per_read]
        for chromosome in CHROMOSOMES:
            chromosome_reads = [read_line for read_line in read_lines if read_line[0] == chromosome]
            blocks = pickle.load(open(self.blocks_file.format(chromosome=chromosome), 'rb'))
            blocks_ind = 0
            for read in chromosome_reads:
                blocks_ind = evaluate_read_likelihood(read, blocks, reads_per_sample, self.groups,
                                                      self.min_sites_per_read, blocks_ind)
        for key, value in reads_per_sample.items():
            value['likelihood'] = np.array(value['likelihood'])
            value['count'] = np.array(value['count'])
            value['sample'] = np.array(value['sample'])
            value['truncated_mixing_ratio'] = 0
            is_sample = np.array([sample in key for sample in value['sample']])
            value['selected_reads_ratio'] = np.sum(is_sample * value['count']) / np.sum(value['count'])
            for k, v in truncated_mixing_ratio.items():
                if k in key:
                    value['truncated_mixing_ratio'] += v
        with open(reads_per_sample_file, 'wb') as f:
            pickle.dump(reads_per_sample, f)


    ##### MIXING FILES RATIO EVALUATION #####
    def evaluate_samples_ratio(self, sample=None, suffix=None, mix_num=None, rep_num=None):
        mixing_params_list = self.extract_mixing_params(sample, suffix, mix_num, rep_num, create_folders=False)
        random.shuffle(mixing_params_list)
        for mixing_params in mixing_params_list:
            t_start = datetime.now()
            self._evaluate_samples_ratio(mixing_params['reads_per_sample_file'], mixing_params['suffix'])
            t_end = datetime.now()
            print(f"predicted samples ratio for {mixing_params['reads_per_sample_file']} in {t_end-t_start}")

    def _evaluate_samples_ratio(self, reads_per_sample_file, suffix):
        reads_per_sample_dict = pickle.load(open(reads_per_sample_file, 'rb'))
        group2ind = extract_groups_index(self.groups, reads_per_sample_dict.keys())
        prior = self.wbc_priors if suffix == self.wbc_suffix else self.cf_priors
        prior_array = extract_extended_prior(prior, group2ind)
        calculate_posterior(reads_per_sample_dict, prior_array, group2ind)
        with open(reads_per_sample_file, 'wb') as f:
            pickle.dump(reads_per_sample_dict, f)

    ##### RESULTS COLLECTION #####
    def analyze_results(self, sample=None, suffix=None, mix_num=None, rep_num=None):
        ratio_analysis_results = {}
        mixing_params_list = self.extract_mixing_params(sample, suffix, mix_num, rep_num, create_folders=False)
        t_start = datetime.now()
        for i, mixing_params in enumerate(mixing_params_list):
            self.collect_results(mixing_params, ratio_analysis_results)
            if i%50 == 0:
                print(f"analyzed {i} out of {len(mixing_params_list)} files in {datetime.now() - t_start}")
                t_start = datetime.now()
        with open(self.analysis_results_file, 'wb') as f:
            pickle.dump(ratio_analysis_results, f)
        print(f"collected model results into {self.analysis_results_file}")

    def collect_results(self, mixing_params, ratio_analysis_results):
        sample = mixing_params['sample']
        suffix = mixing_params['suffix']
        mix_num = mixing_params['mix_num']
        map_file = os.path.join(self.mixing_folder, f'{sample}_{suffix}', 'mix_files_map.csv')
        sample_dict = {'goal_ratios': np.array([]), 'selected_reads_ratios': np.array([]),
                       'argmax_estimation': np.array([]), 'prob_sum_estimation': np.array([])}
        if sample not in ratio_analysis_results.keys():
            ratio_analysis_results.update({sample:
                                    {'cfDNA_mixing_800X': sample_dict.copy(), 'WBC_mixing_800X': sample_dict.copy()}})
        ratio_analysis_results[sample][suffix]['goal_ratios'] = np.append(
            ratio_analysis_results[sample][suffix]['goal_ratios'], parse_goal_ratio(map_file, mix_num))
        reads_per_sample_dict = pickle.load(open(mixing_params['reads_per_sample_file'], 'rb'))
        if sample in reads_per_sample_dict.keys():
            selected_reads_ratio = reads_per_sample_dict[sample]['selected_reads_ratio']
            argmax_estimation = reads_per_sample_dict[sample]['argmax_estimation']
            prob_sum_estimation = reads_per_sample_dict[sample]['prob_sum_estimation']
        else:
            selected_reads_ratio, argmax_estimation, prob_sum_estimation = np.nan, np.nan, np.nan
        ratio_analysis_results[sample][suffix]['selected_reads_ratios'] = np.append(
            ratio_analysis_results[sample][suffix]['selected_reads_ratios'], selected_reads_ratio)
        ratio_analysis_results[sample][suffix]['argmax_estimation'] = np.append(
            ratio_analysis_results[sample][suffix]['argmax_estimation'], argmax_estimation)
        ratio_analysis_results[sample][suffix]['prob_sum_estimation'] = np.append(
            ratio_analysis_results[sample][suffix]['prob_sum_estimation'], prob_sum_estimation)

    ##### SUPPORT METHODS #####
    def extract_mixing_params(self, sample, suffix, mix_num, rep_num, create_folders):
        samples = self.mixing_samples if sample is None else [sample]
        suffixes = [self.wbc_suffix, self.cf_suffix] if suffix is None else [suffix]
        mix_nums = range(TOTAL_NUM_OF_MIXES) if mix_num is None else [mix_num]
        rep_nums = self.repetitions if mix_num is None else [rep_num]

        mixing_params_list = [{
            'mixing_file': self.mixing_file.format(sample=sam, suffix=suf, mix_num=mn, rep_num=rn),
            'truncated_mixing_file': os.path.join(
                self.truncated_mixing_folder.format(mspr=self.min_sites_per_read, sample=sam, suffix=suf),
                self.truncated_mixing_file.format(mix_num=mn, rep_num=rn)),
            'reads_per_sample_file': os.path.join(
                self.reads_per_sample_folder.format(mspr=self.min_sites_per_read, sample=sam, suffix=suf),
                self.reads_per_sample_file.format(mix_num=mn, rep_num=rn)),
            'suffix': suf, 'sample': sam, 'mix_num': mn, 'rep_num': rn} \
            for sam in samples for suf in suffixes for mn in mix_nums for rn in rep_nums]

        if create_folders:
            output_folders = [self.truncated_mixing_folder.format(mspr=self.min_sites_per_read, sample=sam, suffix=suf) \
                          for sam in samples for suf in suffixes] + \
                [self.reads_per_sample_folder.format(mspr=self.min_sites_per_read, sample=sam, suffix=suf) \
                          for sam in samples for suf in suffixes]
            for output_folder in output_folders:
                create_folder(output_folder, 'output folder')

        return mixing_params_list


def extract_priors(priors_file, groups):
    priors = open(priors_file, 'r').readlines()
    priors = [prior.replace('\n', '').split(',') for prior in priors]
    priors_sum = np.sum([float(prior[1]) for prior in priors])
    priors = {prior[0]: float(prior[1]) / priors_sum for prior in priors}
    for prior in priors.keys():
        if prior not in groups:
            raise AssertionError(f'{prior} from priors file is not in groups file {groups}')
    return priors


def extract_cpg_indices(blocks_file, chromosome):
    blocks = pickle.load(open(blocks_file.format(chromosome=chromosome), 'rb'))
    cpg_py_indices = np.array([])
    for block in blocks:
        cpg_py_indices = np.append(cpg_py_indices, block['relevant_cpg_sites'])
    return cpg_py_indices.astype(int)


def evaluate_read_old(chr_read_line, cpg_py_indices, blocks_ind, reads, min_sites_per_read):
    chr_done = False
    read_start_ind = int(chr_read_line[1]) - 1
    read_indices = read_start_ind + np.array([i for i, r in enumerate(chr_read_line[2]) if r != '.'])
    while cpg_py_indices[blocks_ind] < read_start_ind and blocks_ind < cpg_py_indices.size - 1:
        blocks_ind += 1
    if cpg_py_indices.size - blocks_ind < min_sites_per_read:
        chr_done = True
    else:
        overlap = np.intersect1d(read_indices, cpg_py_indices[blocks_ind:blocks_ind + len(chr_read_line[2])]).size
        if overlap >= min_sites_per_read:
            reads.append((read_start_ind, chr_read_line[2], int(chr_read_line[3]), chr_read_line[4]))
    return chr_done, blocks_ind


def set_read_group(relevant_samples, groups):
    samples_indices = np.sort(relevant_samples).astype(int)
    read_group = ':'.join(groups[sample_ind] for sample_ind in samples_indices)

    return read_group


def calculate_posterior(reads_per_sample_dict, prior_array, group2ind):
    updated_prior = prior_array.copy()
    for i in range(5):
        posterior_probability = evaluate_posterior_probability(reads_per_sample_dict, updated_prior, group2ind)
        updated_prior = update_prior(updated_prior, posterior_probability)


def evaluate_posterior_probability(reads_per_sample_dict, prior, group2ind):
    posterior_probability = np.zeros(prior.size)
    for group, value in reads_per_sample_dict.items():
        group_ind = group2ind[group]
        likelihood_array = extract_likelihood_array(group, value['likelihood'], group2ind)
        prior_array = np.tile(prior, (likelihood_array.shape[0], 1))
        posterior_array = likelihood_array * prior_array
        argmax_estimation = (np.argmax(posterior_array, axis=1) == group_ind)
        prob_sum_estimation = posterior_array[:, group_ind] / posterior_array.sum(axis=1)
        value.update({'argmax_estimation': np.sum(argmax_estimation * value['count']) / value['count'].sum(),
                      'prob_sum_estimation': np.sum(prob_sum_estimation * value['count']) / value['count'].sum()})
        posterior_probability[group_ind] = np.sum(prob_sum_estimation * value['count']) / value['count'].sum()

    return posterior_probability


def extract_likelihood_array(group, original_likelihood_array, group2ind):
    likelihood_array = np.zeros([original_likelihood_array.shape[0], len(group2ind)])
    likelihood_array[:, :original_likelihood_array.shape[1]] = original_likelihood_array
    if ':' in group:
        group_ind = group2ind[group]
        sub_groups_ind = [group2ind[subgroup] for subgroup in group.split(':')]
        group_likelihood = likelihood_array[:, sub_groups_ind].mean(axis=1)
        likelihood_array[:, group_ind] = group_likelihood
        likelihood_array[:, sub_groups_ind] = 0

    return likelihood_array


def update_prior(prior, posterior):
    updated_prior = prior + 1.2 * (posterior - prior)
    updated_prior[updated_prior < 1e-6] = 1e-6
    updated_prior[updated_prior > 1 - 1e-6] = 1 - 1e-6
    updated_prior /= updated_prior.sum()

    return updated_prior


def extract_groups_index(unique_groups, all_groups):
    group2ind = {group: i for i, group in enumerate(unique_groups)}
    combined_groups = [group for group in all_groups if ':' in group]
    group2ind.update({combined_group: i + len(unique_groups) for i, combined_group in enumerate(combined_groups)})
    return group2ind


def extract_extended_prior(prior, group2ind):
    prior_array = np.zeros(len(group2ind))
    for extended_group, ind in group2ind.items():
        extended_groups = extended_group.split(':')
        prior_array[ind] = max(np.sum([prior[group] for group in extended_groups]), 0.0005)
    return prior_array


def evaluate_read_likelihood(read, blocks, reads_per_sample, groups, min_sites_per_read, blocks_ind):
    read_start_ind = int(read[1]) - 1
    read_all_indices = np.arange(read_start_ind, read_start_ind + len(read[2]))
    read_relevant_indices = read_start_ind + np.array([i for i, r in enumerate(read[2]) if r != '.'])
    # trace relevant block
    while read_start_ind > blocks[blocks_ind]['all_cpg_sites'][-1]:
        blocks_ind += 1
    block = blocks[blocks_ind]
    overlap = np.intersect1d(read_relevant_indices, block['relevant_cpg_sites']).size
    if overlap < min_sites_per_read:
        return blocks_ind
    # extract likelihood per sample
    relevant_block_indices = np.in1d(block['all_cpg_sites'], read_all_indices)
    likelihood_array = block['methylation_array'][relevant_block_indices, :]
    # adjust likelihood for sites status (C/T/.)
    for i in range(likelihood_array.shape[0]):
        if read[2][i] == 'T':
            likelihood_array[i, :] = 1 - likelihood_array[i, :]
        elif read[2][i] == '.':
            likelihood_array[i, :] = 1
    finite_lines = (np.sum(~np.isfinite(likelihood_array), axis=1) == 0)
    likelihood_array = likelihood_array[finite_lines, :]
    norm_factor = 1 / np.max(likelihood_array.mean(axis=1))
    likelihood = np.prod(norm_factor * likelihood_array, axis=0)
    if np.sum(likelihood) == 0:
        return blocks_ind

    likelihood /= likelihood.sum()
    group = set_read_group(block['relevant_sample'], groups)
    if group not in reads_per_sample.keys():
        reads_per_sample.update({
            group: {'likelihood': [], 'count': [], 'sample': [], 'reads': [], 'cpg_start_ind': []}})
    reads_per_sample[group]['likelihood'].append(likelihood)
    reads_per_sample[group]['count'].append(int(read[3]))
    reads_per_sample[group]['sample'].append(read[4])
    reads_per_sample[group]['reads'].append(read[2])
    reads_per_sample[group]['cpg_start_ind'].append(read_start_ind)

    return blocks_ind


def extract_truncated_mixing_ratio(read_lines):
    count = np.array([int(read_line[3]) for read_line in read_lines])
    samples = np.array([read_line[4] for read_line in read_lines])
    truncated_mixing_ratios = {sample: np.sum((samples == sample) * count) / np.sum(count) for sample in np.unique(samples)}
    return truncated_mixing_ratios


def parse_goal_ratio(map_file, mix_num):
    try:
        map_data = open(map_file, 'r').readlines()[mix_num + 1]
        goal_ratio = float(map_data.split(',')[1])
        return goal_ratio
    except:
        return np.nan

##### LOGGING AND MONITORING #####
def create_folder(folder_name, folder_type):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'generated {folder_type}: {folder_name}')

def eval_likelihood(likelihood):
    if np.sum(~np.isfinite(likelihood)) > 0:
        print(f'likelihood: {likelihood}')
        return False
    if np.sum(likelihood) <= 0:
        print(f'likelihood: {likelihood}')
        return False
    return True


def results_visualization(results_file, sample, blocks_per_sample_dict):
    results_dict = pickle.load(open(results_file, 'rb'))
    wbc_dict = results_dict[sample]['WBC_mixing_800X']
    cf_dict = results_dict[sample]['cfDNA_mixing_800X']

    fp_rate = np.nanmin(wbc_dict['prob_sum_estimation'])
    wbc_argmax_estimation = wbc_dict['argmax_estimation'] - np.nanmin(wbc_dict['argmax_estimation'])
    wbc_prob_sum_estimation = wbc_dict['prob_sum_estimation'] - np.nanmin(wbc_dict['prob_sum_estimation'])
    wbc_argmax_ratio = wbc_argmax_estimation / wbc_dict['selected_reads_ratios']
    wbc_prob_sum_ratio = wbc_prob_sum_estimation / wbc_dict['selected_reads_ratios']

    cf_argmax_estimation = cf_dict['argmax_estimation'] - np.nanmin(cf_dict['argmax_estimation'])
    cf_prob_sum_estimation = cf_dict['prob_sum_estimation'] - np.nanmin(cf_dict['prob_sum_estimation'])
    cf_argmax_ratio = cf_argmax_estimation / cf_dict['selected_reads_ratios']
    cf_prob_sum_ratio = cf_prob_sum_estimation / cf_dict['selected_reads_ratios']

    if np.sum(np.isfinite(wbc_argmax_ratio)) + np.sum(np.isfinite(wbc_prob_sum_ratio)) + \
            np.sum(np.isfinite(cf_argmax_ratio)) + np.sum(np.isfinite(cf_prob_sum_ratio)) > 0:
        plt.figure()
        plt.loglog(wbc_dict['selected_reads_ratios'], wbc_prob_sum_ratio, '.')
        plt.loglog(wbc_dict['selected_reads_ratios'], wbc_argmax_ratio, '.')
        plt.loglog(cf_dict['selected_reads_ratios'], cf_prob_sum_ratio, '.')
        plt.loglog(cf_dict['selected_reads_ratios'], cf_argmax_ratio, '.')
        plt.title(f'{sample}, {blocks_per_sample_dict[sample]} blocks\n background noise = {fp_rate}')
        plt.legend(['wbc / prob_sum', 'wbc / argmax', 'cf / prob_sum', 'cf / argmax'])
        plt.savefig(f'/home/yehonatan/personal/YDTK/figures/{sample}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='samples_ratio_predictor argument parser')
    parser.add_argument('--min_sites_per_read', default=MIN_SITES_PER_READ, type=int)
    parser.add_argument('--max_read_length', default=MAX_READ_LENGTH, type=int)
    parser.add_argument('--min_sites_per_original_read', default=MIN_SITES_PER_ORIGINAL_READ, type=int)
    parser.add_argument('--methylation_threshold', default=METHYLATION_THRESHOLD, type=float)
    parser.add_argument('--max_relevant_samples_per_site', default=MAX_RELEVANT_SAMPLES_PER_SITE, type=int)
    parser.add_argument('--methylized_only', action='store_true')
    parser.add_argument('--num_of_train_mixes', default=NUM_OF_TRAIN_MIXES, type=int)
    parser.add_argument('--max_chunk_size', default=MAX_CHUNK_SIZE, type=int)
    parser.add_argument('--probability_threshold', default=PROBABILITY_THRESHOLD, type=float)
    args = parser.parse_args()
    samples_ratio_predictor = SamplesRatioPredictor(args.min_sites_per_read, args.max_read_length,
                     args.min_sites_per_original_read, args.methylation_threshold, args.max_relevant_samples_per_site,
                     args.methylized_only, args.num_of_train_mixes, args.max_chunk_size, args.probability_threshold)
    """
    samples_ratio_predictor.generate_truncated_mixing_files(sample=None)
    samples_ratio_predictor.split_mixing_files_to_samples(sample=None)
    samples_ratio_predictor.split_mixing_files_to_chromosomes(sample='Small-Int-Ep',
                                                              suffix=samples_ratio_predictor.wbc_suffix,
                                                              mix_num=1, rep_num=1)
    samples_ratio_predictor.evaluate_samples_ratio(sample=None)
    """
    samples_ratio_predictor.analyze_results()
    """
    results_file = '/home/yehonatan/personal/YDTK/results.pkl'
    sample = 'Small-Int-Ep'
    ensemble = 'cfDNA_mixing_800X'
    results_visualization(results_file, sample, ensemble)
    """