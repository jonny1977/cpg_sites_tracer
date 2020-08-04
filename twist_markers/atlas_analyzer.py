import numpy as np
import pandas as pd

def collect_data_to_df(grail_file, max_sequence_length):
    full_df = pd.read_csv(grail_file, sep = '\t')
    relevant_df = full_df[full_df.end - full_df.start <= max_sequence_length]
    samples = relevant_df.columns[5:]

    return relevant_df, samples

def select_markers(relevant_df, cpg_threshold, samples, samples_number):
    markers = []
    cpg_array = relevant_df[samples].to_numpy()
    blocks = [tuple(x) for x in relevant_df[['chr', 'start', 'end', 'startCpG', 'endCpG']].to_numpy()]
    C = get_meth_samples(cpg_array, cpg_threshold)
    T = get_meth_samples(1 - cpg_array, cpg_threshold)
    relevant_lines = np.logical_or(np.logical_and(C==samples_number, T==samples.size-samples_number),
                                   np.logical_and(T==samples_number, C==samples.size-samples_number))
    for ind, relevant_line in enumerate(relevant_lines):
        if relevant_line:
            cpg_stat = cpg_array[ind,:]
            if C[ind] == samples_number:
                relevant_samples_ind = np.where(cpg_stat > cpg_threshold)[0]
            else:
                relevant_samples_ind = np.where(cpg_stat < 1 - cpg_thresh)[0]
            relevant_samples = samples[relevant_samples_ind]
            markers.append({
                'samples': relevant_samples, 'block': blocks[ind], 'cpg_stat': cpg_stat})

    return markers

def get_meth_samples(cpg_array, cpg_threshold):

    cpg_temp_array = cpg_array.copy()
    cpg_temp_array[np.isnan(cpg_array)] = -999
    relevant_samples = np.sum(cpg_temp_array > cpg_threshold, axis=1)

    return relevant_samples


def extract_markers_by_samples_groups(relevant_df, cpg_threshold, samples, max_samples_number):
    markers = []
    for i in range(max_samples_number):
        markers_i = select_markers(relevant_df, cpg_threshold, samples, i+1)
        print('extracted {0} markers appropriate for {1} samples, cpg_threshold = {2}'.format(len(markers_i), i+1, cpg_threshold))
        markers += markers_i
    markers_summary = [(marker['samples'], marker['block'][2] - marker['block'][1]) for marker in markers]

    return markers_summary, markers

def extract_samples_similarity(markers_summary, samples):
    similarity_matrix = np.zeros([samples.size, samples.size])
    samples_dict = {sample:i for i,sample in enumerate(samples)}
    for marker in markers_summary:
        similarity_matrix = adjust_similarity_matrix(similarity_matrix, marker, samples_dict)

    return similarity_matrix

def adjust_similarity_matrix(similarity_matrix, marker, samples_dict):
    marker_samples = list(marker[0])
    for i in range(len(marker_samples)):
        for j in range(i+1, len(marker_samples)):
            similarity_matrix[samples_dict[marker_samples[i]], samples_dict[marker_samples[j]]] += 1
            similarity_matrix[samples_dict[marker_samples[j]], samples_dict[marker_samples[i]]] += 1

    return similarity_matrix

def find_similar_samples(samples, similarity_matrix):
    for i, sample in enumerate(samples):
        line = similarity_matrix[i]
        ind = np.argsort(-line)
        print('{0}: {1} / {2}, {3} / {4}, {5} / {6}, {7} / {8}, {9} / {10}'.format(sample, samples[ind[0]], line[ind[0]],
                samples[ind[1]], line[ind[1]], samples[ind[2]], line[ind[2]], samples[ind[3]], line[ind[3]],
                samples[ind[4]], line[ind[4]]))



grail_file = '/home/yehonatan/personal/YDTK/data/table.s150.p15.5-500.tsv'
cpg_threshold = 0.25
max_samples_number = 60
relevant_df, samples = collect_data_to_df(grail_file, 150)
markers_summary, markers = extract_markers_by_samples_groups(relevant_df, cpg_threshold, samples, max_samples_number)