# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:09:47 2021

@author: ice.ice
"""

# import math
import os
import sys
import time
from argparse import ArgumentParser
from functools import reduce
import configparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.sequence_encoder import ProEncoder, RNAEncoder

# default program settings
'''#######  DATA_SET = ['RPI369', 'RPI488', 'RPI1807', 'RPI2241', 'RPI1446', 'NPInter','NPInter3','NPInter227'']  ##########''' 

DATA_SET = 'NPInter227_2'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"

WINDOW_P_UPLIMIT = 3
WINDOW_P_STRUCT_UPLIMIT = 3
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
VECTOR_REPETITION_CNN = 1

script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
DATA_BASE_PATH = parent_dir + '/data/'
INI_PATH = script_dir + '/utils/data_set_settings.ini'

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='The dataset you want to process.')
args = parser.parse_args()
if args.dataset != None:
    DATA_SET = args.dataset
print("Dataset: %s" % DATA_SET)

def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            #### 数据集除NPInter3外
            # p, r, label = line.split('\t')
            #### 数据集NPInter3适用
            p, r, label = line.split()
            
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs


def read_data_seq(path):
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict


def load_data(data_set):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    # pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')
    pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')
    

    return pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs


def coding_pairs(pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind):
    samples = []
    for pr in pairs:
        if pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs:
            p_seq = pro_seqs[pr[0]]  # protein sequence
            r_seq = rna_seqs[pr[1]]  # rna sequence
            p_struct = pro_structs[pr[0]]  # protein structure
            r_struct = rna_structs[pr[1]]  # rna structure

            p_conjoint = PE.encode_conjoint(p_seq)
            r_conjoint = RE.encode_conjoint(r_seq)
            p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)
            r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)

            if p_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[0], pr))
            elif r_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[1], pr))
            elif p_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[0], pr))
            elif r_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[1], pr))

            else:
                samples.append([[p_conjoint, r_conjoint],
                                [p_conjoint_struct, r_conjoint_struct],
                                kind])
        else:
            print('Skip pair {} according to sequence dictionary.'.format(pr))
    return samples


def standardization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pre_process_data(samples, samples_pred=None):
    # np.random.shuffle(samples)

    p_conjoint = np.array([x[0][0] for x in samples])
    r_conjoint = np.array([x[0][1] for x in samples])
    p_conjoint_struct = np.array([x[1][0] for x in samples])
    r_conjoint_struct = np.array([x[1][1] for x in samples])
    y_samples = np.array([x[2] for x in samples])

    p_conjoint, scaler_p = standardization(p_conjoint)
    r_conjoint, scaler_r = standardization(r_conjoint)
    p_conjoint_struct, scaler_p_struct = standardization(p_conjoint_struct)
    r_conjoint_struct, scaler_r_struct = standardization(r_conjoint_struct)

    p_ctf_len = 7 ** WINDOW_P_UPLIMIT
    r_ctf_len = 4 ** WINDOW_R_UPLIMIT
    p_conjoint_previous = np.array([x[-p_ctf_len:] for x in p_conjoint])
    r_conjoint_previous = np.array([x[-r_ctf_len:] for x in r_conjoint])

    X_samples = [[p_conjoint, r_conjoint],
                  [p_conjoint_struct, r_conjoint_struct],
                  [p_conjoint_previous, r_conjoint_previous]
                  ]

    if samples_pred:
        # np.random.shuffle(samples_pred)

        p_conjoint_pred = np.array([x[0][0] for x in samples_pred])
        r_conjoint_pred = np.array([x[0][1] for x in samples_pred])
        p_conjoint_struct_pred = np.array([x[1][0] for x in samples_pred])
        r_conjoint_struct_pred = np.array([x[1][1] for x in samples_pred])
        y_samples_pred = np.array([x[2] for x in samples_pred])

        p_conjoint_pred = scaler_p.transform(p_conjoint_pred)
        r_conjoint_pred = scaler_r.transform(r_conjoint_pred)
        p_conjoint_struct_pred = scaler_p_struct.transform(p_conjoint_struct_pred)
        r_conjoint_struct_pred = scaler_r_struct.transform(r_conjoint_struct_pred)

        p_conjoint_previous_pred = np.array([x[-p_ctf_len:] for x in p_conjoint_pred])
        r_conjoint_previous_pred = np.array([x[-r_ctf_len:] for x in r_conjoint_pred])

        X_samples_pred = [[p_conjoint_pred, r_conjoint_pred],
                          [p_conjoint_struct_pred, r_conjoint_struct_pred],
                          [p_conjoint_previous_pred, r_conjoint_previous_pred]
                          ]

        return X_samples, y_samples, X_samples_pred, y_samples_pred

    else:
        return X_samples, y_samples


def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))


# load data settings
if DATA_SET in ['RPI369', 'RPI488', 'RPI1807', 'RPI2241', 'RPI1446', 'NPInter', 'NPInter3', 'NPInter227','NPInter227_2']:
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    WINDOW_P_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_UPLIMIT')
    WINDOW_P_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_STRUCT_UPLIMIT')
    WINDOW_R_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_UPLIMIT')
    WINDOW_R_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_STRUCT_UPLIMIT')
    CODING_FREQUENCY = config.getboolean(DATA_SET, 'CODING_FREQUENCY')



## write program parameter settings to result file
settings = (
    """# Analyze data set {}\n
Program parameters:
WINDOW_P_UPLIMIT = {},
WINDOW_R_UPLIMIT = {},
WINDOW_P_STRUCT_UPLIMIT = {},
WINDOW_R_STRUCT_UPLIMIT = {},
    """.format(DATA_SET, WINDOW_P_UPLIMIT, WINDOW_R_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT,
                WINDOW_R_STRUCT_UPLIMIT)
) 
    

PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT)
PRO_STRUCT_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT)
RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT)
RNA_STRUCT_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT)

# read rna-protein pairs and sequences from data files
pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs = load_data(DATA_SET)

# sequence encoder instances
PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY)
RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY)

print("Coding positive protein-rna pairs.\n")
samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=1)
positive_sample_number = len(samples)
print("Coding negative protein-rna pairs.\n")
samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=0)
negative_sample_number = len(samples) - positive_sample_number
sample_num = len(samples)

# positive and negative sample numbers
print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
X, y = pre_process_data(samples=samples)

protein_CT_SS = X[1][0]
RNA_kmer_SS = X[1][1]

protein_CT_SS_NP = pd.DataFrame(data=protein_CT_SS)
RNA_kmer_SS_NP = pd.DataFrame(data=RNA_kmer_SS)


protein_CT_SS_NP.to_csv(DATA_SET + '_protein_CT_SS_PN.csv',header=False,index=False)
RNA_kmer_SS_NP.to_csv(DATA_SET + '_RNA_kmer_SS_PN.csv',header=False,index=False)
