# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:27:35 2021

@author: ice.ice
"""
import itertools
import sys,re,os
import pandas as pd
import numpy as np

# DATASET = 'RPI_M'
# DATASET = 'RPI2241'
DATASET = 'NPInter227'

script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)

# pssmdir = parent_dir +'/PSSM_Raw/'+DATASET +'_PSSM_N/'
# inputFile = parent_dir +'/sequence/'+DATASET +'_RNA_PN.fasta'
# outputf = script_dir + '/result/' + DATASET +"_PN_MonokGap.csv"
# outputFile = parent_dir + '/result/EDP/' + DATASET + "_N_edp.txt"

# inputFile = parent_dir +'/sequence/'+DATASET +'_RNA_P.fasta'
# outputf = script_dir + '/result/' + DATASET +"_P_TNC.csv"
# inputFile = parent_dir +'/sequence/'+DATASET +'_RNA_PN.fasta'
# outputf = script_dir + '/result/' + DATASET +"_PN_TNC.csv"
inputFile = parent_dir +'/sequence/'+DATASET +'_RNA_N5.fasta'
outputf = script_dir + '/result/' + DATASET +"_N5_TNC.csv"



ALPHABET='ACGU'

def readRNAFasta(file):
	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def kmers(seq, k):
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v
def MonoKGap(x, g):  # 1___1
    t=[]
    m = list(itertools.product(ALPHABET, repeat=2))
    L_sequence=(len(x)-g-1)
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C/L_sequence)
    return t

def MonoKGap_vector(input_data,g):   
    fastas=readRNAFasta(input_data)
    vector=[] 
    header=['#']
    for f in range((g-1)*16):
        header.append('Mono.'+str(f))
    vector.append(header)   
    sample=[]
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec=MonoKGap(sequence,g)
        sample=sample+each_vec
        vector.append(sample)
    return vector


def MonoDiKGap(x, g):  # 1___2    
    t=[]
    m = list(itertools.product(ALPHABET, repeat=3))
    L_sequence=(len(x)-g-2)
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C/L_sequence) 
    return t 


def MonoDiKGap_vector(input_data,g):   
    fastas=readRNAFasta(input_data)
    vector=[] 
    header=['#']
    for f in range((g-1)*32):
        header.append('MonoDi.'+str(f))
    vector.append(header)
    sample=[]
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec=MonoDiKGap(sequence,g)
        sample=sample+each_vec
        vector.append(sample)
    return vector





# vector1 = MonoKGap_vector(inputFile, g=4)
# vector2 = MonoDiKGap_vector(inputFile, g=2)

# data_MonoKGap = np.matrix(vector1[1:])[:,1:]
# data_MonoDiKGap = np.matrix(vector2[1:])[:,1:]
# data_MonoKGap_MonoDiKGap = np.hstack((data_MonoKGap ,data_MonoDiKGap))
# csv_data=pd.DataFrame(data_MonoKGap_MonoDiKGap)
# # csv_data.to_csv(outputf)




def TNC(input_data):
    fastas=readRNAFasta(input_data)
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in ALPHABET for aa2 in ALPHABET for aa3 in ALPHABET]
    header = ['#'] + triPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(ALPHABET)):
        AADict[ALPHABET[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

vector=TNC(inputFile)
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv(outputf)









