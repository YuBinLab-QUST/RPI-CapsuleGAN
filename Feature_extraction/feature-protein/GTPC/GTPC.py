import pandas as pd


import re


import fileinput
import sys, getopt
from os import listdir
from os.path import isfile, join

import numpy as np
# from possum_ft import *
import os


'''
Choose DATA_SET = ['RPI369', 'RPI488', 'RPI1807', 
                   'RPI2241', 'RPI1446', 'NPInter',
                   'NPInter3']
'''
DATASET = 'RPI369'
# DATASET = 'RPI488'
# DATASET = 'RPI1807'
# DATASET = 'RPI2241'
# DATASET = 'RPI1446'
# DATASET = 'NPInter'
# DATASET = 'NPInter3'
# DATASET = 'RPI_C'
# DATASET = 'RPI_D'
# DATASET = 'RPI_E'
# DATASET = 'RPI_H'
# DATASET = 'RPI_M'
# DATASET = 'RPI_S'


script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
# opts, args = getopt.getopt(sys.argv[1:], 'i:o:t:p:a:b:h', ['input=','output=','type=','pssmdir=','argument=','veriable=','help'])
# argument=""
# veriable=""
# 
# pssmdir = parent_dir +'/PSSM_Raw/'+DATASET +'_PSSM_P/'
# inputFile = parent_dir +'/sequence/'+DATASET +'_protein_P.fasta'
# outputf = parent_dir + '/result/EDP/' + DATASET +"_P_edp.csv"
# outputFile = parent_dir + '/result/EDP/' + DATASET + "_P_edp.txt"

# inputFile = script_dir +'/sequence/'+DATASET +'_protein_P.fasta'
# outputf = script_dir + '/result/' + DATASET +"_P_GTPC.csv"

inputFile = script_dir +'/sequence/'+DATASET +'_protein_N.fasta'
outputf = script_dir + '/result/' + DATASET +"_N_GTPC.csv"



def readFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input file seems not in fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def GTPC(fastas, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	encodings = []
	header = ['#'] + triple
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])

		code = [name]
		myDict = {}
		for t in triple:
			myDict[t] = 0

		sum = 0
		for j in range(len(sequence) - 3 + 1):
			myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
			sum = sum +1

		if sum == 0:
			for t in triple:
				code.append(0)
		else:
			for t in triple:
				code.append(myDict[t]/sum)
		encodings.append(code)

	return encodings

kw=  {'path': r"RPI369_protein_P_biaohao.txt",}   
# kw=  {'path':script_dir}
fastas1 = readFasta(r"RPI369_protein_P_biaohao.txt")
# fastas1 = readFasta(inputFile)
result1=GTPC(fastas1, **kw)
data1=np.matrix(result1[1:])[:,1:]
data_=pd.DataFrame(data=data1)
# data_.to_csv('RPI369_protein_P_GTPC.csv')







