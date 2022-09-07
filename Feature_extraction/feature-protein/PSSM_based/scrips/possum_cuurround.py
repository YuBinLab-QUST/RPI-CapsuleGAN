# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:30:56 2021

@author: ice.ice
"""
def usage():
    print ("possum.py usage:")
    print ("python possum.py <options> <source files> ")
    print ("-i,--input: input a file in fasta format.")
    print ("-o,--ouput: output a file of the generated feature.")
    print ("-t,--type: specify a feature encoding algorithm.")
    print ("-p,--pssmdir: specify the directory of pssm files.")
    print ("-h,--help: show the help information.")

import fileinput
import sys, getopt
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from possum_ft import *
import os
import pandas as pd


'''
###### 
————————————————
基于PSSM的算法基于原始PSSM概要文件的矩阵转换，
可以将其分为三种类型：行转换，列转换或行列转换的混合。
这些描述符分为四组。
第一组包括AAC-PSSM (20dim)，D-FPSSM (20dim)，平滑PSSM (-dim)，AB-PSSM (400dim)，
          PSSM组合 (400dim)，RPM-PSSM (400dim)和S-FPSSM (400dim)，
它们是通过原始PSSM的行转换生成的。

第二组包含通过列转换生成的描述符，
包括DPC-PSSM (400dim)，k分隔的Bigrams-PSSM (400dim)，
           Trigram-PS-PSSM (8000dim)，EEDP (400dim)和TPC (400dim)。

第三组包括EDP (20dim)，RPSSM (110dim)，Pse-PSSM (40dim)，
           DP-PSSM (-dim)，PSSM-AC (-dim)和PSSM-CC (-dim)，
它们是通过行和列转换的混合生成的。

第四组包括AADP-PSSM (420dim)，AATP (420dim)和MEDP (420dim)，
它们仅将前三组中的描述符组合在一起。
# ————————————————
choose the algoType from 

["d_fpssm","smoothed_pssm","ab_pssm","pssm_composition","rpm_pssm","s_fpssm",
 "dpc_pssm","k_separated_bigrams_pssm","tri_gram_pssm","eedp","tpc","edp",
"rpssm","pse_pssm","dp_pssm","pssm_ac","pssm_cc","aadp_pssm","aatp","medp"]
'''

# algoType="aac_pssm"
# algoType="d_fpssm"
# algoType="smoothed_pssm"
###algoType="ab_pssm"
# algoType="pssm_composition"
# algoType="rpm_pssm"
# algoType="s_fpssm"

# algoType="dpc_pssm"
# algoType="k_separated_bigrams_pssm"
# algoType="tri_gram_pssm"
# algoType="eedp"
# algoType="tpc"

algoType="edp"
# algoType="rpssm"
# algoType="pse_pssm"
# algoType="dp_pssm"
# algoType="pssm_ac"
# algoType="pssm_cc"

# algoType="aatp"
# algoType="aadp_pssm"
# algoType="medp"


'''
Choose DATA_SET = ['RPI369', 'RPI488', 'RPI1807', 
                   'RPI2241', 'RPI1446', 'NPInter',
                   'NPInter3']
'''
# DATASET = 'RPI369'
# DATASET = 'RPI488'
# DATASET = 'RPI1807'
# DATASET = 'RPI2241'
DATASET = 'RPI1446'
# DATASET = 'NPInter'
# DATASET = 'NPInter3'
# DATASET = 'RPI_C'
# DATASET = 'RPI_D'
# DATASET = 'RPI_E'
# DATASET = 'RPI_M'
# DATASET = 'RPI_S'



script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
opts, args = getopt.getopt(sys.argv[1:], 'i:o:t:p:a:b:h', ['input=','output=','type=','pssmdir=','argument=','veriable=','help'])
argument=""
veriable=""
# 
# pssmdir = parent_dir +'/PSSM_Raw/'+DATASET +'_PSSM_P/'
# inputFile = parent_dir +'/sequence/'+DATASET +'_protein_P.fasta'
# outputf = parent_dir + '/result/EDP/' + DATASET +"_P_edp.csv"
# outputFile = parent_dir + '/result/EDP/' + DATASET + "_P_edp.txt"

pssmdir = parent_dir +'/PSSM_Raw/'+DATASET +'_PSSM_N/'
inputFile = parent_dir +'/sequence/'+DATASET +'_protein_N.fasta'
outputf = parent_dir + '/result/EDP/' + DATASET +"_N_edp.csv"
outputFile = parent_dir + '/result/EDP/' + DATASET + "_N_edp.txt"

# DATASET = 'RPI_H'
# pssmdir = parent_dir +'/PSSM_Raw/'+DATASET +'_PSSM_all/'
# inputFile = parent_dir +'/sequence/'+DATASET +'_protein_all.fasta'
# outputf = parent_dir + '/result/EDP/' + DATASET +"_all_edp.csv"
# outputFile = parent_dir + '/result/EDP/' + DATASET + "_all_edp.txt"

for opt, arg in opts:
    if opt in ('-i','--input'):
        inputFile = arg
    elif opt in ('-o','--output'):
        outputFile = arg
    elif opt in ('-t','--type'):
        algoType = arg
    elif opt in ('-p','--pssmdir'):
        pssmdir = arg
    elif opt in ('-a','--argument'):
        argument = int(arg)
    elif opt in ('-b','--veriable'):
        veriable = int(arg)
    elif opt in ('-h', '--help'):
        usage()
        sys.exit(2)
    else:
        usage()
        sys.exit(2)
check_head = re.compile(r'\>')

smplist = []
smpcnt = 0

fileinput.close()

for line, strin in enumerate(fileinput.input(inputFile)):
    if not check_head.match(strin):
        smplist.append(strin.strip())
        smpcnt += 1

onlyfiles = os.listdir(pssmdir)
onlyfiles.sort(key=lambda x:int(x.split('.')[0]))
### onlyfiles = [ f for f in listdir(pssmdir) if isfile(join(pssmdir,f)) ]

fastaDict = {}

for fi in onlyfiles:
    cntnt = ''
    pssmContentMatrix=readToMatrix(fileinput.input(pssmdir+'\\'+fi))
    pssmContentMatrix=np.array(pssmContentMatrix)
    sequence=pssmContentMatrix[:,0]
    seqLength=len(sequence)
    for i in range(seqLength):
        cntnt+=sequence[i]
    if cntnt in fastaDict:
        continue
    fastaDict[cntnt] = fi

finalist = []
for smp in smplist: 
    finalist.append(pssmdir+'\\'+fastaDict[smp])

file_out = open(outputFile,'a+')

for fi in finalist:
    input_matrix=fileinput.input(fi)
    feature_vector=calculateDescriptors(input_matrix, algoType, argument, veriable)
    np.savetxt(file_out, feature_vector)


    # csv_data = np.loadtxt(file_out)
    # csv_data=pd.DataFrame(data=csv_data)
    # csv_data.to_csv(outputf)
    
    ##### long time for waiting


'''
控制台输入

reset -f
import numpy as np
import pandas as pd
csv_data = np.loadtxt("RPI369_P_accpssm.txt")
csv_data=pd.DataFrame(data=csv_data)
csv_data.to_csv("RPI369_P_accpssm.csv")


'''




