
import statistics 
import math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter 
from itertools import groupby

lambda1 = 1.
lr = 0.1
N = 200
lambda0 = 1.

pwd = os.getcwd().split("/")[-1]
parentdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
output_file = f"theory_N1_N_{N}_lr_{lr}_lambda0_{lambda0}_lambda1_{lambda1}.txt"

sourceFile= open(output_file, 'a')
print('#1. N1', '2. predicted theo', file = sourceFile)
sourceFile.close()

#folder_name = "somefolder/"
folder_name = "./"
n = 50
wanted_P = 723

allfiles = [f for f in listdir(folder_name) if (f[:9]=="theory_N_") ]
onlyfiles = []
for f in allfiles: 
	specs = f.split("_")
	print(specs[2])
	if specs[2] == f"{N}":
		print("yes")
		onlyfiles.append(f)

def get_scaling(file, n,wanted_P):
	print(file)
	file = open(file, "r")
	lines = file.readlines()
	lines.pop(0)
	n = len(lines)
	
	lines = [line.split(" ") for line in lines]
	
	num_col = len(lines[0])
	lines = np.reshape(lines, (n,num_col))
	P = lines[:,0]
	theo = lines[:,1]
	theo = [float(i) for i in theo]
	P = [int(i) for i in P]
	wanted = P.index(wanted_P)
	file.close()
	
	return theo[wanted]


def get_specs(f,n):
	specs = f.split("_")
	last = specs[-1].split(".")
	specs = specs[:-1] + last
	theo = get_scaling(f, n,wanted_P)
	print(theo)
	return [int(specs[4]), theo]

def check_file(file,n):
	file = open(file, "r")
	lines = file.readlines()
	#global theo_pred
	#theo_pred = lines[1].split(" ")[3]
	#global qbar
	#qbar = lines[1].split(" ")[4]
	if any(["nan" in line for line in lines]) or any(["inf" in line for line in lines]): 
		print(file, "\n")
		return True      
	lines.pop(0)
	if len(lines) < n:
		return True
	last_lines = lines[-n:]        	
	if len(last_lines) == n:
		return False
	else:
		return True

#to_remove = []
#for f in onlyfiles:
#	if check_file(f, n):
#		to_remove.append(f)	
#
#for f in to_remove: 
#	onlyfiles.remove(f)

scal_list = [get_specs(f,n) for f in onlyfiles]
scal_list = sorted(scal_list, key =itemgetter(0)) 

scal_list = [list(g) for _,g in groupby(scal_list, key=itemgetter(0))]


#file = open("/storage/local/sebastianoariosto/rosiwork/deepRegression/runs_new/teacher_linear_net_1hl_opt_sgd_bias_False/theory_N1_500.txt", "r")
#file = open(f"{parentdir}/theory_N_{N}_N1_{N1}_lambda0_{lambda0}_lambda1_{lambda1}.txt", "r")
#lines = file.readlines()
#theo_pred = [float(lines[i].split(" ")[1]) for i in range(len(lines))]
#qbar = [float(lines[j].split(" ")[2]) for j in range(len(lines))]


a = 0
for item in scal_list:
	mat = np.matrix(item)
	scal_train = np.mean(mat[:,1])
	#theo_pred = np.mean(mat[:,5])
	#qbar = np.mean(mat[:,6])
	sourceFile= open(output_file, 'a')
	print(item[0][0], scal_train,len(mat[:,1]),  file = sourceFile)
	a += 1
	sourceFile.close()
