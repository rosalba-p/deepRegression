
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
N = 500
lambda0 = 1.

pwd = os.getcwd().split("/")[-1]
parentdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
output_file = f"scaling_N1_N_{N}_lr_{lr}_lambda0_{lambda0}_lambda1_{lambda1}.txt"

sourceFile= open(output_file, 'a')
print('#1. N1', '2. train error', '3. train error stdev','4. test error', '5. test error stdev','6. predicted theo','7. qbar','8. n samples', file = sourceFile)
sourceFile.close()

#folder_name = "somefolder/"
folder_name = "./"
n = 50
#wanted_P = 723
wanted_P = 723
allfiles = [f for f in listdir(folder_name) if (f[:10]=="scaling_lr") ]
onlyfiles = []
for f in allfiles: 
	specs = f.split("_")
	print(specs[15])
	if specs[15] == f"{N}":
		print("yes")
		onlyfiles.append(f)

def get_scaling(file, n,wanted_P):

	file = open(file, "r")
	lines = file.readlines()
	lines.pop(0)
	#try:
	#	theo_err = float(lines[0].split(" ")[3])
	#	qbar = float(lines[0].split(" ")[4])
	#except: 
	#	theo_err, qbar = 1,1
	#	lines.pop(0)

	#last_lines = lines[-n:]

	lines = [line.split(" ") for line in lines]
	num_col = len(lines[0])
	for i in lines: 
		if len(i) != num_col:
			lines.remove(i)
	n = len(lines)
	lines = np.reshape(lines, (n,num_col))
	P = lines[:,0]
	trainerr = lines[:,1]
	std_trainerr = lines[:,2]
	testerr = lines[:,3]
	std_testerr = lines[:,4]
	std_trainerr = [float(i) for i in std_trainerr]
	trainerr = [float(i) for i in trainerr]
	P = [int(i) for i in P]
	wanted = P.index(wanted_P)
	std_testerr = [float(i) for i in std_testerr]
	testerr = [float(i) for i in testerr]
	file.close()
	
	return trainerr[wanted],std_trainerr[wanted], testerr[wanted], std_testerr[wanted]


def get_specs(f,n):
	specs = f.split("_")
	last = specs[-1].split(".")
	specs = specs[:-1] + last
	trainerr,std_trainerr, testerr, std_testerr = get_scaling(f, n,wanted_P)
	return [int(specs[-2]), trainerr,std_trainerr, testerr, std_testerr]

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
	scal_test = np.mean(mat[:,3])
	err_train = np.sqrt(np.sum(np.square(mat[:,2])))*(1/np.sqrt(len(mat[:,1])))
	err_test = np.sqrt(np.sum(np.square(mat[:,4])))*(1/np.sqrt(len(mat[:,3])))
	err_test = np.std(mat[:,3])	
	#theo_pred = np.mean(mat[:,5])
	#qbar = np.mean(mat[:,6])
	sourceFile= open(output_file, 'a')
	print(item[0][0], scal_train, err_train, scal_test, err_test, len(mat[:,1]),  file = sourceFile)
	a += 1
	sourceFile.close()
