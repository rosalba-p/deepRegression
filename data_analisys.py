
import statistics 
import math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter 
from itertools import groupby

pwd = os.getcwd().split("/")[-1]
parentdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
output_file = f"{parentdir}/scaling_{pwd}.txt"

sourceFile= open(output_file, 'a')
print(f"#1 P     #2 train error     #3 train error stdev     #4 test error     #5 test error stdev     #6 n samples", file = sourceFile)
sourceFile.close()

#folder_name = "somefolder/"
folder_name = "./"
n = 20

onlyfiles = [f for f in listdir(folder_name) if f[:3]=="run"]


def get_scaling(file, n):

	file = open(file, "r")
	lines = file.readlines()
	lines.pop(0)

	last_lines = lines[-n:]

	last_lines = [line.split(" ") for line in last_lines]
	num_col = len(last_lines[0])
	for i in last_lines: 
		if len(i) != num_col:
			last_lines.remove(i)
	n = len(last_lines)
	last_lines = np.reshape(last_lines, (n,num_col))
	trainerr = last_lines[:,1]
	trainerr = [float(i) for i in trainerr]
	testerr = last_lines[:,2]
	testerr = [float(i) for i in testerr]
	file.close()
	scaling_train = np.mean(trainerr)
	scaling_test = np.mean(testerr) 
	err_train = np.std(trainerr)
	err_test = np.std(testerr)

	return scaling_train, scaling_test, err_train, err_test 


def get_specs(f,n):
	specs = f.split("_")
	last = specs[-1].split(".")
	specs = specs[:-1] + last

	scal_train, scal_test, err_train, err_test = get_scaling(f, n)
	return [int(specs[2]), scal_train, err_train, scal_test, err_test]

def check_file(file,n):
        file = open(file, "r")
        lines = file.readlines()
        if any(["nan" in line for line in lines]) or any(["inf" in line for line in lines]): 
           print(file, "\n")
           return True       	
        lines.pop(0)
        if len(lines) < n:
             return True
        last_lines = lines[-n:]        	
        if len(last_lines) == n:
             return False
             #print(last_lines)
        else:
             return True

to_remove = []
for f in onlyfiles:
	if check_file(f, n):
		to_remove.append(f)	

for f in to_remove: 
	onlyfiles.remove(f)

scal_list = [get_specs(f,n) for f in onlyfiles]
scal_list = sorted(scal_list, key =itemgetter(0)) 

scal_list = [list(g) for _,g in groupby(scal_list, key=itemgetter(0))]
#print(scal_list, "\n")
for item in scal_list:
	mat = np.matrix(item)
	scal_train = np.mean(mat[:,1])
	scal_test = np.mean(mat[:,3])
	err_train = np.sqrt(np.sum(np.square(mat[:,2])))*(1/np.sqrt(len(mat[:,1])))
	err_test = np.sqrt(np.sum(np.square(mat[:,4])))*(1/np.sqrt(len(mat[:,3])))
	sourceFile= open(output_file, 'a')
	print(item[0][0], scal_train, err_train, scal_test, err_test, len(mat[:,1]),  file = sourceFile)
	sourceFile.close()
