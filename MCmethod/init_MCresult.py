import csv
import json
import os
import sys
import numpy as np

# read current path
fileDir = os.path.join(os.getcwd())
sys.path.append(fileDir)

#获取上级目录
#fileDir_creat = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
#fileup = os.path.abspath(os.path.dirname(os.getcwd()))

# there are five obligors in each sectors
nsenario = 10
vden = 0
vnen = [0]*100
eden = 0
enen = [0]*100

contri_dict = {k:{'vden': vden, 'vnen': vnen, 'eden': eden,'enen':enen} for k in '123456789'}
contri_dict["10"] = {'vden': vden, 'vnen': vnen, 'eden': eden,'enen':enen}

# run params
objects = ["sector", "obligor","rc_obligor"]
MODEL = ["CBV1","CBV2","CBV3"]

for ob in objects:

    RUN_FOLDER1 = os.path.join(fileDir, 'MCResult/{}'.format(ob))

    if not os.path.isdir(RUN_FOLDER1):
        os.makedirs(RUN_FOLDER1)

    for model in MODEL:
        PATH_MODEL = os.path.join(RUN_FOLDER1, model)
        if not os.path.isdir(PATH_MODEL):
            os.mkdir(PATH_MODEL)

        with open(PATH_MODEL + "/ismc_data.csv", 'w') as csvfile:
            writer1 = csv.writer(csvfile)
            a = list(map(str, np.arange(1, nsenario + 1)))
            column_name=list(map(lambda x: "loss" + x, a)) + list(map(lambda x: "likehood" + x, a))
            writer1.writerow(column_name)
            print("success initialize the ISMC result file")

        with open(PATH_MODEL + "/pmc_loss.csv", 'w') as csvfile:
            writer2 = csv.writer(csvfile)
            senarios = np.arange(1, nsenario + 1)
            writer2.writerow(list(map(str, senarios)))
            print("success initialize the PMC result file")

        if ob == "rc_obligor":

            with open(PATH_MODEL + '/IS_contri_data.json', 'w') as json_file:
                json.dump(contri_dict, json_file)
                print("success initial the contribution data")

            with open(PATH_MODEL + '/P_contri_data.json', 'w') as json_file:
                json.dump(contri_dict, json_file)
                print("success initial the contribution data")




#a = pan.read_csv(PATH_MODEL + '/pmc_loss.csv')

