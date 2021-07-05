import sys
import os
import json
from IPython.display import display
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import inquirer
from scipy.stats import t
from tqdm import tqdm
import time
import pickle

notebook_path = os.path.split(os.path.abspath(os.getcwd()))[0]
display(notebook_path)
os.chdir(notebook_path)
def open_results():
    results_dir = os.path.join(notebook_path,"projects")
    list_dirs = [x[0] for x in os.walk(results_dir)]
    questions = [
    inquirer.List('folder',
                    message="Which folder do you need us to open?",
                    choices=list_dirs,
                ),
    ]
    answers = inquirer.prompt(questions)
    path = answers['folder']
    os.chdir(path)
    print(os.getcwd())
    list_files = [x for x in os.listdir(path)]
    questions=[
    inquirer.List('file',
                    message="Which file do you need us to open?",
                    choices=list_files,
                ),
    ]
    answers = inquirer.prompt(questions)
    filename = answers['file']
    with open(filename,"rb") as file:
        data = pickle.load(file)
    return data

def rmse(predictions,target,axis=0):
    rmse = np.sqrt(np.mean((predictions-target)**2,axis=axis))
    return rmse

def normalize(vec):
    new_vec = [vec[i]/np.sum(vec) for i in range(len(vec))]
    return new_vec

facets =2
if (facets==2):
    sig2 = np.array([5,15,25])
elif (facets==3):
    sig2 = np.array([5,10,25,5,5,5,2])
sig2 = np.array([4,8,16,5,5,5,2])
data = open_results()
display(normalize(sig2))
display(normalize(np.mean(data,axis=0)))
normed_data = np.array([normalize(data[i,:]) for i in range(data.shape[0])])
display(np.mean(np.abs(normed_data-normalize(sig2)),axis=0))
display(np.std(normed_data,axis=0))
display(rmse(normed_data,normalize(sig2)))