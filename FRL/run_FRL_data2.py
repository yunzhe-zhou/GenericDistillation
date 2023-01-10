import pickle
import argparse
from distill_utils import *
import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer

parser = argparse.ArgumentParser(description='distill')
parser.add_argument('-n_sim', '--n_sim', type=int, default=10)
parser.add_argument('-n_init', '--n_init', type=int, default=1000)
parser.add_argument('-nmax', '--nmax', type=int, default=100000)
parser.add_argument('-n_sample', '--n_sample', type=int, default=1000)
parser.add_argument('-len_max', '--len_max', type=int, default=3)
args0 = parser.parse_args()

string = "FRL_data2" + "_n_sim_" + str(args0.n_sim) + "_n_init_" + str(args0.n_init)+ "_nmax_" + str(args0.nmax)+ "_n_sample_" + str(args0.n_sample) + "_len_max_" + str(args0.len_max)

#Load breast cancer dataset from sklearn
X0, y = load_breast_cancer(return_X_y=True)
X0 = X0[:,20:]
X0 = (X0-np.mean(X0,0))/np.std(X0,0)
data = load_breast_cancer()
feature_names = data.feature_names
feature_names = feature_names[20:]

est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
est.fit(X0)
X = est.transform(X0)
X = np.int64(X)
X = pd.DataFrame(X).astype(object)
X.columns = feature_names
X = pd.get_dummies(X,drop_first=True)
feature_names = X.columns
X = np.array(X)

# fit the teacher model
regr = RandomForestClassifier(random_state=0)
regr.fit(X, y)
F_funct = regr.predict

# construct the distillation class
rule_nonstab = []
rule_stab = []
n_all = []
group_all = []
len_all = []
rule_map0_ls = []
rule_map_ls = []
for seed in tqdm(range(100)):
    distill = distill_constructor(F_funct,X,loss_cross,x_gen_binary2,feature_names,alpha=0.05)
    fitter = sklearn_wrappers.monotonic_sklearn_fitter(num_steps = 5000, min_supp = 10, max_clauses = 2, prior_length_mean = 8, prior_gamma_l_alpha = 1., prior_gamma_l_beta = 0.1, temperature = 1)
    rule_str0, rule_str_ls, n_ls, group_ls, len_ls, rule_map0, rule_map = distill.model_select(seed,n_sim=args0.n_sim,fitter=fitter,n_init = args0.n_init,n_step=10,nmax=args0.nmax,n_start=1000,n_sample=args0.n_sample,testing=True,len_max=args0.len_max,truncate=True)
    rule_nonstab.append(rule_str0)
    rule_stab.append(rule_str_ls)
    n_all.append(n_ls)
    group_all.append(group_ls)
    len_all.append(len_ls)
    rule_map0_ls.append(rule_map0)
    rule_map_ls.append(rule_map)

    result = [rule_nonstab,rule_stab,n_all,group_all,len_all,rule_map0_ls,rule_map_ls,seed]
    with open('result/'+string+'.pkl', 'wb') as f:
        pickle.dump(result, f)