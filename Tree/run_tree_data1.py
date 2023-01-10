import pickle
import argparse
import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn import tree
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
from sklearn.metrics import log_loss

parser = argparse.ArgumentParser(description='distill')
parser.add_argument('-n_sim', '--n_sim', type=int, default=100)
parser.add_argument('-nmax', '--nmax', type=int, default=100000)
parser.add_argument('-len_max', '--len_max', type=int, default=3)
args0 = parser.parse_args()

string = "tree_data1" + "_n_sim_" + str(args0.n_sim) + "_nmax_" + str(args0.nmax) + "_len_max_" + str(args0.len_max)

# cross entropy loss
def loss_cross(F_pred,S_pred):
    return -F_pred*np.log(S_pred + 1e-7) -(1-F_pred)*np.log(1-S_pred+ 1e-7)

# define the sythetic function for X_sim
# def x_gen_binary(X,n):
#     binomial_p = np.mean(X,0)
#     X_sim = []
#     for i in range(X.shape[1]):
#         X_sim.append(np.random.binomial(1,binomial_p[i],n))
#     X_sim = np.array(X_sim).T
#     return X_sim

def x_gen_binary(X,n_sample):
    candidates = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]

    U = np.random.uniform(0,1,n_sample*X.shape[1]).reshape([n_sample,-1])
    X_sample = np.zeros([n_sample,X.shape[1]],dtype=int)
    flip_prop = 0.1
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            X_sample[i,j] = X[i%X.shape[0],j]
            if j not in [9,10,11] and U[i,j]<flip_prop:
                X_sample[i,j] = 1- X_sample[i,j]
        if U[i,9]<flip_prop:
            choices = list(range(4))
            choices.remove(candidates.index(list(X_sample[i,[9,10,11]])))
            index = np.random.choice(choices,size=1)[0]
            X_sample[i,[9,10,11]] = candidates[index]
    np.random.shuffle(X_sample)  
    return X_sample

def model_sample(seed,X,X_sim,n,F_funct,sim_gen,loss_f,num_sim,max_depth):
    F_pred = F_funct(X_sim)
    group = []
    clf_class = []
    loss_class = []
    loss_mean_ls = []
    for k in range(num_sim):
        X_sim0 = sim_gen(X,n)
        y_sim0 = F_funct(X_sim0)
        clf = DecisionTreeClassifier(max_depth = max_depth,criterion='entropy')
        clf.fit(X_sim0, y_sim0)
        rule = clf.tree_.feature.tolist()

        S_pred = clf.predict_proba(X_sim)[:,1]
        loss = loss_f(F_pred,S_pred)

        if rule not in group:
            group.append(rule)
            loss_class.append(loss)
            loss_mean_ls.append(np.mean(loss))
            clf_class.append(clf)
        else:
            idx = group.index(rule)
            loss_mean = np.mean(loss)
            if loss_mean < loss_mean_ls[idx]:
                loss_mean_ls[idx] = loss_mean
                loss_class[idx] = loss
                clf_class[idx] = clf

    if len(group)==1:
        return -1, group, None, clf
    else:
        order =list(np.argsort(loss_mean_ls))
        return order[0], group, loss_class, clf_class

def stablize_nsample(X_sim,loss_class,id_x,nmax,alpha):
    n = X_sim.shape[0]
    loss_best = loss_class[id_x]
    loss_rest = copy.deepcopy(loss_class)
    loss_rest.pop(id_x)
    
    test_update = False
    n_new = n
    while n_new<=nmax:
        # calculate the test statistics for each j and the required sample size for stablization
        p_ls = []
        for j in range(len(loss_rest)):
            d_j = np.mean(loss_rest[j] - loss_best)
            var_j = np.var(loss_rest[j] - loss_best)
            z_n = d_j/np.sqrt(2*var_j/n_new)
            p_j = 1 - stats.norm.cdf(z_n)
            p_ls.append(p_j)

        if np.sum(p_ls) < alpha:
            break
        else:
            test_update = True
            if n_new == nmax:
                break
            n_new += 0.1 * n
            n_new = np.min([n_new,nmax])

    return test_update, n_new

def model_select(seed,X,F_funct,sim_gen,loss_f,num_sim,max_depth,nmax,alpha,n_init,n_step):
    np.random.seed(seed+100)
    n = n_init
    for step in range(n_step):
        X_sim = sim_gen(X,n)
        id_x, group, loss_class, clf_class =  model_sample(seed,X,X_sim,n,F_funct,sim_gen,loss_f,num_sim,max_depth)

        rule_str = copy.deepcopy(group[id_x])

        if step == 0:
            rule_str0 = copy.deepcopy(rule_str)
            F_pred = F_funct(X_sim)
            clf = DecisionTreeClassifier(max_depth = max_depth,criterion='entropy')
            clf.fit(X_sim, F_pred)
            rule_fit0 = clf.tree_.feature.tolist()  

        if len(group)==1:
            break

        if n>=nmax:
            break
        else:
            test_update, n_new = stablize_nsample(X_sim,loss_class,id_x,nmax,alpha)
            if test_update == True:
                n = int(n_new)
            else:
                break

    F_pred = F_funct(X_sim)
    clf = DecisionTreeClassifier(max_depth = max_depth,criterion='entropy')
    clf.fit(X_sim, F_pred)
    rule_fit = clf.tree_.feature.tolist()   
    try:
        return rule_str0, rule_str, n, group, rule_fit0, rule_fit, clf_class[id_x]
    except:
        return rule_str0, rule_str, n, group, rule_fit0, rule_fit, clf_class


data_file = '../uci_mammo_data.csv'

data = pd.read_csv(data_file, index_col = None)
feature_names = data.columns
X = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])

for i in range(X.shape[0]):
    if tuple(X[i,[9,10,11]])==(1,0,0):
        X[i,[9,10,11]] = [0,0,0]
    elif tuple(X[i,[9,10,11]])==(0,0,0):
        X[i,[9,10,11]] = [1,0,0]
    elif tuple(X[i,[9,10,11]])==(0,1,0):
        X[i,[9,10,11]] = [0,1,0]
    elif tuple(X[i,[9,10,11]])==(0,1,1):
        X[i,[9,10,11]] = [0,0,1]

# fit the teacher model
regr = RandomForestClassifier(random_state=0)
regr.fit(X, y)
F_funct = regr.predict

rule_nonstab = []
rule_stab = []
n_ls = []
group_ls = []
rule_fit0_ls = []
rule_fit_ls = []
clf_class_ls = []
for seed in tqdm(range(100)):
    rule_str0, rule_str, n, group, rule_fit0,rule_fit, clf_best = model_select(seed,X,F_funct,x_gen_binary,loss_cross,num_sim=args0.n_sim,max_depth=args0.len_max,nmax=args0.nmax,alpha=0.05,n_init=1000,n_step=10)
    rule_nonstab.append(rule_str0)
    rule_stab.append(rule_str)
    n_ls.append(n)
    group_ls.append(group)
    rule_fit0_ls.append(rule_fit0)
    rule_fit_ls.append(rule_fit)
    clf_class_ls.append(clf_best)
    
    result = [rule_nonstab,rule_stab,n_ls,group_ls,rule_fit0_ls,rule_fit_ls, clf_class_ls]
    with open('result/'+string+'.pkl', 'wb') as f:
        pickle.dump(result, f)