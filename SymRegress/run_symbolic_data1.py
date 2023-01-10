import pickle
import argparse
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
from gplearn.genetic import SymbolicClassifier
from sklearn.utils.random import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sympy import symbols, sympify, expand
from sklearn import metrics

parser = argparse.ArgumentParser(description='distill')
parser.add_argument('-n_sim', '--n_sim', type=int, default=10000)
parser.add_argument('-nmax', '--nmax', type=int, default=100000)
parser.add_argument('-len_max', '--len_max', type=int, default=3)
args0 = parser.parse_args()

string = "symbolic_data1" + "_n_sim_" + str(args0.n_sim) + "_nmax_" + str(args0.nmax) + "_len_max_" + str(args0.len_max)

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

def simply_rule(rule,len_x):
    rule_copy = copy.deepcopy(rule)
    for i in range(len(rule_copy)):
        if str(type(rule_copy[i])) == "<class 'gplearn.functions._Function'>":
            rule_copy[i] = rule_copy[i].name    

    stack = []
    while rule_copy != []:
        item = rule_copy.pop()
        if type(item)==int:
            stack.append("X"+str(item))
        elif type(item)==float:
            stack.append(str(item))
        elif item == "add":
            value1 = stack.pop()
            value2 = stack.pop()
            stack.append("("+value1+"+"+value2+")")
        elif item == "mul":
            value1 = stack.pop()
            value2 = stack.pop()
            stack.append("("+value1+"*"+value2+")")
        elif item == "sub":
            value1 = stack.pop()
            value2 = stack.pop()
            stack.append("("+value1+"-"+value2+")")
        elif item in ["sin","cos","log","sig"]:
            value = stack.pop()
            stack.append("("+value+")")

    for i in range(len_x):
        locals()['X'+str(i)] = symbols('X'+str(i))
    eq = expand(sympify(stack[0]))

    eq_str = str(eq)
    eq_str_new=''
    skip = False

    for k in eq_str:
        if skip == True and k not in ["0","1","2","3","4","5","6","7","8","9"]:
            skip=False
        if k!="." and not skip:
            if k=="-":
                k = "+"
            eq_str_new += k
        elif k==".":
            eq_str_new = eq_str_new[:-1]
            eq_str_new += str(1)
            skip = True

    eq_str_new = str(expand(sympify(eq_str_new)))
    # print(eq_str_new)

    count = 0    
    for k in eq_str_new:
        if count>1 and k in ["X","s","c","l"]  and eq_str_new[count-1]=="*":
            p = count-2
            while eq_str_new[p] in ["0","1","2","3","4","5","6","7","8","9"," "]:
                p = p - 1
                if p==-1:
                    break
            if p==-1:
                eq_str_new = eq_str_new[count:]
                count = 1
                continue
            elif eq_str_new[p] in ["+","-","*","("]:
                eq_str_new = eq_str_new[0:(p+1)]+eq_str_new[count:]
                count = p+2
                continue
        count += 1

    eq_str_new = str(expand(sympify(eq_str_new)))
    # print("after: ",eq_str_new,"\n")
    return eq_str_new

def model_sample(seed,X,X_sim,n,F_funct,sim_gen,loss_f,num_sim,n_gen,max_depth,init_depth,function_set):
    F_pred = F_funct(X_sim)

    est_gp =  SymbolicClassifier(population_size=num_sim,
                    generations=n_gen, stopping_criteria=0.01,
                    p_crossover=0.7, p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                    max_samples=0.9, verbose=1,
                    parsimony_coefficient=0, random_state=seed,init_depth=init_depth,function_set=function_set)

    X_sim0 = sim_gen(X,n)
    y_sim0 = F_funct(X_sim0)
    est_gp.fit(X_sim0, y_sim0,max_depth)

    group = []
    loss_class = []
    loss_mean_ls = []
    for k in range(num_sim):
        rule = copy.deepcopy(est_gp._programs[-1][k])

        S_pred = est_gp._programs[-1][k].execute(X_sim)
        S_pred = est_gp._transformer(S_pred)
        loss = loss_f(F_pred,S_pred)

        rule_str = simply_rule(rule.program,X_sim0.shape[1])
        if rule_str not in group:
            group.append(rule_str)
            loss_class.append(loss)
            loss_mean_ls.append(np.mean(loss))
            if k==0 or np.mean(loss)==np.min(loss_mean_ls):
                model_best = copy.deepcopy(est_gp._programs[-1][k])
        else:
            idx = group.index(rule_str)
            loss_mean = np.mean(loss)
            if loss_mean < loss_mean_ls[idx]:
                loss_mean_ls[idx] = loss_mean
                loss_class[idx] = loss
                if np.mean(loss)==np.min(loss_mean_ls):
                    model_best = copy.deepcopy(est_gp._programs[-1][k])

    if len(group)==1:
        return -1, group, None, model_best
    else:
        order =list(np.argsort(loss_mean_ls))
        return order[0], group, loss_class, model_best

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
            if var_j==0:
                continue
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

def model_select(seed,X,F_funct,sim_gen,loss_f,num_sim,n_gen,max_depth,nmax,alpha,n_init,n_step,init_depth,function_set):
    np.random.seed(seed+100)
    n = n_init
    group_step = []
    for step in range(n_step):
        X_sim = sim_gen(X,n)
        id_x, group, loss_class, model_best =  model_sample(seed,X,X_sim,n,F_funct,sim_gen,loss_f,num_sim,n_gen,max_depth,init_depth,function_set)
        group_step.append(group)
        rule_str = copy.deepcopy(group[id_x])

        if step == 0:
            rule_str0 = copy.deepcopy(rule_str)
            F_pred = F_funct(X_sim)

            clf =  SymbolicClassifier(population_size=num_sim,
                            generations=n_gen, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0, random_state=seed,init_depth=init_depth,function_set=function_set)
            clf.fit(X_sim, F_pred,max_depth)
            rule_fit0 = simply_rule(clf._program.program,X.shape[1])

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
    clf =  SymbolicClassifier(population_size=num_sim,
                    generations=n_gen, stopping_criteria=0.01,
                    p_crossover=0.7, p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                    max_samples=0.9, verbose=1,
                    parsimony_coefficient=0, random_state=seed,init_depth=init_depth,function_set=function_set)
    clf.fit(X_sim, F_pred,max_depth)
    rule_fit = simply_rule(clf._program.program,X.shape[1])

    np.random.seed(0)
    X_sim = sim_gen(X,10000)
    F_pred = F_funct(X_sim)
    S_pred = model_best.execute(X_sim)
    S_pred = clf._transformer(S_pred)
    fpr, tpr, thresholds = metrics.roc_curve(F_pred, S_pred, pos_label=1)
    loss_best = metrics.auc(fpr, tpr)

    return rule_str0, rule_str, n, group_step, rule_fit0, rule_fit, loss_best


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
loss_best_ls = []
for seed in tqdm(range(100)):
    rule_str0, rule_str, n, group_step, rule_fit0,rule_fit, loss_best = model_select(seed,X,F_funct,x_gen_binary,loss_cross,num_sim=args0.n_sim
                                                                  ,n_gen=10,max_depth=args0.len_max,nmax=args0.nmax,alpha=0.05,n_init=1000,n_step=10,init_depth=(1, 2),function_set=('add', 'mul'))
    rule_nonstab.append(rule_str0)
    rule_stab.append(rule_str)
    n_ls.append(n)
    group_ls.append(group_step)
    rule_fit0_ls.append(rule_fit0)
    rule_fit_ls.append(rule_fit)
    loss_best_ls.append(loss_best)

    result = [rule_nonstab,rule_stab,n_ls,group_ls,rule_fit0_ls,rule_fit_ls,loss_best_ls]
    with open('result/'+string+'.pkl', 'wb') as f:
        pickle.dump(result, f)