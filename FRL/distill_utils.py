from memory_profiler import memory_usage
import numpy as np
import copy
from scipy import stats
import monotonic.monotonic.sklearn_wrappers as sklearn_wrappers
import monotonic.monotonic.rule as rule
import monotonic.monotonic.model as model
import monotonic.monotonic.distributions as distributions
import monotonic.monotonic.mcmc_step_fs as mcmc_step_fs
import monotonic.monotonic.mcmc as mcmc
import monotonic.monotonic.predictors as predictors
import monotonic.monotonic.utils as monotonic_utils
import os
import psutil
from sklearn.metrics import log_loss

# mean square loss
def loss_mse(F_pred,S_pred):
    return (F_pred - S_pred)**2

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

def x_gen_binary1(X,n_sample):
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

def x_gen_binary2(X,n_sample):
    candidates = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]

    U = np.random.uniform(0,1,n_sample*10).reshape([n_sample,-1])
    X_sample = np.zeros([n_sample,X.shape[1]],dtype=int)
    flip_prop = 0.1
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            X_sample[i,[3*j,3*j+1,3*j+2]] = X[i%X.shape[0],[3*j,3*j+1,3*j+2]]
            if U[i,j]<flip_prop:
                choices = list(range(4))
                choices.remove(candidates.index(list(X_sample[i,[3*j,3*j+1,3*j+2]])))
                index = np.random.choice(choices,size=1)[0]
                X_sample[i,[3*j,3*j+1,3*j+2]] = candidates[index]
    np.random.shuffle(X_sample)   
    return X_sample


##### distillation framework #####

class distill_constructor(object):
    """
    F_funct: predictor of teacher model (input: X_sim with dim n*d, output: y_pred with dim n)
    X: covariates of the original dataset
    loss_f: loss function to evaluate the agreement between potential models and the teacher
    x_gen: generating function for samples X_sim
    alpha: significant level
    feature_names: the names of all the features
    """
    def __init__(self,F_funct,X,loss_f,x_gen,feature_names,alpha=0.05):
        self.F_funct = F_funct
        self.X = X
        self.loss_f = loss_f
        self.x_gen = x_gen
        self.alpha = alpha
        self.feature_names = feature_names

    # generate sythetic X_sim
    def sim_gen(self,X,n):
        X_sim = self.x_gen(X,n)
        return X_sim

    def sample(self,n_sim,fitter,n,X_sim,n_start,n_sample,len_max=None,truncate=False):
        F_pred = self.F_funct(X_sim)
        predictor,x_ns,y_ns,_ = fitter.fit(X_sim,F_pred,self.feature_names,True,len_max)
        rule_map = [item.x_names for item in predictor.horse.the_theta.rule_f_ls]

        group = []
        loss_class = []
        loss_mean_ls = []
        len_ls = []
        for _ in range(n_sim):
            while True:
                try:
                    X_sim0 = self.sim_gen(self.X,n)
                    y_sim0 = self.F_funct(X_sim0)
                    predictor0,x_ns0,y_ns0,classifier_constructor0 = fitter.fit(X_sim0,y_sim0,self.feature_names,False,len_max)
                    theta_dist = predictor0.horse.theta_dist
                    gamma_ls_gibbs = mcmc_step_fs.gamma_ls_gibbs_step_f(theta_dist)
                    w_ns_zeta_ns_gibbs = mcmc_step_fs.w_ns_zeta_ns_gibbs_step_f(theta_dist)
                    mcmc_step_fs_gibbs = [mcmc_step_f_constructor(theta_dist) for mcmc_step_f_constructor in classifier_constructor0.mcmc_step_f_constructors]
                    theta = theta_dist.sample(x_ns0)
                    break      
                except:
                    print("error")
            for k in range(n_start+n_sample):
                if k%100 == 0:
                   print(k)
                   process = psutil.Process(os.getpid())
                   print(process.memory_percent())
                gamma_ls_gibbs(x_ns0, y_ns0, theta).make_change(theta)
                for mcmc_step_f_gibbs in mcmc_step_fs_gibbs:
                    try:
                        theta_copy = copy.deepcopy(theta)
                        diff_f = mcmc_step_f_gibbs(x_ns0, y_ns0, theta,True)
                        diff_f(theta)
                        if len(theta.rule_f_ls)>len_max:
                            theta = theta_copy
                    except:
                        print("error")
                w_ns_zeta_ns_gibbs(x_ns0, y_ns0, theta).make_change(theta)

                if k>=n_start:
                    theta_new = copy.deepcopy(theta)
                    rule = [item.x_names for item in theta_new.rule_f_ls]

                    if rule in group:
                        continue
                    else:
                        theta_new.gamma_ls = theta_new.get_greedy_optimal_gamma_ls(x_ns, y_ns)
                        
                        predictor_s = copy.deepcopy(predictor)
                        predictor_s.horse.the_theta = theta_new
                        S_funct = predictor_s.decision_function
                        S_pred = S_funct(X_sim)
                        loss = self.loss_f(F_pred,S_pred)

                        group.append(rule)
                        loss_class.append(loss)
                        loss_mean_ls.append(np.mean(loss))

            print("iter: ",_) 
            print(len(group))    
            len_ls.append(len(group))
        if len(group)==1:
            return -1, group, None, None,rule_map
        else:
            order =list(np.argsort(loss_mean_ls))
            return order[0], group, loss_class, len_ls,rule_map

    # test whether the sample size is enough for stablization and tell how much sample size is needed
    # X_sim: sythetic dataset with dim n*d
    # loss_class: the selected list of losses of student models in different equivalent class
    # id_x: the id for the best student equivalent class
    # nmax: maximum length of the rule list
    def stablize_nsample(self,X_sim,loss_class,id_x,nmax):
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

            if np.sum(p_ls) < self.alpha:
                break
            else:
                test_update = True
                if n_new == nmax:
                    break
                n_new += 0.1 * n
                n_new = np.min([n_new,nmax])

        return test_update, n_new

    def model_select(self,seed,n_sim,fitter,n_init,n_step,nmax,n_start,n_sample,testing,len_max,truncate):
        np.random.seed(seed+100)
        n = n_init
        
        group_ls = []
        rule_str_ls = []
        n_ls = []
        len_ls = []
        for step in range(n_step):
            print(step)
            # generate sythetic dataset for fitting bayesian model
            while True:
                try:
                    X_sim = self.sim_gen(self.X,n)
                    if truncate:
                        id_x, group, loss_class, len_step, rule_map =  self.sample(n_sim,fitter,n,X_sim,n_start,n_sample,len_max,truncate)
                    else:
                        id_x, group, loss_class, len_step, rule_map =  self.sample(n_sim,fitter,n,X_sim,n_start,n_sample)
                    break
                except:
                    print("model_select_error")

            group_ls.append(group)
            len_ls.append(len_step)
            n_ls.append(n)

            if step == 0:
                rule_map0 = rule_map

            if step==0 and id_x == -1:
                rule_str0 = group[0]
                rule_str_ls.append(group[0])
                return rule_str0, rule_str_ls, n_ls, group_ls, len_ls, rule_map0, rule_map
            if step>0 and id_x == -1:
                rule_str_ls.append(group[0])
                return rule_str0, rule_str_ls, n_ls, group_ls, len_ls, rule_map0, rule_map

            rule_str = copy.deepcopy(group[id_x])

            if step == 0:
                rule_str0 = copy.deepcopy(rule_str)

            rule_str_ls.append(rule_str)

            if len(group)==1:
                break

            if n>=nmax:
                break
            else:
                test_update, n_new = self.stablize_nsample(X_sim,loss_class,id_x,nmax)
                if test_update == True:
                    n = int(n_new)
                else:
                    break

        return rule_str0, rule_str_ls, n_ls, group_ls, len_ls, rule_map0, rule_map
