import numpy as np
import functools
import scipy.stats
import monotonic.monotonic.utils as monotonic_utils
import monotonic.monotonic.distributions as distributions
import itertools
import pdb
import pandas as pd
import monotonic.monotonic.extra_utils as caching
from functools import reduce

##### representation of a decision list #####

class barebones_theta(distributions.dist):

    @property
    def L(self):
        return len(self.rule_f_ls)

    def get_z(self, x):
        return np.argmax([rule_f(x) for rule_f in self.rule_f_ls] + [True])
    
    @property
    def rule_f_idx_ls(self):
        return tuple([rule_f_l.idx for rule_f_l in self.rule_f_ls])

    def get_z_ns(self, x_ns):
        return barebones_theta.z_ns_helper(self.rule_f_ls, x_ns)

    @staticmethod
    @caching.cache_fxn_decorator(lambda identifier, rule_f_ls, x_ns: hash(identifier) + monotonic_utils.rule_list_hash(rule_f_ls) + hash(x_ns))
    def z_ns_helper(rule_f_ls, x_ns):
        # make a num_rules x N matrix.  each rule contributes a row
#        print np.argmax(np.array([rule_f_l.batch_call(x_ns) for rule_f_l in rule_f_ls]+[np.ones(len(x_ns),dtype=bool)]), axis=0)
#         print(rule_f_ls)
        try:
            add1 = [rule_f_l.batch_call(x_ns) for rule_f_l in rule_f_ls]
            add2 = [np.ones(len(x_ns),dtype=bool)]
            count = len(add1[0])
            for item in add1:
                if count!=len(item):
                    print("error z")
                    raise ValueError("Get some error")     
            if count!=len(x_ns):
                print("error z")
                raise ValueError("Get some error")     
            return np.argmax(np.array(add1+add2), axis=0)
        except:
            print("error z_all")
            raise ValueError("Get some error")
        # try:
        #     return np.argmax(np.array([rule_f_l.batch_call(x_ns) for rule_f_l in rule_f_ls]+[np.ones(len(x_ns),dtype=bool)]), axis=0)
        # except:
        #     raise ValueError("Get some error")            

    def get_p_ns(self, x_ns):
        return barebones_theta.p_ns_helper(self.rule_f_ls, self.p_ls, x_ns)

    @staticmethod
    @caching.cache_fxn_decorator(lambda identifier, rule_f_ls, p_ls, x_ns: hash(identifier) + monotonic_utils.rule_list_hash(rule_f_ls) + monotonic_utils.array_hash(p_ls) + hash(x_ns))
    def p_ns_helper(rule_f_ls, p_ls, x_ns):
        return p_ls[barebones_theta.z_ns_helper(rule_f_ls, x_ns)]

    @caching.cache_method_decorator(lambda inst, x_ns, y_ns: hash(inst) + hash(x_ns) + hash(id(y_ns)))
    def reduced_batch_loglik(self, x_ns, y_ns):
        return np.sum(np.log(self.get_p_ns(x_ns)) * y_ns + np.log(1. - self.get_p_ns(x_ns)) * (1. - y_ns))    

    def get_data_p_ls(self, x_ns, y_ns):
        return barebones_theta.get_data_p_ls_helper(self.rule_f_ls, x_ns, y_ns)
        def temper(y):
            return max(min(y,.99), .01)
        return np.array([temper(np.mean(y_ns[self.get_z_ns(x_ns)==i]) if np.sum(self.get_z_ns(x_ns)==i) > 0 else 0.5) for i in range(0,self.L+1)])

    @staticmethod
    def get_data_p_ls_helper(rule_f_ls, x_ns, y_ns):
        L = len(rule_f_ls)
        def temper(y):
            return max(min(y,.99), .01)
        return np.array([temper(np.mean(y_ns[barebones_theta.z_ns_helper(rule_f_ls,x_ns)==i]) if np.sum(barebones_theta.z_ns_helper(rule_f_ls,x_ns)==i) > 0 else 0.5) for i in range(0,L+1)])
        
    def __hash__(self):
        return monotonic_utils.rule_list_hash(self.rule_f_ls) + monotonic_utils.array_hash(self.p_ls)

    def __init__(self, rule_f_ls, p_ls):
        self.rule_f_ls, self.p_ls = rule_f_ls, p_ls

    def informative_df(self, x_ns, y_ns):
        short_reprs = [rule_f_l.idx for rule_f_l in self.rule_f_ls] + ['default']
        long_reprs = [rule_f_l.long_repr() for rule_f_l in self.rule_f_ls] + ['default']

        z_ns = self.get_z_ns(x_ns)
        y_ns_by_z = [y_ns[z_ns==z] for z in range(self.L+1)]
        supports = np.array([len(y) for y in y_ns_by_z])
        num_poss = np.array([y.sum() for y in y_ns_by_z], dtype=float)
        pos_props = num_poss / supports

#        print np.sum([self.rule_f_ls[2](x) for x in x_ns])
#        print np.sum(self.rule_f_ls[2].batch_call(x_ns))
        
#        pdb.set_trace()
        
        overall_supports, overall_pos_props = list(map(list,list(zip(*[rule_f_l.get_support_and_pos_props(x_ns,y_ns) for rule_f_l in self.rule_f_ls]))))
        
        num_negs = supports - num_poss
        log_probs = num_poss * np.log(self.p_ls) + num_negs * np.log(1.0 - self.p_ls)
        ans = pd.DataFrame({'rule':short_reprs, 'rule_features':long_reprs, 'support':supports, 'positive_proportion':pos_props, 'logprob':log_probs, 'overall_support':overall_supports + [None], 'overall_positive_proportion':overall_pos_props + [None]})
        ans.loc[ans.shape[0]] = pd.Series({'logprob':np.sum(log_probs), 'support':np.sum(supports)})
        return ans
        
        
class reduced_theta(barebones_theta):
    """
    non-augmented model, defines distribution p(y_ns|theta;x_ns)
    length of rule_f_ls should be 1 less than that of gamma_ls
    """
    
    @property
    def r_ls(self):
        return np.log(self.v_ls)

    @property
    def p_ls(self):
        return np.array(list(map(monotonic_utils.logistic, self.r_ls)))

    @property
    def v_ls(self):
        return reduced_theta.v_ls_helper(self.gamma_ls)

    @staticmethod
    def v_ls_helper(gamma_ls):

        def reverse(l):
            return [x for x in reversed(l)]

        return np.exp(np.array(reverse(np.cumsum(reverse(np.log(gamma_ls))))))

    def get_v_ns(self, x_ns):
        return self.v_ls[self.get_z_ns(x_ns)]
    
    def __init__(self, rule_f_ls, gamma_ls):
        self.rule_f_ls, self.gamma_ls = rule_f_ls, gamma_ls

    @caching.cache_method_decorator(lambda inst, x_ns, y_ns: hash(inst) + hash(x_ns) + hash(id(y_ns)))
    def get_greedy_optimal_gamma_ls(self, x_ns, y_ns):
        data_r_ls = list(map(monotonic_utils.logit, reversed(self.get_data_p_ls(x_ns, y_ns))))
        monotonic_r_ls = np.array(reduce(lambda accum, x: accum+[max(accum+[x])], data_r_ls,[]))
        monotonic_v_ls = np.exp(monotonic_r_ls)
        return monotonic_utils.reverse_np_array(np.exp([monotonic_r_ls[l]-monotonic_r_ls[l-1] if l > 0 else monotonic_r_ls[l] for l in range(self.L+1)]))
        
    def informative_df(self, x_ns, y_ns):
        """
        for each node, show rule index, positive proportion, support at the node, p_l, gamma_l, logprob contribution, overall support of the rule, repr of the rule
        """
        z_ns = self.get_z_ns(x_ns)
        y_ns_by_z = [None for z in range(self.L+1)]
        poss = [None for z in range(self.L+1)]
        supports = [None for z in range(self.L+1)]
        short_reprs = [rule_f_l.idx for rule_f_l in self.rule_f_ls] + ['default']
        long_reprs = [rule_f_l.long_repr() for rule_f_l in self.rule_f_ls] + ['default']
        y_ns_by_z = [y_ns[z_ns==z] for z in range(self.L+1)]
        supports = np.array([len(y) for y in y_ns_by_z])
        num_poss = np.array([y.sum() for y in y_ns_by_z], dtype=float)
        pos_props = num_poss / supports
        num_negs = supports - num_poss
        log_probs = num_poss * np.log(self.p_ls) + num_negs * np.log(1.0 - self.p_ls)
        overall_supports = [rule_f_l.support for rule_f_l in self.rule_f_ls] + [1.]
        ans = pd.DataFrame({'rule':short_reprs, 'rule_features':long_reprs, 'support':supports, 'positive_proportion':pos_props, 'logprob':log_probs, 'gamma':self.gamma_ls, 'overall_support':overall_supports})
        #ans.loc[ans.shape[0]] = pd.Series({'logprob':np.sum(log_probs), 'support':np.sum(supports)})
        return ans
        
    def __hash__(self):
        return monotonic_utils.rule_list_hash(self.rule_f_ls) + monotonic_utils.array_hash(self.gamma_ls)
        
class theta(reduced_theta):
    """
    augmented model parameters.
    """
    def __init__(self, rule_f_ls, gamma_ls, w_ns, zeta_ns):
        self.w_ns, self.zeta_ns = w_ns, zeta_ns
        reduced_theta.__init__(self, rule_f_ls, gamma_ls)

    def batch_loglik(self, x_ns, y_ns):
        if np.array_equal(self.get_w_ns(x_ns) > 0, y_ns == 1):
            return 0
        else:
            assert False

    def batch_sample(self, x_ns):
        return monotonic_utils.y_data(list(map(int, np.array(self.w_ns) != 0)))
            
##### prior over z_ns #####

class zeta_ns_dist(distributions.dist):
    """
    p(zeta_ns;N)
    """
    def __init__(self, rate):
        self.rate = rate
    
    def batch_loglik(self, zeta_ns):
        return scipy.stats.expon(scale=self.rate).logpdf(zeta_ns)

    def batch_sample(self, N):
        return scipy.stats.expon(scale=self.rate).rvs(N)

##### prior over w_ns #####
        
class w_ns_given_zeta_ns_given_v_ns_dist(distributions.dist):
    """
    p(w_ns|zeta_ns, v_ns)
    """
    def batch_loglik(self, xxx_todo_changeme, w_ns):
        (zeta_ns, v_ns) = xxx_todo_changeme
        return np.sum(scipy.stats.poisson(mu=zeta_ns*v_ns).logpmf(w_ns))
    
    def batch_sample(self, xxx_todo_changeme1):
        (zeta_ns, v_ns) = xxx_todo_changeme1
        try:
            return scipy.stats.poisson(mu=zeta_ns*v_ns).rvs()
        except:
            print('asdf')
            raise ValueError("Get some error")
#             pdb.set_trace()
#         return scipy.stats.poisson(mu=zeta_ns*v_ns).rvs()

##### prior over gamma_ls given L #####

class gamma_ls_given_L_dist(distributions.dist):
    """
    p(gamma_ls|L)
    """
    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        self.horse = distributions.gamma_dist(alpha, beta)

    @caching.cache_method_decorator(lambda inst, L, gamma_ls: hash(id(inst)) + hash(L) + monotonic_utils.array_hash(gamma_ls))
    def batch_loglik(self, L, gamma_ls):
        return scipy.stats.gamma(a=self.alpha*np.ones(L+1), scale=1./(self.beta*np.ones(L+1))).logpdf(gamma_ls)

    def batch_sample(self, L):
        return scipy.stats.gamma(a=self.alpha*np.ones(L+1), scale=1./(self.beta*np.ones(L+1))).rvs()

    def iterative_sample(self, sampled_gammas):
        return monotonic_utils.sample_truncated_gamma(self.alpha, self.beta, 1.)
    
    def get(self, L, i):
        return self.horse
    
##### prior over decision lists #####
            
class theta_dist(distributions.dist):
    """
    accepts theta, works for theta of any N.  constituent distributions take x_ns as argument in loglik and sample
    """
    def __init__(self, rule_f_ls_given_L_dist, gamma_ls_given_L_dist, zeta_ns_dist, L_dist):
        self.rule_f_ls_given_L_dist, self.gamma_ls_given_L_dist, self.zeta_ns_dist, self.L_dist = rule_f_ls_given_L_dist, gamma_ls_given_L_dist, zeta_ns_dist, L_dist
        self.w_ns_given_zeta_ns_given_v_ns_dist = w_ns_given_zeta_ns_given_v_ns_dist()

    @caching.cache_method_decorator(lambda inst, reduced_theta: hash(id(inst)) + hash(reduced_theta))
    def reduced_loglik(self, reduced_theta):
        log_p = 0.0
        log_p += self.L_dist.loglik(reduced_theta.L)
        try:
            assert len(reduced_theta.gamma_ls) == reduced_theta.L+1
        except:
            print(reduced_theta.gamma_ls, reduced_theta.rule_f_ls)
            raise ValueError("Get some error")
#             pdb.set_trace()
        log_p += np.sum(self.gamma_ls_given_L_dist.batch_loglik(reduced_theta.L, reduced_theta.gamma_ls))
        try:
            log_p += self.rule_f_ls_given_L_dist.loglik(reduced_theta.L, reduced_theta.rule_f_ls)
        except:
            print([r.long_repr() for r in reduced_theta.rule_f_ls])
            raise ValueError("Get some error")
#             pdb.set_trace()
        return log_p

    @property
    def possible_rules(self):
        return self.rule_f_ls_given_L_dist.possible_rules
    
    def loglik(self, x_ns, theta):
        """
        assumes N for theta (has N w_ns and zeta_ns) and N for x_ns are the same
        """
        log_p = self.reduced_loglik(theta)
        log_p += np.sum(self.zeta_ns_dist.batch_loglik(theta.zeta_ns))
        log_p += np.sum(self.w_ns_given_zeta_ns_given_v_ns_dist.batch_loglik((theta.zeta_ns, theta.get_v_ns(x_ns)), theta.w_ns))
        return log_p
    
    def sample(self, x_ns):
        N = len(x_ns)
        L = self.L_dist.sample() + 1
        rule_f_ls = self.rule_f_ls_given_L_dist.sample(L)
        gamma_ls = self.gamma_ls_given_L_dist.batch_sample(L)
        zeta_ns = self.zeta_ns_dist.batch_sample(N)
        v_ls = reduced_theta.v_ls_helper(gamma_ls)
        v_ns = v_ls[reduced_theta.z_ns_helper(rule_f_ls, x_ns)]
        w_ns = self.w_ns_given_zeta_ns_given_v_ns_dist.batch_sample((zeta_ns, v_ns))
        theta_sample = theta(rule_f_ls, gamma_ls, w_ns, zeta_ns)
        return theta_sample
    
class theta_dist_constructor(monotonic_utils.f_base):

    def __init__(self, rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist):
        self.rule_f_ls_given_L_dist_constructor, self.gamma_ls_given_L_dist, self.L_dist = rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist

    def __call__(self, x_ns, y_ns):
        """
        x_ns is to feed to rule miner.  x_ns does not determine N of theta
        """
        rule_f_ls_given_L_dist = self.rule_f_ls_given_L_dist_constructor(x_ns, y_ns)
        return theta_dist(rule_f_ls_given_L_dist, self.gamma_ls_given_L_dist, zeta_ns_dist(1.0), self.L_dist)

##### some priors over the rules #####

class agnostic_rule_f_ls_only_order_unknown_dist(distributions.dist):
    """
    all orderings have the same probability
    """
    def __init__(self, fixed_rule_f_ls):
        self.fixed_rule_f_ls = set(fixed_rule_f_ls)
    
    def loglik(self, L, rule_f_ls):
        assert np.all([rule_f_l in self.fixed_rule_f_ls for rule_f_l in rule_f_ls])
        assert L == len(self.fixed_rule_f_ls)
        return -np.log(float(len(self.fixed_rule_f_ls)))

class uniform_base_rule_dist(distributions.dist):

    def __init__(self, possible_rule_fs):
        self.possible_rule_fs = possible_rule_fs

    def loglik(self, rule_f):
        return -np.log(len(self.possible_rule_fs))

    def sample(self):
        return np.random.choice(self.possible_rule_fs)
    
class with_replacement_rule_f_ls_given_L_dist(distributions.dist):

    def __init__(self, base_rule_dist):
        self.base_rule_dist = base_rule_dist

    def loglik(self, L, rule_f_ls):
        return np.sum([self.base_rule_dist.loglik(rule_f_l) for rule_f_l in rule_f_ls])

    def sample(self, L):
        return reduce(lambda sampled, dummy: sampled + [self.iterative_sample(sampled)], [])

    def iterative_sample(self, sampled_rule_fs):
        return self.base_rule_dist.sample()

class fixed_set_rule_f_ls_given_L_dist(distributions.dist):

    def __init__(self, possible_rule_fs):
        self.possible_rule_fs = set(possible_rule_fs)
        self.probs = -1.*np.cumsum(np.log(np.arange(len(self.possible_rule_fs), 0, -1)))
        self.possible_rule_fs_list = list(possible_rule_fs)

    @property
    def possible_rules(self):
        return self.possible_rule_fs_list
                            
    def loglik(self, L, rule_f_ls):
        return self.probs[len(rule_f_ls)-1]

    def iterative_sample(self, sampled_rule_fs):
        # if len(sampled_rule_fs) == len(self.possible_rule_fs_list):
        #     pdb.set_trace()
        #     return rule.dummy_rule_f()
        asdf = set(sampled_rule_fs)
        for i in np.random.permutation(len(self.possible_rule_fs_list)):
            if self.possible_rule_fs_list[i] not in asdf:
                return self.possible_rule_fs_list[i]
        # print(len(self.possible_rule_fs_list))
        # print(len(sampled_rule_fs))
        # print(asdf)
        assert False
        return np.random.choice(tuple(self.possible_rule_fs - set(sampled_rule_fs)))

    def iterative_sample_loglik(self, sampled_rule_fs, iterative_sample):
        return -np.log(len(self.possible_rule_fs) - len(sampled_rule_fs))

    def sample(self, L):
        try:
            ans = list(np.random.choice(self.possible_rule_fs_list, L,replace=False))
            # print(ans)
            if len(ans) > 0:
                return ans
        except ValueError:
            print("error L")
            raise ValueError("Get some error")
#                 pdb.set_trace()
#                 return list(self.possible_rule_fs_list)
    
class fixed_set_rule_f_ls_given_L_dist_constructor(monotonic_utils.f_base):

    def __init__(self, rule_miner_f):
        self.rule_miner_f = rule_miner_f

    def __call__(self, x_data, y_data):
        rule_fs = self.rule_miner_f(x_data, y_data)
        return fixed_set_rule_f_ls_given_L_dist(rule_fs)
