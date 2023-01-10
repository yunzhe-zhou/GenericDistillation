import monotonic.monotonic.mcmc as mcmc
import monotonic.monotonic.utils as monotonic_utils
import numpy as np
import monotonic.monotonic.model as model
import itertools
import pandas as pd
import monotonic.monotonic.extra_utils as caching
import pdb

class classifier(monotonic_utils.f_base):

    def __call__(self, x):
        raise NotImplementedError

# class single_decision_list_predictor(classifier):

#     def __init__(self, the_theta, train_x_ns, train_y_ns, train_rules):

#         self.the_theta, self.train_x_ns, self.train_y_ns = the_theta, train_x_ns, train_y_ns
#         self._train_info = the_theta.informative_df(train_x_ns, train_y_ns)
#         self._train_info.loc[self._train_info.shape[0]-1, 'rule'] = len(train_rules)
#         self.train_rules = train_rules
        
#     def __call__(self, x):
#         return self.the_theta.p_ls[self.the_theta.get_z(x)]

#     @property
#     def train_info(self):
#         return self._train_info
        
# class map_predictor_constructor(monotonic_utils.f_base):
#     """
#     uses reduced model to calculate theta with the highest map probability
#     """
#     def __init__(self, n_steps, mcmc_step_f_constructors, theta_dist_constructor):
#         self.n_steps, self.mcmc_step_f_constructors, self.theta_dist_constructor = n_steps, mcmc_step_f_constructors, theta_dist_constructor

#     def __call__(self, train_x_ns, train_y_ns):
#         theta_dist = self.theta_dist_constructor(train_x_ns, train_y_ns)
#         mcmc_step_fs = [mcmc_step_f_constructor(theta_dist) for mcmc_step_f_constructor in self.mcmc_step_f_constructors]
#         thetas = mcmc.get_thetas(train_x_ns, train_y_ns, self.n_steps, mcmc_step_fs, theta_dist)
#         posterior_logprobs = [theta_dist.reduced_loglik(theta) + theta.reduced_batch_loglik(train_x_ns, train_y_ns) for theta in thetas]
#         best_theta, best_posterior_logprobs = max(zip(thetas, posterior_logprobs), key = lambda theta_posterior_logprob: theta_posterior_logprob[1])
#         return single_decision_list_predictor(best_theta, train_x_ns, train_y_ns, theta_dist.possible_rules)

class single_decision_list_predictor(classifier):

    def __init__(self, the_theta, train_x_ns, train_y_ns, train_rules,theta_dist):

        self.the_theta, self.train_x_ns, self.train_y_ns = the_theta, train_x_ns, train_y_ns
        # self._train_info = the_theta.informative_df(train_x_ns, train_y_ns)
        # self._train_info.loc[self._train_info.shape[0]-1, 'rule'] = len(train_rules)
        self.train_rules = train_rules
        self.theta_dist = theta_dist
        
    def __call__(self, x):
        return self.the_theta.p_ls[self.the_theta.get_z(x)]

    @property
    def train_info(self):
        return self._train_info
        
class map_predictor_constructor(monotonic_utils.f_base):
    """
    uses reduced model to calculate theta with the highest map probability
    """
    def __init__(self, n_steps, mcmc_step_f_constructors, theta_dist_constructor):
        self.n_steps, self.mcmc_step_f_constructors, self.theta_dist_constructor = n_steps, mcmc_step_f_constructors, theta_dist_constructor

    def __call__(self, train_x_ns, train_y_ns,if_map,len_max=None):
        theta_dist = self.theta_dist_constructor(train_x_ns, train_y_ns)
        mcmc_step_fs = [mcmc_step_f_constructor(theta_dist) for mcmc_step_f_constructor in self.mcmc_step_f_constructors]
        if if_map == True:
            thetas = mcmc.get_thetas(train_x_ns, train_y_ns, self.n_steps, mcmc_step_fs, theta_dist,len_max)
            posterior_logprobs = [theta_dist.reduced_loglik(theta) + theta.reduced_batch_loglik(train_x_ns, train_y_ns) for theta in thetas]
            best_theta, best_posterior_logprobs = max(zip(thetas, posterior_logprobs), key = lambda theta_posterior_logprob: theta_posterior_logprob[1])
        else:
            best_theta = None
        return single_decision_list_predictor(best_theta, train_x_ns, train_y_ns, theta_dist.possible_rules,theta_dist)

class greedy_constructor(monotonic_utils.f_base):
    """
    mines rules, sorts by support, places into 
    """
    def __init__(self, rule_miner_f):
        self.rule_miner_f = rule_miner_f

    def __call__(self, train_x_ns, train_y_ns):
        rule_fs = self.rule_miner_f(train_x_ns, train_y_ns)

        supports, pos_props = list(zip(*[rule_f.get_support_and_pos_props(train_x_ns,train_y_ns) for rule_f in rule_fs]))

        sorted_rule_fs, sorted_pos_props = list(map(list,list(zip(*sorted(zip(rule_fs, pos_props), key = lambda rule_f_pos_prop:rule_f_pos_prop[1], reverse=True)))))

        data_ps = model.barebones_theta.get_data_p_ls_helper(sorted_rule_fs, train_x_ns, train_y_ns)
        return single_decision_list_predictor(model.barebones_theta(sorted_rule_fs, data_ps), train_x_ns, train_y_ns, rule_fs)
