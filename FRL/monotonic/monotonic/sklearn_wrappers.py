import monotonic.monotonic.utils as monotonic_utils
import numpy as np
import pandas as pd
from . import extra_utils as caching
import monotonic.monotonic.rule as rule
import monotonic.monotonic.model as model
import monotonic.monotonic.distributions as distributions
import monotonic.monotonic.mcmc_step_fs as mcmc_step_fs
import monotonic.monotonic.mcmc as mcmc
import monotonic.monotonic.predictors as predictors
import copy

class monotonic_sklearn_predictor(object):
    """
    this mimic sklearn classifiers, so X is a numpy array
    """
    def __init__(self, horse):
        self.horse = horse

    def decision_function(self, X):
        return np.array([self.horse(x) for x in X])

    def predict_proba(self, X):
        one_probs = self.decision_function(X)
        return np.array([1.0-one_probs,one_probs]).T

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    @property
    def train_info(self):
        return self.horse.train_info
        
class monotonic_sklearn_fitter_from_my_constructor(object):

    def __init__(self, classifier_constructor):
        self.classifier_constructor = classifier_constructor
        self.x_ns = None
        self.y_ns = None
        
    def fit(self, X, y, x_names=None,if_map=False,len_max=None):
        """
        can pass in names of each feature as x_names if desired
        """
        assert len(X) == len(y)
        self.x_ns = monotonic_utils.nparray_to_x_data(X, x_names)
        self.y_ns = monotonic_utils.nparray_to_y_data(y)
        self.predictor = monotonic_sklearn_predictor(self.classifier_constructor(self.x_ns, self.y_ns,if_map,len_max))
        return self.predictor,self.x_ns,self.y_ns,self.classifier_constructor

    # def sample(self,n_start,n_sample,len_max=None,truncate=False):
    #     theta_dist = self.predictor.horse.theta_dist
    #     theta = theta_dist.sample(self.x_ns)
    #     theta_use = copy.deepcopy(theta)
    #     gamma_ls_gibbs = mcmc_step_fs.gamma_ls_gibbs_step_f(theta_dist)
    #     w_ns_zeta_ns_gibbs = mcmc_step_fs.w_ns_zeta_ns_gibbs_step_f(theta_dist)
    #     mcmc_step_fs_gibbs = [mcmc_step_f_constructor(theta_dist) for mcmc_step_f_constructor in self.classifier_constructor.mcmc_step_f_constructors]

    #     theta_info = []
    #     for k in range(n_start+n_sample):
    #         gamma_ls_gibbs(self.x_ns, self.y_ns, theta).make_change(theta)
    #         for mcmc_step_f_gibbs in mcmc_step_fs_gibbs:
    #             diff_f = mcmc_step_f_gibbs(self.x_ns, self.y_ns, theta,True)
    #             diff_f(theta)
    #         w_ns_zeta_ns_gibbs(self.x_ns, self.y_ns, theta).make_change(theta)
    #         theta_new = copy.deepcopy(theta)
    #         if k>=n_start:
    #             if truncate:
    #                 while len(theta_new.rule_f_ls)>len_max:
    #                     theta_new.rule_f_ls.pop(len(theta_new.rule_f_ls)-1)
    #                 theta_new.gamma_ls = theta_new.get_greedy_optimal_gamma_ls(self.x_ns, self.y_ns)
    #             theta_info.append([copy.deepcopy(theta_new.gamma_ls),copy.deepcopy(theta_new.rule_f_ls)])
    #     return theta_use, theta_info


class monotonic_sklearn_fitter(monotonic_sklearn_fitter_from_my_constructor):

    def __init__(self, num_steps = 5000, min_supp = 5, max_clauses = 2, prior_length_mean = 8, prior_gamma_l_alpha = 1., prior_gamma_l_beta = 0.1, temperature = 1):
        rule_miner_f = rule.rule_miner_f(min_supp, max_clauses)
        rule_f_ls_given_L_dist_constructor = model.fixed_set_rule_f_ls_given_L_dist_constructor(rule_miner_f)
        L_dist = distributions.poisson_dist(prior_length_mean)
        gamma_ls_dist_alpha, gamma_ls_dist_beta = prior_gamma_l_alpha, prior_gamma_l_beta
        gamma_ls_given_L_dist = model.gamma_ls_given_L_dist(gamma_ls_dist_alpha, gamma_ls_dist_beta)
        accept_proposal_f = mcmc_step_fs.simulated_annealing_accept_proposal_f(mcmc_step_fs.constant_temperature_f(temperature))
        mcmc_step_f_constructors = [\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.rule_swap_only_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.add_or_remove_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.replace_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    ]
        theta_dist_constructor = model.theta_dist_constructor(rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist)
        my_predictor_constructor = predictors.map_predictor_constructor(num_steps, mcmc_step_f_constructors, theta_dist_constructor)
        self.classifier_constructor = my_predictor_constructor

class monotonic_nfoil_sklearn_fitter(monotonic_sklearn_fitter_from_my_constructor):

    def __init__(self, num_steps = 5000, prior_length_mean = 8, prior_gamma_l_alpha = 1., prior_gamma_l_beta = 0.1, temperature = 1):
        work_folder = '/Users/fultonw/Downloads/nfoil/datain'
        nfoil_path = '/Users/fultonw/Downloads/nfoil/bin/nfoil'
        rule_miner_f = rule.nfoil_rule_miner_f(work_folder, nfoil_path, 1)

        rule_f_ls_given_L_dist_constructor = model.fixed_set_rule_f_ls_given_L_dist_constructor(rule_miner_f)
        L_dist = distributions.poisson_dist(prior_length_mean)
        gamma_ls_dist_alpha, gamma_ls_dist_beta = prior_gamma_l_alpha, prior_gamma_l_beta
        gamma_ls_given_L_dist = model.gamma_ls_given_L_dist(gamma_ls_dist_alpha, gamma_ls_dist_beta)
        accept_proposal_f = mcmc_step_fs.simulated_annealing_accept_proposal_f(mcmc_step_fs.constant_temperature_f(temperature))
        mcmc_step_f_constructors = [\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.rule_swap_only_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.add_or_remove_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.replace_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    ]
        theta_dist_constructor = model.theta_dist_constructor(rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist)
        my_predictor_constructor = predictors.map_predictor_constructor(num_steps, mcmc_step_f_constructors, theta_dist_constructor)
        self.classifier_constructor = my_predictor_constructor
        
class greedy_sklearn_fitter(monotonic_sklearn_fitter_from_my_constructor):

    def __init__(self):
        work_folder = '/Users/fultonw/Downloads/nfoil/datain'
        nfoil_path = '/Users/fultonw/Downloads/nfoil/bin/nfoil'
        rule_miner_f = rule.nfoil_rule_miner_f(work_folder, nfoil_path, 1)
        self.classifier_constructor = predictors.greedy_constructor(rule_miner_f)
