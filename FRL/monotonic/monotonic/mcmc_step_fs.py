import monotonic.monotonic.mcmc as mcmc
from collections import namedtuple
#import random
import monotonic.monotonic.model as model
import copy
import scipy.stats
import numpy as np
import monotonic.monotonic.utils as monotonic_utils
import pdb
import itertools
import copy

debug_prob = -1
strong_debug = -1

##### mcmc_step_f constructors and requisite constructors #####

class generic_mcmc_step_f_constructor(monotonic_utils.f_base):

    def __init__(self, cls, obj_f_constructor, accept_proposal_f):
        self.cls, self.obj_f_constructor, self.accept_proposal_f = cls, obj_f_constructor, accept_proposal_f
    
    def __call__(self, theta_dist):
        return self.cls(theta_dist, self.obj_f_constructor(theta_dist), self.accept_proposal_f)

class reduced_posterior_obj_f_constructor(monotonic_utils.f_base):

    def __call__(self, theta_dist):
        return lambda x_ns, y_ns, theta: theta_dist.reduced_loglik(theta) + theta.reduced_batch_loglik(x_ns, y_ns)
        
class generic_gibbs_mcmc_step_f_constructor(monotonic_utils.f_base):

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, theta_dist):
        return self.cls(theta_dist)

class simulated_annealing_accept_proposal_f(monotonic_utils.f_base):

    def __init__(self, temperature_f):
        self.temperature_f = temperature_f
        self.step_count = 0

    def __call__(self, old_loglik, new_loglik):
        assert np.isfinite(old_loglik)
        assert np.isfinite(new_loglik)
        temperature = self.temperature_f(self.step_count)
        self.step_count += 1
        accept_prob = np.exp(-(old_loglik - new_loglik) / temperature)
        # print(accept_prob)
        return np.random.random() < accept_prob

class constant_temperature_f(monotonic_utils.f_base):

    def __init__(self, the_temperature):
        self.the_temperature = the_temperature

    def __call__(self, step_count):
        return self.the_temperature
    
##### gibbs steps #####

class gibbs_diff_f(mcmc.diff_f):
    """
    assumes variable can be refered to using __dict__
    """
    def __init__(self, param_idx, new_val):
        self.param_idx, self.new_val, = param_idx, new_val

    @property
    def accepted(self):
        return True

    def make_change(self, theta):
        """
        works regardless of whether param_idx is a string or a list of strings
        """
        try:
            theta[self.param_idx] = self.new_val
        except (KeyError, TypeError):
            for param_idx, new_val in zip(self.param_idx, self.new_val):
                theta[param_idx] = new_val
                
class gamma_ls_gibbs_step_f(mcmc.mcmc_step_f):

    def __init__(self, theta_dist):
        self.theta_dist = theta_dist
    
    def __call__(self, x_ns, y_ns, theta):

        assert theta.L+1 == len(theta.gamma_ls)
        
        def reverse_cumsum(v):
            return [y for y in reversed(np.cumsum([x for x in reversed(v)]))]

        old_gamma_ls = copy.deepcopy(theta.gamma_ls)

        for l in range(0,theta.L+1):
            c_l = np.exp(reverse_cumsum(np.log(theta.gamma_ls)) - np.log(theta.gamma_ls[l]))
            c_l[(l+1):] = 0
            gamma_l_dist = self.theta_dist.gamma_ls_given_L_dist.get(theta.L+1, l)
            gamma_l_dist_alpha, gamma_l_dist_beta = gamma_l_dist.alpha, gamma_l_dist.beta
            gamma_l_gibbs_dist_alpha = gamma_l_dist_alpha + (theta.get_z_ns(x_ns) <= l).dot(theta.w_ns)
            gamma_l_gibbs_dist_beta = gamma_l_dist_beta + c_l[theta.get_z_ns(x_ns)].dot(theta.zeta_ns)

            if l == theta.L:
                theta.gamma_ls[l] = monotonic_utils.sample_truncated_gamma(gamma_l_gibbs_dist_alpha, gamma_l_gibbs_dist_beta, 0.001)
            else:
                theta.gamma_ls[l] = monotonic_utils.sample_truncated_gamma(gamma_l_gibbs_dist_alpha, gamma_l_gibbs_dist_beta, 1.)
                
        new_gamma_ls = copy.deepcopy(theta.gamma_ls)
        theta.gamma_ls = old_gamma_ls
        return gibbs_diff_f('gamma_ls', new_gamma_ls)

class w_ns_gibbs_step_f(mcmc.mcmc_step_f):

    def __init__(self, theta_dist):
        self.theta_dist = theta_dist
        
    def __call__(self, x_ns, y_ns, theta):
        rates = theta.zeta_ns * theta.v_ns
        new_w_ns = monotonic_utils.vectorized_zero_truncated_poisson_sample(rates)
        new_w_ns[y_ns==0] = 0
        return gibbs_diff_f('w_ns', new_w_ns)

class zeta_ns_gibbs_step_f(mcmc.mcmc_step_f):
    
    def __init__(self, theta_dist):
        self.theta_dist = theta_dist

    def __call__(self, x_ns, y_ns, theta):
        return gibbs_diff_f('zeta_ns', scipy.stats.gamma(a=theta.w_ns + 1, scale=1.0 / (theta.get_v_ns(x_ns) + 1)).rvs())

class w_ns_zeta_ns_gibbs_step_f(mcmc.mcmc_step_f):

    def __init__(self, theta_dist):
        self.theta_dist = theta_dist

    def __call__(self, x_ns, y_ns, theta):
        w_ns = scipy.stats.geom(p=1.0 / (1 + theta.get_v_ns(x_ns))).rvs()
        w_ns[y_ns==0] = 0
        alpha_ns = 1.0 + w_ns
        beta_ns = 1.0 + theta.get_v_ns(x_ns)
        zeta_ns = scipy.stats.gamma(a=alpha_ns, scale=1./beta_ns).rvs()
        return gibbs_diff_f(['w_ns','zeta_ns'],[w_ns,zeta_ns])
    
# changing the rule_f list

class replace_rule_mh_diff_f(mcmc.diff_f):

    def __init__(self, replace_pos, replace_rule_f, new_gamma_ls, _accepted):
        self.replace_pos, self.replace_rule_f, self.new_gamma_ls, self._accepted = replace_pos, replace_rule_f, new_gamma_ls, _accepted
        
    @property
    def accepted(self):
        return self._accepted

    def make_change(self, theta):
        theta.rule_f_ls[self.replace_pos] = self.replace_rule_f
        theta.gamma_ls = self.new_gamma_ls
        
class replace_rule_mh_step_f(mcmc.mcmc_step_f):

    def __init__(self, theta_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_dist = theta_dist
        
    def __call__(self, x_ns, y_ns, theta, sampling=False):

        log_q_ratio = 0.

        old_loglik = self.obj_f(x_ns, y_ns, theta)
        
        def q_replace_pos():
            replace_pos = np.random.randint(0, theta.L)
            return replace_pos, 0

        replace_pos, replace_pos_log_q_ratio = q_replace_pos()
        log_q_ratio += replace_pos_log_q_ratio
        
        def q_replace_rule_f():
            replace_rule_f = self.theta_dist.rule_f_ls_given_L_dist.iterative_sample(theta.rule_f_ls)
            return replace_rule_f, 0

        replacement_rule_f, replace_rule_f_log_q_ratio = q_replace_rule_f()
        log_q_ratio += replace_rule_f_log_q_ratio
        
        def q_new_gamma_ls_optimize_all():
            replaced_rule_f = theta.rule_f_ls[replace_pos]
            theta.rule_f_ls[replace_pos] = replacement_rule_f
            new_gamma_ls = theta.get_greedy_optimal_gamma_ls(x_ns, y_ns)
            theta.rule_f_ls[replace_pos] = replaced_rule_f
            return new_gamma_ls, 0

        q_new_gamma_ls = q_new_gamma_ls_optimize_all

        new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
        log_q_ratio += new_gamma_ls_log_q_ratio

        # all the decisions have been made.  now make the actual changes
        replaced_rule_f = theta.rule_f_ls[replace_pos]
        theta.rule_f_ls[replace_pos] = replacement_rule_f
        old_gamma_ls = theta.gamma_ls
        theta.gamma_ls = new_gamma_ls
        
        # decide whether to change
        new_loglik = self.obj_f(x_ns, y_ns, theta)
        # print("replace")
        accept = self.accept_proposal_f(old_loglik, new_loglik)
        
        # undo the change, because this should not actual modify things
        theta.rule_f_ls[replace_pos] = replaced_rule_f
        theta.gamma_ls = old_gamma_ls
        
        return replace_rule_mh_diff_f(replace_pos, replacement_rule_f, new_gamma_ls, accept)
    
class rule_swap_only_mh_diff_f(mcmc.diff_f):

    def __init__(self, xxx_todo_changeme, xxx_todo_changeme1, new_gamma_ls, _accepted):
        (rule_f_a_idx, rule_f_a) = xxx_todo_changeme
        (rule_f_b_idx, rule_f_b) = xxx_todo_changeme1
        (self.rule_f_a_idx, self.rule_f_a), (self.rule_f_b_idx, self.rule_f_b), self.new_gamma_ls, self._accepted = (rule_f_a_idx, rule_f_a), (rule_f_b_idx, rule_f_b), new_gamma_ls, _accepted
         
    @property
    def accepted(self):
        return self._accepted
        
    def make_change(self, theta):
        theta.rule_f_ls = copy.copy(theta.rule_f_ls)
        theta.gamma_ls = self.new_gamma_ls
        monotonic_utils.swap_list_items(theta.rule_f_ls, self.rule_f_a_idx, self.rule_f_b_idx)
    
class rule_swap_only_mh_step_f(mcmc.mcmc_step_f):

    def __init__(self, theta_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_dist = theta_dist

    def __call__(self, x_ns, y_ns, theta, sampling=False):

        if theta.L == 1:
            return rule_swap_only_mh_diff_f((None, None), (None, None), None, False)
                    
        else:

            log_q_ratio = 0.

            old_loglik = self.obj_f(x_ns, y_ns, theta)
            
            # decide which rules to swap
            def q_swap_idx():
                idx_a, idx_b = np.random.choice(list(range(theta.L)), 2, replace=False)
                return (idx_a, idx_b), 0

            (idx_a, idx_b), swap_idx_log_q_ratio = q_swap_idx()
            log_q_ratio += swap_idx_log_q_ratio

            def q_new_gamma_ls_optimize_all():
                monotonic_utils.swap_list_items(theta.rule_f_ls, idx_a, idx_b)
                new_gamma_ls = theta.get_greedy_optimal_gamma_ls(x_ns, y_ns)
                monotonic_utils.swap_list_items(theta.rule_f_ls, idx_a, idx_b)
                return new_gamma_ls, 0

            q_new_gamma_ls = q_new_gamma_ls_optimize_all
            
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio

            # all the decisions have been made.  now make the actual changes              
            monotonic_utils.swap_list_items(theta.rule_f_ls, idx_a, idx_b)
            old_gamma_ls = theta.gamma_ls
            theta.gamma_ls = new_gamma_ls
            
            # decide whether to change
            new_loglik = self.obj_f(x_ns, y_ns, theta)
            # print("swap")
            accept = self.accept_proposal_f(old_loglik, new_loglik)

            # undo the change, because this should not actual modify things
            monotonic_utils.swap_list_items(theta.rule_f_ls, idx_a, idx_b)
            theta.gamma_ls = old_gamma_ls
            return rule_swap_only_mh_diff_f((idx_a, theta.rule_f_ls[idx_a]), (idx_b, theta.rule_f_ls[idx_b]), new_gamma_ls, accept)
                        
class add_or_remove_rule_mh_diff_f(mcmc.diff_f):

    def __init__(self, change_type, _accepted, pos, xxx_todo_changeme2 = (None, None)):
        (added_rule_f, new_gamma_ls) = xxx_todo_changeme2
        self.change_type, self._accepted, self.pos, (self.added_rule_f, self.new_gamma_ls) = change_type, _accepted, pos, (added_rule_f, new_gamma_ls)

    @property
    def accepted(self):
        return self._accepted
        
    def make_change(self, theta):
        if self.change_type == add_or_remove_rule_mh_step_f.ADD:
            theta.rule_f_ls.insert(self.pos, self.added_rule_f)
            theta.gamma_ls = self.new_gamma_ls
        elif self.change_type == add_or_remove_rule_mh_step_f.REMOVE:
            theta.rule_f_ls.pop(self.pos)
            theta.gamma_ls = self.new_gamma_ls
    
class add_or_remove_rule_mh_step_f(mcmc.mcmc_step_f):
    """
    can either:
    insert a rule before l-th rule (l = 0...L-1)
    remove l-th rule (l = 0...L-1)
    move l-th rule (l = 0...L-1) to before l'-th rule (l = 0...L)
    """
    ADD = 0
    REMOVE = 1

    def __init__(self, theta_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_dist = theta_dist
    
    def __call__(self, x_ns, y_ns, theta, sampling=False):

        # inserts/removals at end of list (aka position L) cause problems.  whether to allow inserts there
#        allow_end_changes = True
        allow_end_changes = False

        log_q_ratio = 0.

        old_loglik = self.obj_f(x_ns, y_ns, theta)
        
        # decide whether to add or remove node
        def q_add_or_remove():

            if theta.L == 1:
                return add_or_remove_rule_mh_step_f.ADD, 0.
            elif theta.L == len(self.theta_dist.rule_f_ls_given_L_dist.possible_rule_fs):
                return add_or_remove_rule_mh_step_f.REMOVE, 0.
            else:
                add_prob = 0.5
                if scipy.stats.uniform.rvs() < add_prob:
                    _log_q_ratio = np.log(add_prob) - np.log(1.-add_prob)
                    return add_or_remove_rule_mh_step_f.ADD, _log_q_ratio
                else:
                    _log_q_ratio = np.log(1.-add_prob) - np.log(add_prob)
                    return add_or_remove_rule_mh_step_f.REMOVE, _log_q_ratio

        add_or_remove, add_or_remove_log_q_ratio = q_add_or_remove()
        log_q_ratio += add_or_remove_log_q_ratio
                
        if add_or_remove == add_or_remove_rule_mh_step_f.ADD:

            # decide the insert position
            def q_insert_pos():
                if allow_end_changes:
                    insert_pos = np.random.randint(0, theta.L+1)
                    return insert_pos, 0
                else:
                    insert_pos = np.random.randint(0, theta.L)
                    return insert_pos, 0

            insert_pos, insert_pos_log_q_ratio = q_insert_pos()
            log_q_ratio += insert_pos_log_q_ratio
                
            # decide the rule to insert
            def q_insert_rule():
                # pretend the rule was already there, just was not being used.  thus not an actual change
                insert_rule = self.theta_dist.rule_f_ls_given_L_dist.iterative_sample(theta.rule_f_ls)
                return insert_rule, 0

            insert_rule_f, insert_rule_f_log_q_ratio = q_insert_rule()
            log_q_ratio += insert_rule_f_log_q_ratio
            
            # decide the new gamma_ls using simplest method: drawn a single gamma from prior, and set gamma_ls[insert_pos] to it
            def q_new_gamma_ls_simple():
                # pretend the gamma value was already there, just was not being used.  thus not an actual change
                new_gamma_ls = copy.copy(theta.gamma_ls)
                if allow_end_changes:
                    insert_gamma = self.theta_dist.gamma_ls_given_L_dist.iterative_sample(theta.gamma_ls)
                    #raise NotImplementedError
                else:
                    insert_gamma = self.theta_dist.gamma_ls_given_L_dist.iterative_sample(theta.gamma_ls)
                new_gamma_ls = np.insert(new_gamma_ls, insert_pos, insert_gamma)
                return new_gamma_ls, 0

            def q_new_gamma_ls_optimize_all():
                theta.rule_f_ls.insert(insert_pos, insert_rule_f)
                new_gamma_ls = theta.get_greedy_optimal_gamma_ls(x_ns, y_ns)
                theta.rule_f_ls.pop(insert_pos)
                return new_gamma_ls, 0

            def q_new_gamma_ls_hybrid():
                if np.random.random() < 0.5:
                    return q_new_gamma_ls_optimize_all()
                else:
                    return q_new_gamma_ls_simple()

            # specify which gamma_ls proposal to use, and decide on the proposal
            # if sampling == True:
            #     q_new_gamma_ls = q_new_gamma_ls_simple
            # else:
            #     q_new_gamma_ls = q_new_gamma_ls_optimize_all

            q_new_gamma_ls = q_new_gamma_ls_optimize_all
            
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio

                
            # all the decisions have been made.  now make the actual changes
            theta.rule_f_ls.insert(insert_pos, insert_rule_f)
            old_gamma_ls = theta.gamma_ls
            theta.gamma_ls = new_gamma_ls
            
            # decide whether to change
            new_loglik = self.obj_f(x_ns, y_ns, theta)
            # print("add")
            accept = self.accept_proposal_f(old_loglik, new_loglik)
                
            # undo the change, because this should not actual modify things
            theta.rule_f_ls.pop(insert_pos)
            theta.gamma_ls = old_gamma_ls

            return add_or_remove_rule_mh_diff_f(add_or_remove_rule_mh_step_f.ADD, accept, insert_pos, (insert_rule_f, new_gamma_ls))

        elif add_or_remove == add_or_remove_rule_mh_step_f.REMOVE:

            # decide remove position
            def q_remove_pos():
                if allow_end_changes:
                    remove_pos = np.random.randint(0, theta.L)
                    return remove_pos, 0.
                else:
                    if theta.L == 1:
                        assert False
                        return 0, 0.
                    else:
                        remove_pos = np.random.randint(0, theta.L-1)
                        return remove_pos, 0.

            remove_pos, remove_pos_log_q_ratio = q_remove_pos()
            log_q_ratio += remove_pos_log_q_ratio

            def q_new_gamma_ls_simple():
                new_gamma_ls = copy.copy(theta.gamma_ls)
                new_gamma_ls = np.delete(new_gamma_ls, remove_pos)
                return new_gamma_ls, 0.

            def q_new_gamma_ls_optimize_all():
                old_rule_f = theta.rule_f_ls[remove_pos]
                theta.rule_f_ls.pop(remove_pos)
                new_gamma_ls = theta.get_greedy_optimal_gamma_ls(x_ns, y_ns)
                theta.rule_f_ls.insert(remove_pos, old_rule_f)
                return new_gamma_ls, 0

            def q_new_gamma_hybrid():
                if np.random.random() < 0.5:
                    return q_new_gamma_ls_optimize_all()
                else:
                    return q_new_gamma_ls_simple()

            # if sampling == True:
            #     q_new_gamma_ls = q_new_gamma_ls_simple
            # else:
            #     q_new_gamma_ls = q_new_gamma_ls_optimize_all
            q_new_gamma_ls = q_new_gamma_ls_optimize_all
            
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio
                
            # all the decisions have been made.  now make the actual changes                
            removed_rule_f = theta.rule_f_ls[remove_pos]
            old_gamma_ls = theta.gamma_ls
            theta.rule_f_ls.pop(remove_pos)
            theta.gamma_ls = new_gamma_ls

            # decide whether to change
            new_loglik = self.obj_f(x_ns, y_ns, theta)
            # print("remove")
            accept = self.accept_proposal_f(old_loglik, new_loglik)

            # undo the change, because this should not actual modify things
            theta.rule_f_ls.insert(remove_pos, removed_rule_f)
            theta.gamma_ls = old_gamma_ls

            return add_or_remove_rule_mh_diff_f(add_or_remove_rule_mh_step_f.REMOVE, accept, remove_pos, (None, new_gamma_ls))
