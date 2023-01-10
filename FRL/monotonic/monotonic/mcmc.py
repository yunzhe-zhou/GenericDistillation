import numpy as np
import monotonic.monotonic.utils as monotonic_utils
import copy
import monotonic.monotonic.model as model
import copy
class diff_f(monotonic_utils.f_base):
    """
    represents the change to old theta proposed by a mcmc step
    """
    @property
    def accepted(self):
        pass

    def make_change(self, theta):
        """
        changes theta in place
        """
        pass

    def __call__(self, theta):

        if self.accepted:
            self.make_change(theta)
         
class mcmc_step_f(monotonic_utils.f_base):

    def __call__(self, x_ns, y_ns, theta):
        """
        should not modify theta, at least in the end
        """
        pass

#@monotonic_utils.do_cprofile
def get_diff_fs(x_ns, y_ns, mcmc_step_fs, n_steps, start_theta,len_max):
    diff_fs = []
    theta = start_theta
    for i in range(n_steps):
        for mcmc_step_f in mcmc_step_fs:
            diff_f = mcmc_step_f(x_ns, y_ns, theta)
            theta_copy = copy.deepcopy(theta)
            diff_f(theta)
            if len(theta.rule_f_ls)>len_max:
                theta = theta_copy
            else:
                diff_fs.append(diff_f)
    return diff_fs

def get_thetas_from_diff_fs(diff_fs, start_theta, params_to_copy):
    thetas = [copy.deepcopy(start_theta)]
    cur_theta = copy.deepcopy(start_theta)
    for diff_f in diff_fs:
        diff_f(cur_theta)
        new_theta = copy.copy(cur_theta)
        for param in params_to_copy:
            new_theta[param] = copy.deepcopy(cur_theta[param])
        thetas.append(new_theta)
    return thetas

class get_traces_f(monotonic_utils.f_base):

    def __init__(self, n_steps, mcmc_step_f_constructors, theta_dist_constructor):
        self.n_steps, self.mcmc_step_f_constructors, self.theta_dist_constructor = n_steps, mcmc_step_f_constructors, theta_dist_constructor
    
    def __call__(self, x_ns, y_ns):
        theta_dist = self.theta_dist_constructor(x_ns)
        mcmc_step_fs = [mcmc_step_f_constructor(theta_dist) for mcmc_step_f_constructor in self.mcmc_step_f_constructors(theta_dist)]
        thetas = mcmc.get_thetas(x_ns, y_ns, self.n_steps, mcmc_step_fs, theta_dist)
        return thetas
        
def get_thetas(x_ns, y_ns, n_steps, mcmc_step_fs, theta_dist,len_max):
    assert len(x_ns) == len(y_ns)
    try:
        start_theta = theta_dist.sample(x_ns)
    except:
        print("get_theta_error")
        raise ValueError("Get some error")
    # print(start_theta.__dict__)
    diff_fs = get_diff_fs(x_ns, y_ns, mcmc_step_fs, n_steps, copy.deepcopy(start_theta),len_max)
    thetas = get_thetas_from_diff_fs(diff_fs, start_theta, ['rule_f_ls', 'gamma_ls'])
    return thetas
