import monotonic.monotonic.utils as monotonic_utils
import scipy.stats
import numpy as np

class dist(monotonic_utils.my_object):

    def loglik(self, x):
        pass

    def sample(self):
        pass
            
class gamma_dist(dist):

    def __init__(self, alpha, beta, left_truncate = 0):
        self.alpha, self.beta = alpha, beta
        self.horse = scipy.stats.gamma(a=alpha, scale=1./beta)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpdf(x)

class poisson_dist(dist):

    def __init__(self, rate, left_truncate = 0):
        self.rate = rate
        self.horse = scipy.stats.poisson(mu=rate)

    def sample(self):
        # return 8
        return 2
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpmf(x)

class constant_dist(dist):

    def __init__(self, val):
        self.val = val

    def sample(self, *args):
        return self.val
        
    def batch_sample(self, *args):
        return self.val
        
    def loglik(self, *args):
        return 0
    
class exp_dist(dist):

    def __init__(self, rate):
        self.rate = rate
        self.horse = scipy.stats.expon(scale = 1./rate)

    def loglik(self, x):
        return self.horse.logpdf(x)

    def sample(self):
        return self.horse.rvs()
