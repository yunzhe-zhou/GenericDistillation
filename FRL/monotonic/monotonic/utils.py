import numpy as np
import scipy.stats
import copy
import monotonic.monotonic.extra_utils as caching
import pdb

class my_object(object):

    @classmethod
    def get_cls(cls):
        return cls
    
    def __getitem__(self, s):
        try:
            return self.__dict__[s]
        except KeyError:
            return self.get_cls().__dict__[s].__get__(self)

    def __setitem__(self, s, val):
        try:
            self.__dict__[s] = val
        except KeyError:
            cls.__dict__[s].__set__(self, val)

obj_base = my_object
f_base = my_object

import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


class x_data(object):

    def short_repr(self):
        return self.name

    @property
    def x_names(self):
        return self._x_names
        
    def __init__(self, xs, name=None, x_names=None):
        if x_names is None:
            self._x_names = np.array(list(map(str, list(range(len(next(iter(xs))))))))
        else:
            self._x_names = np.array(list(map(str, x_names)))
        self.name = name
        self.xs = xs

    def __iter__(self):
        return iter(self.xs)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        if isinstance(i,int):
            return self.xs[i]
        else:
            return x_data(self.xs[i], self.name, self.x_names)

    def __hash__(self):
        return hash(self.name)
        
def y_data(ys):
    return np.array(ys)
            
def raw_dataframe_to_x_data(d):
    """
    assume label is in last column of dataframe
    """
    xs = [tuple(row) for (row_name, row) in d.iloc[:,0:-1].iterrows()]
    try:
        name = d.name
    except AttributeError:
        name = str(id(d) % 10000)
    return x_data(xs, name, d.columns[0:-1])

def raw_dataframe_to_y_data(d):
    return np.array(d.iloc[:,-1])

def raw_dataframe_to_xy_data(d):
    return raw_dataframe_to_x_data(d), raw_dataframe_to_y_data(d)

def nparray_to_x_data(X, x_names=None):
    x_names = list(map(str, list(range(X.shape[1])))) if x_names is None else x_names
    xs = [x for x in X]
    name = str(id(X) % 10000)
    return x_data(xs, name, x_names)

def nparray_to_y_data(y):
    return y_data(y)

def get_simulated_x_data(N):
    xs = [np.array([i]) for i in range(N)]
    name = str(id(xs) % 10000)
    return x_data(xs, name)

class constant_f(f_base):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val
        
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(x):
    try:
        return np.log(x) - np.log(1-x)
    except:
        import pdb
        pdb.set_trace()

def reverse_np_array(ar):
    return np.array([x for x in reversed(ar)])

def vectorized_zero_truncated_poisson_sample(rates):
    ts = -np.log(1.0 - (np.random.uniform(size=len(rates)) * (1.0 - np.exp(-1.0*rates))))
    new_rates = rates - ts
    return 1.0 + scipy.stats.poisson(mu=new_rates).rvs()

def sample_truncated_gamma(alpha, beta, left_truncate):
    d = scipy.stats.gamma(a=alpha, scale=1./beta)
    omitted = d.cdf(left_truncate)
    u = np.random.sample()    
    ans = d.ppf(u * (1.-omitted) + omitted)
    # print("alpha",alpha)
    # print("beta",beta)
    # print(ans)
    eps = .01
    if not np.isfinite(ans):
        # import pdb
        # pdb.set_trace()
        return left_truncate + eps
    else:
        return ans

def sample_zero_truncated_negative_binomial(success_probs, num_successes):
    """
    returns number of failures required to get num_successes
    """
    d = scipy.stats.nbinom(n=num_successes,p=success_probs)
    assert len(success_probs) == len(num_successes)
    zero_probs = d.pmf(np.zeros(len(success_probs)))
    slicers = np.random.uniform(low=zero_probs, high=np.ones(len(success_probs)))
    return d.ppf(slicers)
    
def swap_list_items(l, idx_a, idx_b):
    try:
        a_handle = l[idx_a]
        b_handle = l[idx_b]
        l[idx_b] = a_handle
        l[idx_a] = b_handle
    except:
        print(idx_a, idx_b, len(l))

def levenshtein(s1, s2):

    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)
 
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1 # than s2
            substitutions = previous_row[j] + (c1!=c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]

def array_hash(ar):
    return hash(np.sum([hash(el) for el in ar]))

def rule_list_hash(rule_list):
    return hash(tuple([rule.idx for rule in rule_list]))
