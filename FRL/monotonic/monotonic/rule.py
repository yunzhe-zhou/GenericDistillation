import pdb
import numpy as np
import fim
import monotonic.monotonic.utils as monotonic_utils
import pandas as pd
import monotonic.monotonic.extra_utils as caching
import itertools
import string
import os

class rule_f(monotonic_utils.f_base):
    """
    just a boolean fxn on x
    """
    def __init__(self, _short_repr, idx, r, support = None, x_names = None):
        self._short_repr, self.r, self.support, self.x_names = _short_repr, r, support, x_names
        self.cardinality = len(r)
        self.idx = idx

    def long_repr(self):
        return str(self.x_names)

    def short_repr(self):
        return self._short_repr

    def __hash__(self):
        return hash(self.short_repr())
    
    def __call__(self, x):
        for r_i in self.r:
            if not x[r_i]:
                return False
        return True

    @caching.hash_cache_method_decorator
    def batch_call(self, x_data):
#        print 'okok', np.array([self(x) for x in x_data])
        return np.array([self(x) for x in x_data])

    def __repr__(self):
        return self.long_repr()
    
    def get_support_and_pos_props(self, x_ns, y_ns):

        this_y_ns = np.array([y_n for (x_n, y_n) in zip(x_ns, y_ns) if self(x_n)])
        this_support = len(this_y_ns)
        this_pos_prop = np.sum(this_y_ns) / float(this_support)
        return this_support, this_pos_prop

class dummy_rule_f(monotonic_utils.f_base):

    def __init__(self, short_repr, idx):
        self.short_repr, self.idx = short_repr, idx
    
    def long_repr(self):
        return 'dummy'

    def short_repr(self):
        return 'dummy'
    
    def __hash__(self):
        return hash(0)

    def __call__(self, x):
        return False

    def batch_call(self, x_data):
        return np.array([False for x in x_data])

    def get_support_and_pos_props(self, x_ns, y_ns):
        return 0, 0.

    @property
    def support(self):
        return None
    
#@caching.default_write_fxn_decorator
class rule_miner_f(monotonic_utils.f_base):
    """
    supp is the proportion of data that needs to satisfy the rule
    zmin is the cardinality
    """
    def __init__(self, supp, zmax):
        self.supp, self.zmax = supp, zmax
    
    def __call__(self, x_data, y_data):
        """
        assumes x_data has x_names attribute
        """
        def which_are_1(v):
            return list(pd.Series(list(range(len(v))))[list(map(bool,v))])

        length = float(len(x_data))
        raw = fim.fpgrowth([which_are_1(x) for x in x_data], supp = self.supp, zmax = self.zmax)
        try:
            return [rule_f(str((x_data.short_repr(),i)), i, r, s[0]/length, list(x_data.x_names[list(r)])) for (i, (r, s)) in enumerate(raw)]
        except:
            return [rule_f(str((x_data.short_repr(),i)), i, r, s/length, list(x_data.x_names[list(r)])) for (i, (r, s)) in enumerate(raw)]
        
class hard_coded_rule_f(monotonic_utils.f_base):
    """
    returns True on hardcoded set of data.  truth should be boolean vector
    """
    def __init__(self, _short_repr, idx, truth):
        if np.random.uniform() < 0.5:
            truth = np.zeros(len(truth), dtype=bool)
        self._short_repr, self.truth = _short_repr, truth
        self.support = np.sum(self.truth) / len(self.truth)
        self.idx = idx

    def short_repr(self):
        return self._short_repr

    def long_repr(self):
        return 'nothing'
        
    def __hash__(self):
        return hash(self.short_repr())

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __call__(self, x):
        return self.truth[x[0]]

    @caching.hash_cache_method_decorator
    def batch_call(self, x_data):
        return self.truth[np.array([x[0] for x in x_data])]
    
class get_hard_coded_rule_f(monotonic_utils.f_base):

    def __init__(self, p_true, num_rules):
        self.p_true, self.num_rules = p_true, num_rules

    def __call__(self, x_data):
        return [hard_coded_rule_f(str((x_data.short_repr(),j)), j, np.array([np.random.random() < self.p_true for i in range(len(x_data))])) for j in range(self.num_rules)]

def get_nfoil_input_string_with_negation(x_ns, y_ns):
    """
    assumes x_data is binary vectors
    """

    def x_name_to_f_name(x_name):
        return 'f%s' % x_name

    def x_name_to_t_name(x_name):
        return 't%s' % x_name

    def i_to_i_name(i):
        return 'e%d' % i
            
    import io
    output = io.StringIO()

    output.write(\
"""\
classes(pos,neg).
type(class(k)).
"""\
)

    for x_name in x_ns.x_names:
        output.write(\
"""\
type(%s(k,%s)).
""" % (x_name_to_f_name(x_name),x_name_to_t_name(x_name)))

    for x_name in x_ns.x_names:
        output.write(\
"""\
rmode(%s(+,0)).
rmode(%s(+,1)).
"""\
% (x_name_to_f_name(x_name),x_name_to_f_name(x_name)))
        
    for (i, (x_n, y_n)) in enumerate(zip(x_ns, y_ns)):
        output.write(\
"""\
%s(%s).
"""\
% ('pos' if y_n==0 else 'neg', i_to_i_name(i)))
        for (x_name, x_ni) in zip(x_ns.x_names, x_n):
            output.write(\
"""\
%s(%s,%d).
"""\
% (x_name_to_f_name(x_name), i_to_i_name(i), 1 if x_ni==1 else 0))
    output.flush()
    return output.getvalue()


    
class nfoil_rule_miner_f(monotonic_utils.f_base):

    def __init__(self, working_folder, nfoil_path, depth=1, max_clause=2):
        self.working_folder, self.nfoil_path, self.depth, self.max_clause = working_folder, nfoil_path, depth, max_clause
    
    def __call__(self, x_ns, y_ns):

        identifier = 'temp'
        input_file = '%s/%s_input' % (self.working_folder, identifier)
        output_file = '%s/%s_output' % (self.working_folder, identifier)

        import monotonic.monotonic.nfoil_helpers as nfoil_helpers
        
        rule_feature_indicies = nfoil_helpers.mine_nfoil_rules(x_ns, y_ns, x_ns.x_names, input_file, output_file, self.nfoil_path, self.max_clause, np.random.randint(0,9999), self.depth)

        mined_rules = []
        
        for (idx, r) in enumerate(rule_feature_indicies):
            mined_rules.append(rule_f(str((x_ns.name,idx)), idx, r, None, [x_ns.x_names[i] for i in r]))
        
        dummy_rules = [dummy_rule_f(str((x_ns.name,i,'dummy')),len(mined_rules)+i) for i in range(10)]

        return mined_rules + dummy_rules
