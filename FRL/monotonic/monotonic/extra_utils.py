import pickle
import hashlib
import functools
import types

which_hash_f = 'sha256'

def get_hash(obj):
    try:
        pickle_s = pickle.dumps(obj)
    except TypeError as e:
        print(e)
        pdb.set_trace()
    m = hashlib.new(which_hash_f)
    m.update(pickle_s)
    return m.hexdigest()

def hash_get_key(identifier, *args, **kwargs):
#    return hash(sum(map(hash, args + kwargs.values())))

    try:
        return hash(hash(identifier)+sum(map(hash, args)))
    except:
        import pdb
        pdb.set_trace()

def cache(f, key_f, identifier, d, *args, **kwargs):
#    return f(*args, **kwargs)
    if len(d) > 100:
        d.clear()
    key = key_f(identifier, *args, **kwargs)
    try:
        return d[key]
    except KeyError:
        ans = f(*args, **kwargs)
        d[key] = ans
        return ans

class decorated_method(object):
    
    def __init__(self, f, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, inst, *args, **kwargs):
        """
        this is the new class method.  inst is the instance on which the method is called
        """
        raise NotImplementedError

    def __get__(self, inst, cls):
        return functools.partial(self.__call__, inst)

class method_decorator(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, f):
        return decorated_method(f)
    
class cache_decorated_method(decorated_method):

    def __init__(self, f, key_f, d):
        self.f, self.key_f, self.d = f, key_f, d

    def __call__(self, inst, *args, **kwargs):
        return cache(functools.partial(self.f, inst), self.key_f, inst, self.d, *args, **kwargs)

class cache_method_decorator(method_decorator):

    def __init__(self, key_f):
        self.key_f = key_f
        self.d = {}

    def __call__(self, f):
        return cache_decorated_method(f, self.key_f, self.d)

class fxn_decorator(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, f):
        
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            raise NotImplementedError

        return wrapped_f

def get_callable_name(f):
    if isinstance(f, functools.partial):
        return get_callable_name(f.func)
    elif isinstance(f, types.FunctionType):
        return f.__name__
    else:
        try:
            return f.__class__.__name
        except:
            return repr(f)
        
class cache_fxn_decorator(fxn_decorator):

    def __init__(self, key_f):
        self.key_f = key_f
        self.d = {}

    def __call__(self, f):
        
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return cache(f, self.key_f, get_callable_name(f), self.d, *args, **kwargs)

        return wrapped_f
    
hash_cache_method_decorator = cache_method_decorator(hash_get_key)
