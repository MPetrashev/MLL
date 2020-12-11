from inspect import signature
from utils.LazyDict import LazyDict


class lazy_property(object):
    """
    This decorator can be used for lazy evaluation of an object attribute.
    Property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self.is_static = isinstance(fget, staticmethod)
        self.is_class = isinstance(fget, classmethod)
        if self.is_class or self.is_static:
            fget = fget.__func__ # staticmethod

        self.func_name = fget.__name__
        self.fget = fget
        self.n_parameters = len(signature(self.fget).parameters) - (0 if self.is_static else 1)

    def __get__(self, obj, cls):
        if obj is None:
            obj = cls

        if self.n_parameters>1:
            if self.is_static:
                value = LazyDict(lambda key: self.fget(*key))
            else:
                value = LazyDict(lambda key: self.fget(obj, *key))
        elif self.n_parameters == 1:
            if self.is_static:
                value = LazyDict(lambda key: self.fget(key))
            else:
                value = LazyDict(lambda key: self.fget(obj, key))
            # value = LazyDict(lambda key: self.fget(obj, key) if not self.is_static else self.fget(key))
        else:
            value = self.fget(obj) if not self.is_static else self.fget()
        setattr(obj, self.func_name, value)

        return value
