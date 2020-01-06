class lazy_property():
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__ if hasattr(fget, '__name__') else fget.__func__.__name__

    def __get__(self, obj, cls):
        if obj is None:
            value = self.fget.__func__()
            setattr(cls, self.func_name, value)
        else:
            value = self.fget(obj)
            setattr(obj, self.func_name, value)

        return value
