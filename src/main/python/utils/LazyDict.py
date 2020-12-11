from collections.abc import Mapping

#todo don't forget about functools.wraps
# @see https://stackoverflow.com/questions/16669367/setup-dictionary-lazily
from typing import Callable


class LazyDict(Mapping):
    """
    Basic implementation of Lazy dictionary where on 1st access to the value the factory method would be called with a
    passed key to construct the underlying value.
    """
    def __init__(self, factory: Callable):
        self.func = factory
        self._raw_dict = {}

    def __getitem__(self, key):
        if key not in self._raw_dict:
            self._raw_dict[key] = self.func(key)
        return self._raw_dict[key]

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)
