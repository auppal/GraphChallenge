import numpy as np
import compressed_array
from collections import Iterable
from collections import defaultdict
import os

def is_compressed(M):
    return not isinstance(M, np.ndarray)

star = slice(None, None, None)
class fast_sparse_array(object):
    def __init__(self, tup, width, base_type=list, dtype=None):
        if (tup[0] > 0):
            self.x = compressed_array.create(tup[0], width)
        self.width = width
        self.shape = tup
        self.impl = "compressed_native"
    def __getitem__(self, idx):
        return compressed_array.getitem(self.x, idx[0], idx[1])
    def __setitem__(self, idx, val):
        compressed_array.setitem(self.x, idx[0], idx[1], val)
    def set_axis_dict_old(self, idx, axis, d_new):
        try:
            compressed_array.setaxis(self.x, idx, axis, d_new.keys(), d_new.values())
        except AttributeError:
            keys,values = compressed_array.keys_values_dict(d_new)
            compressed_array.setaxis(self.x, idx, axis, keys, values)
    def set_axis_dict(self, idx, axis, d_new):
        compressed_array.setaxis_from_dict(self.x, idx, axis, d_new)
    def set_axis_dict_old(self, idx, axis, d_new):
        keys,values = compressed_array.keys_values_dict(d_new)
        compressed_array.setaxis(self.x, idx, axis, keys, values)
        # I am surprised this works.
        # There should be entries in the old row that are not in the new one.
        # But apparently:             assert(len(dict_keys_func(self.rows[idx]) - dict_keys_func(d_new)) == 0)
        # and:                        assert(len(dict_keys_func(self.cols[idx]) - dict_keys_func(d_new)) == 0)
        # This seems to be a consequence of moving nodes from one community to another and how this affects the edge counts.
    def __str__(self):
        return "Not implemented"
    def count_nonzero(self):
        return sum(len(d) for d in self.rows)
    # There is a complicated issue. If the graph is streamed in parts there may be a node without edges. In that case we should return an empty array.
    # But we need a default dtype for a length zero array.
    # In any case there is a possible performance advantage to setting the dtype apriori.
    def take(self, idx, axis):
        return compressed_array.take(self.x, idx, axis)
    def copy(self):
        c = fast_sparse_array((0,0), width=self.width)
        c.x = compressed_array.copy(self.x)
        c.shape = self.shape
        return c

def is_sorted(x):
    return len(x) == 1 or (x[1:] >= x[0:-1]).all()

def take_nonzero(A, idx, axis, sort):
    if not is_compressed(A):
        a = np.take(A, idx, axis)
        idx = a.nonzero()[0]
        val = a[idx]
        return idx, val
    else:
        idx,val = A.take(idx, axis)
        if sort:
            s = np.argsort(idx)
            idx = idx[s]
            val = val[s]
        return idx,val

def nonzero_slice(A, sort=True):
    if not is_compressed(A):
        idx = A.nonzero()[0]
        val = A[idx]
    else:
        idx = np.array([k for (k,v) in A], dtype=int)
        val = np.array([v for (k,v) in A], dtype=int)
        if sort:
            s = np.argsort(idx)
            idx = idx[s]
            val = val[s]
    return idx,val
