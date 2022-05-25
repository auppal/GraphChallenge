import numpy as np
import compressed_array
from collections import Iterable
from collections import defaultdict

def is_compressed(M):
    return not isinstance(M, np.ndarray)

class nonzero_dict(dict):
    def __getitem__(self, idx):
        try:
            return dict.__getitem__(self, idx)
        except KeyError:
            return 0
        except TypeError:
            # idx is likely iterable, the trick is to default to 0 for each element that is not in the dict
            L = [(k in self and dict.__getitem__(self,k)) or 0 for k in idx]
            return np.array(L)
    def copy(self):
        d = nonzero_dict(self)
        return d
    def keys(self):
        return np.fromiter(dict.keys(self), dtype=int, count=len(self))
    def values(self):
        return np.fromiter(dict.values(self), dtype=int, count=len(self))
    def sum(self):
        return np.fromiter(dict.values(self), dtype=int).sum()

star = slice(None, None, None)
class fast_sparse_array(object):
    def __init__(self, tup, width, base_type=list, dtype=None):
        if (tup[0] > 0):
            self.x = compressed_array.create(tup[0], width)
        self.width = width
        self.shape = tup
    def __getitem__(self, idx):
        return compressed_array.getitem(self.x, idx[0], idx[1])
    def __setitem__(self, idx, val):
        compressed_array.setitem(self.x, idx[0], idx[1], val)
    def set_axis_dict(self, idx, axis, d_new):
        compressed_array.setaxis(self.x, idx, axis, d_new.keys(), d_new.values())
    def set_axis_dict_old(self, idx, axis, d_new):
        if axis == 0:
            for k,v in d_new.items():
                compressed_array.setitem(self.x, idx, k, v)
        else:
            for k,v in d_new.items():
                compressed_array.setitem(self.x, k, idx, v)
            
        # I am surprised this works.
        # There should be entries in the old row that are not in the new one.
        # But apparently:             assert(len(dict_keys_func(self.rows[idx]) - dict_keys_func(d_new)) == 0)
        # and:                        assert(len(dict_keys_func(self.cols[idx]) - dict_keys_func(d_new)) == 0)
        # This seems to be a consequence of moving nodes from one community to another and how this affects the edge counts.
    def __str__(self):
        s = ""
        for i in range(self.shape[0]):
            s += str(self.rows[i]) + "\n"
        return s
    def count_nonzero(self):
        return sum(len(d) for d in self.rows)
    # There is a complicated issue. If the graph is streamed in parts there may be a node without edges. In that case we should return an empty array.
    # But we need a default dtype for a length zero array.
    # In any case there is a possible performance advantage to setting the dtype apriori.
    def take(self, idx, axis):
        keys,vals = compressed_array.take(self.x, idx, axis)
        return (keys,vals)
    def take_dict(self, idx, axis):
        k,v = compressed_array.take(self.x, idx, axis)
        return nonzero_dict(zip(k,v))
    def copy(self):
        c = fast_sparse_array((0,0), width=self.width)
        c.x = compressed_array.copy(self.x)
        c.shape = self.shape
        return c

def is_sorted(x):
    return len(x) == 1 or (x[1:] >= x[0:-1]).all()

def take_nonzero(A, idx, axis, sort):
    assert(A is not None)
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
