import numpy as np
from collections.abc import Iterable

def is_compressed(M):
    return not isinstance(M, np.ndarray)

if hasattr(dict, "viewkeys"):
    dict_keys_func = dict.viewkeys
else:
    dict_keys_func = dict.keys

if hasattr(dict, "viewvalues"):
    dict_values_func = dict.viewvalues
else:
    dict_values_func = dict.values

if hasattr(dict, "viewitems"):
    dict_items_func = dict.viewitems
else:
    dict_items_func = dict.items

class nonzero_dict(dict):
    # This is works fine and is more elegant, but slower.
    # def __setitem__(self, idx, val):
    #     if val == 0:
    #         try:
    #             del self[idx]
    #         except KeyError:
    #             pass
    #     else:
    #         dict.__setitem__(self, idx, val)
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
        return np.fromiter(dict_keys_func(self), dtype=int, count=len(self))
#        return np.array(list(dict_keys_func(self)), dtype=int)
#        #return np.fromiter(dict_keys_func(self), dtype=int)
    def values(self):
        return np.fromiter(dict_values_func(self), dtype=int, count=len(self))
#        return np.array(list(dict_values_func(self)), dtype=int)
#        #return np.fromiter(dict_values_func(self), dtype=int)
    def sum(self):
        return np.fromiter(dict_values_func(self), dtype=int).sum()
    def dict_keys(self):
        return dict_keys_func(self)

class nonzero_key_value_list(object):
    def __init__(self):
        self.k = []
        self.v = []
    def __setitem__(self, idx, val):
        #print("xxx Inside __setitem__ %s %s" % (str(idx), str(val)))
        try:
            loc = self.k.index(idx)
        except ValueError:
            loc = -1

        if loc == -1:
            if val != 0:
                self.k.append(idx)
                self.v.append(val)
        else:
            if val != 0:
                self.k[loc] = idx
                self.v[loc] = val
            else:
                self.k = self.k[:loc] + self.k[loc+1:]
                self.v = self.v[:loc] + self.v[loc+1:]

    def __getitem__(self, idx):
        #print("xxx Inside __getitem__ %s" % (str(idx)))
        try:
            loc = self.k.index(idx)
            return self.v[loc]
        except ValueError:
            return 0
    def __contains__(self, idx):
        return idx in self.k
    def __delitem__(self, idx):
        try:
            loc = self.k.index(idx)
        except ValueError:
            raise KeyError()
        self.k = self.k[:loc] + self.k[loc+1:]
        self.v = self.v[:loc] + self.v[loc+1:]
    def items(self):
        return zip(self.k,self.v)
    def __iter__(self):
        raise NotImplementedError("__iter__ not implemented")
    def keys(self):
        return np.array(self.k, dtype=int)
    def values(self):
        return np.array(self.v, dtype=int)
    def copy(self):
        d = nonzero_key_value_list()
        d.k = self.k.copy()
        d.v = self.v.copy()
        return d

class nonzero_key_value_sorted_array(object):
    def __init__(self):
        self.k = np.array([], dtype=int)
        self.v = np.array([], dtype=int)
    def __setitem__(self, idx, val):
        # print("xxx Inside __setitem__ %s %s" % (str(idx), str(val)))
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            if val != 0:
                self.k = np.insert(self.k, loc, idx)
                self.v = np.insert(self.v, loc, val)
        else:
            if val != 0:
                self.k[loc] = idx
                self.v[loc] = val
            else:
                self.k = np.delete(self.k, loc)
                self.v = np.delete(self.v, loc)
    def __getitem__(self, idx):
        # print("xxx Inside __getitem__ %s" % (str(idx)))
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            return 0
        else:
            return self.v[loc]
    def __contains__(self, idx):
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            return False
        else:
            return True
    def __delitem__(self, idx):
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            raise KeyError()
        else:
            self.k = np.delete(self.k, loc)
            self.v = np.delete(self.v, loc)
    def items(self):
        return zip(self.k,self.v)
    def __iter__(self):
        raise NotImplementedError("__iter__ not implemented")
    def keys(self):
        return self.k
    def values(self):
        return self.v
    def copy(self):
        d = nonzero_key_value_sorted_array()
        d.k = self.k.copy()
        d.v = self.v.copy()
        return d
    def sum(self):
        return self.v.sum()

nonzero_data = nonzero_dict
#nonzero_data = nonzero_key_value_sorted_array
#nonzero_data = nonzero_key_value_list

star = slice(None, None, None)
class fast_sparse_array(object):
    def __init__(self, tup, base_type=list, dtype=None):
        # The dtype is not really used except as a hint for conversion into a dense array.
        self.dtype=dtype
        self.base_type = base_type
        if base_type is list:
            self.rows = [nonzero_data() for i in range(tup[0])]
            self.cols = [nonzero_data() for i in range(tup[1])]
        elif base_type is np.ndarray:
            self.rows = np.array([nonzero_data() for i in range(tup[0])])
            self.cols = np.array([nonzero_data() for i in range(tup[1])])
        elif base_type is dict:
            self.rows = base_type({i : nonzero_data() for i in range(tup[0])})
            self.cols = base_type({i : nonzero_data() for i in range(tup[1])})
        self.shape = tup
        self.debug = 0
        if self.debug:
            self.M_ver = np.zeros(self.shape, dtype=int)
        return
    def __getitem__(self, idx):
        # print("Enter __getitem__ %s" % (str(idx)))
        if 0: #self.debug:
            return self.M_ver.__getitem__(idx)

        i,j = idx

        if type(i) is slice and i == star:
            L = [(k,v) for (k,v) in dict_items_func(self.cols[j])]
        elif type(j) is slice and j == star:
            L = [(k,v) for (k,v) in dict_items_func(self.rows[i])]
        elif isinstance(i, Iterable):
            return np.array([self.cols[j][k] for k in i], dtype=int)
        elif isinstance(j, Iterable):
            return np.array([self.rows[i][k] for k in j], dtype=int)
        else:
            if j in self.rows[i]:
                L = self.rows[i][j]
            else:
                L = 0

        if self.debug:
            L0 = self.M_ver.__getitem__(idx)
            if isinstance(L, Iterable):
                nz = L0.nonzero()[0]
                L_i = np.array([k for (k,v) in L])
                L_v = np.array([v for (k,v) in L])
                s = np.argsort(L_i)
                L_i = L_i[s]
                L_v = L_v[s]
                assert(len(nz) == len(L))
                assert((nz == L_i).all())
                assert((L0[nz] == L_v).all())
            else:
                assert(L0 == L)

        return L
    def __setitem__(self, idx, val):
        #print("Inside __setitem__ %s %s" % (str(idx), str(val)))
        i,j = idx
        self.rows[i][j] = val
        self.cols[j][i] = val
        if self.debug:
            self.M_ver.__setitem__(idx, val)
            self.verify()
            self.verify_conistency()
    def set_axis_dict(self, idx, axis, d_new):
        # I am surprised this works.
        # There should be entries in the old row that are not in the new one.
        # But apparently:             assert(len(dict_keys_func(self.rows[idx]) - dict_keys_func(d_new)) == 0)
        # and:                        assert(len(dict_keys_func(self.cols[idx]) - dict_keys_func(d_new)) == 0)
        # This seems to be a consequence of moving nodes from one community to another and how this affects the edge counts.
        if axis == 0:
            for k,v in d_new.items():
                self.cols[k][idx] = v
            self.rows[idx] = d_new
        elif axis == 1:
            for k,v in d_new.items():
                self.rows[k][idx] = v
            self.cols[idx] = d_new
    def __str__(self):
        s = ""
        for i in range(self.shape[0]):
            s += str(self.rows[i]) + "\n"
        return s
    def count_nonzero(self):
        return sum(len(d) for d in self.rows)
    def verify(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                a = self.__getitem__((i,j))
                b = self.M_ver.__getitem__((i,j))
                if a != b:
                    raise Exception("Mismatch at element (%d %d) and values (%d %d)" % (i,j,a,b))
    def verify_conistency(self):
        for i in range(len(self.rows)):
            for k in self.rows[i].dict_keys():
                if i not in self.cols[k]:
                    print("fail i,k",i,k)
                assert(i in self.cols[k])
        for i in range(len(self.cols)):
            for k in self.cols[i].dict_keys():
                assert(i in self.rows[k])

    # There is a complicated issue. If the graph is streamed in parts there may be a node without edges. In that case we should return an empty array.
    # But we need a default dtype for a length zero array.
    # In any case there is a possible performance advantage to setting the dtype apriori.
    def take(self, idx, axis):
        if axis == 0:
            return (np.array(list(dict_keys_func(self.rows[idx])), dtype=int), np.array(list(dict_values_func(self.rows[idx])), dtype=int))
#            return (self.rows[idx].keys(),self.rows[idx].values())
        elif axis == 1:
            return (np.array(list(dict_keys_func(self.cols[idx])), dtype=int), np.array(list(dict_values_func(self.cols[idx])), dtype=int))
#            return (self.cols[idx].keys(),self.cols[idx].values())
        else:
            raise Exception("Invalid axis %s" % (axis))
    def take_dict(self, idx, axis):
        if axis == 0:
            return self.rows[idx]
        elif axis == 1:
            return self.cols[idx]
        else:
            raise Exception("Invalid axis %s" % (axis))
    def copy(self):
        c = fast_sparse_array((0,0))
        c.dtype = self.dtype
        c.shape = self.shape
        c.base_type = self.base_type
        if self.base_type is list:
            c.rows = [i.copy() for i in self.rows]
            c.cols = [i.copy() for i in self.cols]
        elif base_type is np.ndarray:
            c.rows = np.array([i.copy() for i in self.rows])
            c.cols = np.array([i.copy() for i in self.cols])
        else:
            raise Exception("Unknown base type")
        return c

def take_nonzero(A, idx, axis, sort):
    if type(A) is np.ndarray:
        a = np.take(A, idx, axis)
        idx = a.nonzero()[0]
        val = a[idx]
        return idx, val
    elif type(A) is fast_sparse_array:
        idx,val = A.take(idx, axis)
        if sort:
            s = np.argsort(idx)
            idx = idx[s]
            val = val[s]
        return idx,val
    else:
        raise Exception("Unknown array type for A (type %s) = %s" % (type(A), str(A)))
