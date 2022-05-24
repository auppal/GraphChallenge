#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define errExit(msg)    do { perror(msg); exit(EXIT_FAILURE);	\
  } while (0)

/* An array of hash tables. There are n_cols tables, each with an allocated size of n_rows. */
struct hash {
  uint64_t *keys;
  uint64_t *vals;
  size_t w; /* width of each hash table */
  size_t n; /* the number of tables */
};

struct compressed_array {
  char *shm_path;
  size_t buf_size;
  void *buf;
  uint64_t *magic;
  struct hash rows;
  struct hash cols;
};



/* From khash */
#define kh_int64_hash_func(key) (khint32_t)((key)>>33^(key)^(key)<<11)
#define hash(x) (((x) >> 33 ^ (x) ^ (x) << 11))
#define EMPTY_FLAG (1UL << 63)

void *shared_memory_get(const char *shm_path, size_t buf_size)
{
#if 0
  /* The shm_open approach is for if we ever want to have processes besides 
   * self and children share the memory region.
   */

  /* Create shared memory object and set its size to the size
     of our structure. */

  int fd = shm_open(shm_path, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);

  if (fd == -1) {
    perror("shm_open");
    return NULL;
  }

  if (ftruncate(fd, sizeof(struct compressed_array)) == -1) {
   perror("ftruncate");
   return NULL;
  }

  /* Map the object into the caller's address space. */

  void *rc = mmap(NULL, buf_size, PROT_READ | PROT_WRITE,
			     MAP_SHARED, fd, 0);

  if (rc == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }
#else
  void *rc = mmap(NULL, buf_size, PROT_READ | PROT_WRITE,
			     MAP_ANONYMOUS | MAP_SHARED, -1, 0);

  if (rc == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }  
#endif
  return rc;
}

struct compressed_array *compressed_copy(const struct compressed_array *y)
{
  struct compressed_array *x = malloc(sizeof(struct compressed_array));

  /* XXX Actually using shm_path would give a shared memory region and would not be valid for a new copy */
  memcpy(x, y, sizeof(struct compressed_array));
  x->shm_path = NULL;

  x->buf = shared_memory_get(x->shm_path, x->buf_size);

  if (!x->buf) {
    free(x);
    return NULL;
  }

  memcpy(x->buf, y->buf, x->buf_size);
  return x;
}

struct compressed_array *compressed_array_create(size_t n_nodes, size_t max_degree)
{
  struct compressed_array *x = malloc(sizeof(struct compressed_array));
  
  if (x == NULL) {
    perror("malloc");
    return NULL;
  }

  char *shm_path = strdup("compressed_array");

  if (!shm_path) {
    perror("strdup");
    free(x);
    return NULL;
  }

  max_degree = n_nodes;
  size_t table_width = max_degree * 1.43;
  size_t table_size = n_nodes * table_width;
  size_t buf_size = 4 * table_size * sizeof(uint64_t);

  x->shm_path = shm_path;
  x->buf_size = buf_size;
  x->buf = shared_memory_get(shm_path, buf_size);

  if (!x->buf) {
    free(x);
    return NULL;
  }

  x->magic = x->buf;
  *x->magic = 0x123456;

  x->rows.keys = (uint64_t *) x->buf + 0 * table_size;
  x->rows.vals = (uint64_t *) x->buf + 1 * table_size;
  x->rows.w = table_width;
  x->rows.n = n_nodes;
  
  x->cols.keys = (uint64_t *) x->buf + 2 * table_size;
  x->cols.vals = (uint64_t *) x->buf + 3 * table_size;
  x->cols.w = table_width;
  x->cols.n = n_nodes;

  memset(x->buf, 0xff, buf_size);

  return x;
}

inline int compressed_get_single(struct compressed_array *x, uint64_t i, uint64_t j, uint64_t *val)
{
#if 0
  *val = *(x->rows.keys + (i * x->rows.n) + j);
  return 0;
#else
  uint64_t *base, *loc;
  long k, offset, L;
  uint64_t hj = hash(j);  

  /* get from row[i][j] */
  L = x->rows.w;
  offset = hj % L;
  base = (x->rows.keys + i * L);

  for (k=0; k<L; k++) {
    long idx = (k + offset) % L;
    loc = base + idx;

    if (*loc & EMPTY_FLAG) {
      break;
    }
    else if (*loc == j) {
      *val = *(x->rows.vals + i * L + idx);
      return 0;
    }
  }
#endif
  return -1;
}

int compressed_set_single(struct compressed_array *x, uint64_t i, uint64_t j, uint64_t val)
{
#if 0
  *(x->rows.keys + (i * x->rows.n) + j) = val;
#else
  uint64_t *base, *loc;
  long k, offset, L;
  uint64_t hi = hash(i);
  uint64_t hj = hash(j);

  //fprintf(stderr, "Set item %ld %ld = %ld\n", i, j, val);
  
  /* set row[i][j] = val */
  L = x->rows.w;
  offset = hj % L;
  base = (x->rows.keys + i * L);

  for (k=0; k<L; k++) {
    long idx = (k + offset) % L;
    loc = base + idx;

    if (*loc == j || *loc & EMPTY_FLAG) {
      *loc = j;
      *loc &= ~EMPTY_FLAG;
      *(x->rows.vals + i * L + idx) = val;
      //fprintf(stderr, "Setting at row %ld idx %ld to %ld\n", i, idx, val);
      break;
    }
  }

  if (k == L) {
    return -1;
  }

  /* set col[j][i] = val */
  L = x->cols.w;
  offset = hi % L;
  base = (x->cols.keys + j * L);

  for (k=0; k<L; k++) {
    long idx = (k + offset) % L;    
    loc = base + idx;

    if (*loc == i || *loc & EMPTY_FLAG) {
      *loc = i;
      *loc &= ~EMPTY_FLAG;
      *(x->cols.vals + j * L + idx) = val;
      break;
    }
  }

  if (k == L) {
    return -1;
  }

  /* set col[j][i] = val */
  return 0;
#endif
}

int compressed_take(struct compressed_array *x, long idx, long axis, uint64_t *keys, uint64_t *vals, long *p_cnt)
{
  uint64_t *p, *q;
  long k, L;
  long cnt = 0;

  //  fprintf(stderr, "compressed take idx %ld axis %ld\n", idx, axis);
  
  if (axis == 0) {
      L = x->rows.w;    
      p = (x->rows.keys + idx * L);
      q = (x->rows.vals + idx * L);

  }
  else { /* axis == 0 */
      L = x->cols.w;
      p = (x->cols.keys + idx * L);
      q = (x->cols.vals + idx * L);      
  }

  for (k=0; k<L; k++) {
    if ((*p & EMPTY_FLAG) == 0) {
      *keys++ = *p;
      *vals++ = *q;
      cnt++;
    }
    p++;
    q++;
  }

  *p_cnt = cnt;
  return 0;
}


void compressed_array_destroy(struct compressed_array *p)
{
  if (p) {
    munmap(p->buf, p->buf_size);
    fprintf(stderr, "unlink path");
#if 0
    shm_unlink(p->shm_path);
#endif
    free(p);
  }
}

/* Python interface functions */
static void destroy(PyObject *obj)
{
  void *x = PyCapsule_GetPointer(obj, "compressed_array");
  compressed_array_destroy(x);
}


static PyObject* create(PyObject *self, PyObject *args)
{
  PyObject *ret;
  long n_nodes, max_degree;

  if (!PyArg_ParseTuple(args, "ll", &n_nodes, &max_degree)) {
    return NULL;
  }

  struct compressed_array *x = compressed_array_create(n_nodes, max_degree);

  fprintf(stderr, "Created tables for n_nodes %ld with degree %ld = %p\n", n_nodes, max_degree, x);

  ret = PyCapsule_New(x, "compressed_array", destroy);
  return ret;
  
}

static PyObject* copy(PyObject *self, PyObject *args)
{
  PyObject *obj;  
  PyObject *ret;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  struct compressed_array *y = PyCapsule_GetPointer(obj, "compressed_array");
  struct compressed_array *x = compressed_copy(y);

  ret = PyCapsule_New(x, "compressed_array", destroy);
  return ret;
}

static PyObject* setitem(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long i, j, val;

  if (!PyArg_ParseTuple(args, "Olll", &obj, &i, &j, &val))
    return NULL;
  
  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");
  compressed_set_single(x, i, j, val);
  Py_RETURN_NONE;
}

static PyObject* getitem(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long i, j;
  unsigned long val;

  if (!PyArg_ParseTuple(args, "Oll", &obj, &i, &j))
    return NULL;
  
  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  if (compressed_get_single(x, i, j, &val) < 0) {
    val = 0;
  }

  PyObject *ret = Py_BuildValue("k", val);
  return ret;
}

static PyObject* take_multi(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_i, *obj_j, *py_arr;

  if (!PyArg_ParseTuple(args, "OOO", &obj, &obj_i, &obj_j)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  int64_t i, j;
  long axis = 0;
  
  i = PyLong_AsLongLong(obj_i);

  if (i == -1) {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */

    j = PyLong_AsLongLong(obj_j);

    if (j == -1) {
      return NULL;
    }
    py_arr = PyArray_FROM_OTF(obj_i, NPY_LONG, NPY_IN_ARRAY);
    axis = 0;
  }
  else {
    py_arr = PyArray_FROM_OTF(obj_j, NPY_LONG, NPY_IN_ARRAY);
    axis = 1;
  }
  
  const long *arr = (const long *) PyArray_DATA(py_arr);
  long k, N = (long) PyArray_DIM(py_arr, 0);
  uint64_t *vals = malloc(N * sizeof(uint64_t));

  memset(vals, 0xff, N * sizeof(uint64_t));
  
  if (axis == 0) {
    for (k=0; k<N; k++) {
      if (compressed_get_single(x, arr[k], j, &vals[k]) < 0) {
	vals[k] = 0;
      }

      //fprintf(stderr, "C get %ld %ld = %ld\n", arr[k], j, *vals);      
    }
  }
  else {
    for (k=0; k<N; k++) {
      if (compressed_get_single(x, i, arr[k], &vals[k]) < 0) {
	vals[k] = 0;
      }
    }
  }

#if 0
  for(k=0;k<N;k++) {
    fprintf(stderr, "vals %ld = %ld\n", k, vals[k]);
  }
#endif
  
  npy_intp dims[] = {N};
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  
  
  PyObject *ret = Py_BuildValue("O", vals_obj);
  return ret;
}


static PyObject* take(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long i, axis;

  if (!PyArg_ParseTuple(args, "Oll", &obj, &i, &axis))
    return NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

#if 1
  long cnt, cnt_max = x->rows.w;

  uint64_t *keys = malloc(cnt_max * sizeof(uint64_t));
  uint64_t *vals = malloc(cnt_max * sizeof(uint64_t));

  compressed_take(x, i, axis, keys, vals, &cnt);  

  npy_intp dims[] = {cnt};

  PyObject *keys_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, keys);
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);

  PyArray_ENABLEFLAGS((PyArrayObject*) keys_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  
  
#if 0
  for (i=0; i<x->rows.w; i++) {
    keys[i] = i;
  }
#endif
#else  
  int N = 22;
  uint64_t *kk = malloc(N * sizeof(uint64_t));
  uint64_t *vv = malloc(N * sizeof(uint64_t));

  for (i=0; i<N; i++) {
    kk[i] = i;
    vv[i] = i;
  }

  npy_intp dims[] = {N};
  PyObject *keys_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, kk);
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vv);
  PyArray_ENABLEFLAGS((PyArrayObject*) keys_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  
  
#endif
  PyObject *ret = Py_BuildValue("OO", keys_obj, vals_obj);
  return ret;
}


static PyObject* get_magic(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long val = 0;

  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;
  
  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");
  val = *x->magic;
  PyObject *ret = Py_BuildValue("l", val);
  return ret;
}

static PyObject* set_magic(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long val = 0;

  if (!PyArg_ParseTuple(args, "Ol", &obj, &val))
    return NULL;
  
  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");
  fprintf(stderr, "Set magic to 0x%lx\n", val);
  *x->magic = val;

  Py_RETURN_NONE;
}

static PyMethodDef compressed_array_methods[] =
  {
   {
    "create",
    create,
    METH_VARARGS,
    "Create a new object."
   },
   {
    "copy",
    copy,
    METH_VARARGS,
    "Copy an existing object."
   },   
   {
    "get_magic",
    get_magic,
    METH_VARARGS,
    "Get magic value."
   },
   {
    "setitem",
    setitem,
    METH_VARARGS,
    "Set an item."
   },
   {
    "getitem",
    getitem,
    METH_VARARGS,
    "Get an item."
   },
   {
    "take",
    take,
    METH_VARARGS,
    "Take items along an axis."
   },
   {
    "take_multi",
    take_multi,
    METH_VARARGS,
    "Take several items along an axis."
   },
   {
    "set_magic",
    set_magic,
    METH_VARARGS,
    "Set magic value."
   },      
   {NULL, NULL, 0, NULL}
  };

static struct PyModuleDef compressed_array_definition =
  {
   PyModuleDef_HEAD_INIT,
   "compressed_array",
   "A Python module that supports compressed array operations.",
   -1,
   compressed_array_methods
  };

PyMODINIT_FUNC PyInit_compressed_array(void)
{
  Py_Initialize();
  import_array();
  return PyModule_Create(&compressed_array_definition);
}

