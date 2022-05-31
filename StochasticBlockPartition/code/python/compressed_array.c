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


/* From khash */
#define kh_int64_hash_func(key) (khint32_t)((key)>>33^(key)^(key)<<11)
#define hash(x) (((x) >> 33 ^ (x) ^ (x) << 11))
#define EMPTY_FLAG (1UL << 63)

/* An array of hash tables. There are n_cols tables, each with an allocated size of n_rows. */
struct hash {
  uint64_t *keys;
  int64_t *vals;
  size_t cnt;
  size_t limit;
  size_t width; /* width of each hash table */
};

struct hash *hash_create(size_t initial_size)
{
  size_t buf_size = sizeof(struct hash) + 2 * initial_size * sizeof(uint64_t);

  char *buf = malloc(buf_size);
  if (!buf) {
    return NULL;
  }

  struct hash *h = (struct hash *) buf;
  h->width = initial_size;
  h->cnt = 0;
  h->limit = h->width * 0.70;
  h->keys = (uint64_t *) (buf + sizeof(struct hash) + 0 * h->width * sizeof(uint64_t));
  h->vals = (int64_t *) (buf + sizeof(struct hash) + 1 * h->width * sizeof(int64_t));
  memset(h->keys, 0xff, h->width * sizeof(uint64_t));

  return h;
}

void hash_destroy(struct hash *h)
{
  if (h) {
    free(h);
  }
}

struct hash *hash_copy(const struct hash *y)
{
  size_t buf_size = sizeof(struct hash) + 2 * y->width * sizeof(uint64_t);

  char *buf = malloc(buf_size);
  if (!buf) {
    return NULL;
  }

  struct hash *h = (struct hash *) buf;
  h->width = y->width;
  h->cnt = y->cnt;
  h->limit = y->limit;
  h->keys = (uint64_t *) (buf + sizeof(struct hash) + 0 * h->width * sizeof(uint64_t));
  h->vals = (int64_t *) (buf + sizeof(struct hash) + 1 * h->width * sizeof(int64_t));
  
  memcpy(h->keys, y->keys, y->width * sizeof(uint64_t));
  memcpy(h->vals, y->vals, y->width * sizeof(int64_t));

  return h;
}

void hash_print(struct hash *h);
struct hash *hash_insert_single(struct hash *h, uint64_t k, int64_t v);
#define RESIZE_DEBUG (0)

inline struct hash *hash_resize(struct hash *h)
{
  size_t i;
  struct hash *h2;

  if (h->cnt == h->limit) {
    /* Resize */

#if RESIZE_DEBUG
    fprintf(stderr, "Before resize cnt is %ld\n", h->cnt);
    hash_print(h);
#endif

    h2 = hash_create(h->width * 2);
    if (!h2) {
      return NULL;
    }
#if RESIZE_DEBUG
    fprintf(stderr, "insert: ");
#endif

    long ins = 0;
    for (i=0; i<h->width; i++) {
      if ((h->keys[i] & EMPTY_FLAG) == 0) {
	//fprintf(stderr, " %ld ", h->keys[i]);	
	hash_insert_single(h2, h->keys[i], h->vals[i]);
	ins++;
      }
    }
#if RESIZE_DEBUG
    fprintf(stderr, "\nAfter resize inserted %ld\n", ins);
    hash_print(h2);
    fprintf(stderr, "\n\n");
#endif        

    if (h->cnt != h2->cnt) {
      fprintf(stderr, "Mismatch found abort\n");
      abort();
    }

    hash_destroy(h);
    h = h2;
  }

  return h;
}


struct hash *hash_insert_single(struct hash *h, uint64_t k, int64_t v)
{
  /* To avoid a subtle logic bug, first check for existance. 
   * Beacuse not every insertion will cause an increase in cnt.
   */
  size_t i, width = h->width;
  uint64_t kh = hash(k);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
    if (h->keys[idx] == k) {
      h->vals[idx] = v;
      break;      
    }
    else if (h->keys[idx] & EMPTY_FLAG) {
      h->keys[idx] = k & ~EMPTY_FLAG;
      h->vals[idx] = v;
      h->cnt++;
      break;
    }
  }

  h = hash_resize(h);
  return h;
}

inline int hash_search(const struct hash *h, uint64_t k, int64_t *v)
{
  size_t i, width = h->width;
  uint64_t kh = hash(k);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
    if (h->keys[idx] == k) {
      *v = h->vals[idx];
      return 0;
    }
    else if (h->keys[idx] & EMPTY_FLAG) {
      *v = 0; /* Default value */
      return -1;
    }
  }

  return -1;
}

inline void hash_search_multi(const struct hash *h, const uint64_t *keys, int64_t *vals, size_t n)
{
  size_t i;

  for (i=0; i<n; i++) {
    hash_search(h, keys[i], &vals[i]);
  }
}

inline int64_t hash_sum(const struct hash *h)
{
  size_t i;
  int64_t s = 0;

  for (i=0; i<h->width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      s += h->vals[i];
    }
  }

  return s;
}


inline size_t hash_keys(const struct hash *h, uint64_t *keys, size_t max_cnt)
{
  size_t i, width = h->width, cnt = 0;

  for (i=0; i<width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      if (cnt == max_cnt) {
	break;
      }
      *keys++ = h->keys[i];
      cnt++;
    }
  }

  return cnt;
}

inline size_t hash_vals(const struct hash *h, int64_t *vals, size_t max_cnt)
{
  size_t i, width = h->width, cnt = 0;

  for (i=0; i<width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      if (cnt == max_cnt) {
	break;
      }
      *vals++ = h->vals[i];
      cnt++;
    }
  }

  return cnt;
}

void hash_print(struct hash *h)
{
  size_t i, width = h->width;

  fprintf(stderr, "Print dict %p with %ld items\n", h, h->cnt);
  fprintf(stderr, "{ ");
  for (i=0; i<width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      fprintf(stderr, "%ld:%ld ", h->keys[i], h->vals[i]);      
    }
  }
  fprintf(stderr, "}\n");
}

inline void hash_accum_constant(const struct hash *h, size_t C)
{
  size_t i, width = h->width;

  for (i=0; i<width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      h->vals[i] += C;
    }
  }
}

inline struct hash *hash_accum_multi(struct hash *h, const uint64_t *keys, const int64_t *vals, size_t n_keys)
{
  size_t j, i;;

#if 0  
  fprintf(stderr, "Before accum_multi\n");
  hash_print(h);
#endif

  for (j=0; j<n_keys; j++) {
    uint64_t kh = hash(keys[j]);
#if 0
    fprintf(stderr, " Insert %ld +%ld\n", keys[j], vals[j]);
#endif
    for (i=0; i<h->width; i++) {
      size_t idx = (kh + i) % h->width;
      if (h->keys[idx] == keys[j]) {
	h->vals[idx] += vals[j];
	break;
      }
      else if (h->keys[idx] & EMPTY_FLAG) {
	/* Not found assume the previous default value of zero and set a new entry. */
	h->keys[idx] = keys[j];
	h->vals[idx] = vals[j];
	h->cnt++;
	break;
      }
    }

    h = hash_resize(h); /* Be careful to not re-use h->width. */
  }

#if 0  
  int flag = 0;
  if (h->cnt == h->limit) {
    fprintf(stderr, "Resize on accum before %ld %ld\n", h->cnt, h->limit);
    flag = 1;
  }
#endif  

#if 0
  if (flag) {
    fprintf(stderr, "Resize on accum after %ld %ld\n", h->cnt, h->limit);
  }
#endif
  
  return h;
}


struct compressed_array {
  size_t n_row, n_col;  
  struct hash **rows;
  struct hash **cols;
};


struct compressed_array *compressed_array_create(size_t n_nodes, size_t initial_width)
{
  struct compressed_array *x = malloc(sizeof(struct compressed_array));
  
  if (x == NULL) {
    perror("malloc");
    return NULL;
  }

  x->n_row = n_nodes;
  x->n_col = n_nodes;

  size_t i;
  x->rows = calloc(x->n_row, sizeof(struct hash *));

  if (!x->rows) {
    free(x);
    return NULL;
  }

  x->cols = calloc(x->n_col, sizeof(struct hash *));

  if (!x->cols) {
    free(x->rows);
    free(x);
    return NULL;
  }
  
  for (i=0; i<n_nodes; i++) {
    x->rows[i] = hash_create(initial_width * 2);
    x->cols[i] = hash_create(initial_width * 2);
    if (!x->rows[i] || !x->cols[i])
      break;
  }

  if (i != n_nodes) {
    for (; i>=0; i--) {
      hash_destroy(x->rows[i]);
      hash_destroy(x->cols[i]);
    }
    free(x->rows);
    free(x->cols);
    free(x);
    return NULL;
  }

  return x;
}

void compressed_array_destroy(struct compressed_array *x)
{
  size_t i;

  //fprintf(stderr, "Destroy %p\n", x);

  for (i=0; i<x->n_col; i++) {
    //fprintf(stderr, "Destroy hash %p\n", x->cols[i]);
    hash_destroy(x->cols[i]);
  }

  for (i=0; i<x->n_row; i++) {
    hash_destroy(x->rows[i]);
  }

  free(x->rows);
  free(x->cols);
  free(x);
}

struct compressed_array *compressed_copy(const struct compressed_array *y)
{
  struct compressed_array *x = malloc(sizeof(struct compressed_array));
  
  if (x == NULL) {
    perror("malloc");
    return NULL;
  }

  x->n_row = y->n_row;
  x->n_col = y->n_col;

  size_t i;
  x->rows = calloc(x->n_row, sizeof(struct hash *));

  if (!x->rows) {
    free(x);
    return NULL;
  }

  x->cols = calloc(x->n_col, sizeof(struct hash *));

  if (!x->cols) {
    free(x->rows);
    free(x);
    return NULL;
  }
  
  for (i=0; i<x->n_row; i++) {
    x->rows[i] = hash_copy(y->rows[i]);
    x->cols[i] = hash_copy(y->cols[i]);
    if (!x->rows[i] || !x->cols[i])
      break;
  }

  if (i != x->n_row) {
    for (; i>=0; i--) {
      hash_destroy(x->rows[i]);
      hash_destroy(x->cols[i]);
    }
    free(x->rows);
    free(x->cols);
    free(x);
    return NULL;
  }

  return x;
}

inline int compressed_get_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t *val)
{
  /* Just get from row[i][j] */
  return hash_search(x->rows[i], j, val);
}

inline void compressed_set_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t val)
{
  /* XXX There is a bug in this logic if any one fails. */
  x->rows[i] = hash_insert_single(x->rows[i], j, val);
  x->cols[j] = hash_insert_single(x->cols[j], i, val);
}

/* Take values along a particular axis */
int compressed_take_keys_values(struct compressed_array *x, long idx, long axis, uint64_t **p_keys, int64_t **p_vals, long *p_cnt)
{
  size_t cnt;
  struct hash *h = (axis == 0 ? x->rows[idx] : x->cols[idx]);

  cnt = h->cnt;

  uint64_t *keys;
  int64_t *vals;

  if (cnt == 0) {
    *p_keys = NULL;
    *p_vals = NULL;
    *p_cnt = 0;
    return 0;
  }
  else {
    keys = malloc(cnt * sizeof(uint64_t));
    vals = malloc(cnt * sizeof(int64_t));
  }

  hash_keys(h, keys, cnt);
  hash_vals(h, vals, cnt);

  *p_keys = keys;
  *p_vals = vals;
  *p_cnt = cnt;
  return 0;
}

inline struct hash *compressed_take(struct compressed_array *x, long idx, long axis)
{
  return (axis == 0 ? x->rows[idx] : x->cols[idx]);
}


#if 0
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
#endif


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
  ret = PyCapsule_New(x, "compressed_array", destroy);
  return ret;
}

inline struct hash **create_dict(struct hash *p)
{
  struct hash **ph = malloc(sizeof(struct hash **));
  if (ph) {
    *ph = p;
  }
  return ph;
}

static void destroy_dict(PyObject *obj)
{
  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");
  if (ph) {
    hash_destroy(*ph);
    free(ph);
  }
}

static PyObject* print_dict(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O", &obj))
      return NULL;

    struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");
    hash_print(*ph);
    Py_RETURN_NONE;
}

static PyObject* keys_values_dict(PyObject *self, PyObject *args)
{
  PyObject *obj;

  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }
  struct hash *h = *ph;


  size_t cnt = h->cnt;
  uint64_t *keys = malloc(cnt * sizeof(uint64_t));
  int64_t *vals = malloc(cnt * sizeof(int64_t));

  hash_keys(h, keys, cnt);
  hash_vals(h, vals, cnt);
#if 0
  fprintf(stderr, "Return %ld items\n", cnt);
  int i;
  for (i=0; i<cnt; i++) {
    fprintf(stderr, "keys %d=%ld\n", i, keys[i]);
  }
#endif  
  npy_intp dims[] = {cnt};
  PyObject *keys_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, keys);
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);

  PyArray_ENABLEFLAGS((PyArrayObject*) keys_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  

  PyObject *ret = Py_BuildValue("NN", keys_obj, vals_obj);
  return ret;
}


static PyObject* accum_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_k, *obj_v;

  if (!PyArg_ParseTuple(args, "OOO", &obj, &obj_k, &obj_v))
    return NULL;

  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }
  struct hash *h = *ph;  

  obj_k = PyArray_FROM_OTF(obj_k, NPY_LONG, NPY_IN_ARRAY);
  obj_v = PyArray_FROM_OTF(obj_v, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *keys = (const uint64_t *) PyArray_DATA(obj_k);
  const long *vals = (const long *) PyArray_DATA(obj_v);
  long N = (long) PyArray_DIM(obj_k, 0);

  h = hash_accum_multi(h, keys, vals, N);

  Py_DECREF(obj_k);
  Py_DECREF(obj_v);

  *ph = h;
  Py_RETURN_NONE;
}

static PyObject* empty_dict(PyObject *self, PyObject *args)
{
  PyObject *ret;
  long N;

  if (!PyArg_ParseTuple(args, "l", &N))
    return NULL;

  struct hash *ent = hash_create(N);
  struct hash **ph = create_dict(ent);
  
  ret = PyCapsule_New(ph, "compressed_array_dict", destroy_dict);
  return ret;
}

static PyObject* take_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, *ret;
  long idx, axis;

  if (!PyArg_ParseTuple(args, "Oll", &obj, &idx, &axis))
    return NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  /* Returns a copy, not a reference. */
#if 0
  struct hash *ent = compressed_take(x, idx, axis);
#else
  struct hash *orig = compressed_take(x, idx, axis);
  //hash_print(orig);
  struct hash *ent = hash_copy(orig);
#endif

  struct hash **ph = create_dict(ent);
  ret = PyCapsule_New(ph, "compressed_array_dict", destroy_dict);
  return ret;
}

static PyObject* set_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_ent;
  long idx, axis;

  if (!PyArg_ParseTuple(args, "OllO", &obj, &idx, &axis, &obj_ent))
    return NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");
  struct hash **ph = PyCapsule_GetPointer(obj_ent, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  struct hash *ent = *ph;

  /* XXX Should we copy or set a reference? */
  /* Depends on garbage collection from Python */
#if 0
  if (axis == 0) {
    x->rows[idx] = ent;
  }
  else {
    x->cols[idx] = ent;
  }
#else
  if (axis == 0) {
    hash_destroy(x->rows[idx]);
    x->rows[idx] = hash_copy(ent);
  }
  else {
    hash_destroy(x->cols[idx]);
    x->cols[idx] = hash_copy(ent);
  }
  
#endif

  Py_RETURN_NONE;
}

static PyObject* copy_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, * ret;

  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  struct hash *h = *ph;

  struct hash *h2 = hash_copy(h);
  struct hash **ph2 = create_dict(h2);

  ret = PyCapsule_New(ph2, "compressed_array_dict", destroy_dict);
  return ret;
}

static PyObject* sum_dict(PyObject *self, PyObject *args)
{
  PyObject *obj;

  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  uint64_t val = hash_sum(*ph);
  PyObject *ret = Py_BuildValue("k", val);
  return ret;
}


static PyObject* getitem_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_k;

  if (!PyArg_ParseTuple(args, "OO", &obj, &obj_k)) {
    return NULL;
  }

  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");

  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  struct hash *h = *ph;
  long k_int = PyLong_AsLongLong(obj_k);

  if (k_int != -1) {
    /* Return a single item. */
    int64_t val = 0;
    hash_search(h, k_int, &val);
    PyObject *ret = Py_BuildValue("k", val);
    return ret;
  }

  PyErr_Restore(NULL, NULL, NULL); /* clear the exception */  

  obj_k = PyArray_FROM_OTF(obj_k, NPY_LONG, NPY_IN_ARRAY);
  const uint64_t *keys = (const uint64_t *) PyArray_DATA(obj_k);
  long N = (long) PyArray_DIM(obj_k, 0);

  int64_t *vals = malloc(N * sizeof(int64_t));

  hash_search_multi(h, keys, vals, N);

  npy_intp dims[] = {N};
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);

  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  
  Py_DECREF(obj_k);
 
  PyObject *ret = Py_BuildValue("N", vals_obj);
  return ret;
}

static PyObject* copy(PyObject *self, PyObject *args)
{
  PyObject *obj, *ret;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  struct compressed_array *y = PyCapsule_GetPointer(obj, "compressed_array");
  struct compressed_array *x = compressed_copy(y);

  ret = PyCapsule_New(x, "compressed_array", destroy);
  return ret;
}


static PyObject* select_copy(PyObject *self, PyObject *args)
{
  PyObject *obj_dst, *obj_src, *obj_where;

  if (!PyArg_ParseTuple(args, "OOO", &obj_dst, &obj_src, &obj_where)) {
    return NULL;
  }

  struct compressed_array *src = PyCapsule_GetPointer(obj_src, "compressed_array");
  struct compressed_array *dst = PyCapsule_GetPointer(obj_dst, "compressed_array");  

  obj_where = PyArray_FROM_OTF(obj_where, NPY_LONG, NPY_IN_ARRAY);
  const long *where = (const long *) PyArray_DATA(obj_where);
  long i, W = (long) PyArray_DIM(obj_where, 0);

  for (i=0; i<W; i++) {
    long idx = where[i];
    /* XXX Should we copy the references or do a deep copy? */
#if 0
    dst->rows[idx] = src->rows[idx];
    dst->cols[idx] = src->cols[idx];
#else
    hash_destroy(dst->rows[idx]);
    hash_destroy(dst->cols[idx]);
    dst->rows[idx] = hash_copy(src->rows[idx]);
    dst->cols[idx] = hash_copy(src->cols[idx]);
#endif
  }

  Py_DECREF(obj_where);
  Py_RETURN_NONE;
}

/* XXX Error in this function */
static PyObject* setaxis(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_k, *obj_v;
  long i, axis = 0;

  if (!PyArg_ParseTuple(args, "OllOO", &obj, &i, &axis, &obj_k, &obj_v)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  /* Handle one dimension with multiple elements. */
  obj_k = PyArray_FROM_OTF(obj_k, NPY_LONG, NPY_IN_ARRAY);
  obj_v = PyArray_FROM_OTF(obj_v, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *keys = (const uint64_t *) PyArray_DATA(obj_k);
  const int64_t *vals = (const int64_t *) PyArray_DATA(obj_v);

  long N = (long) PyArray_DIM(obj_k, 0);
#if 0
  struct hash *h = (axis == 0 ? x->rows[i] : x->cols[i]);
  hash_set_multi(h, keys, vals, N);
#else
  long j;
  if (axis == 0) {
    for (j=0; j<N; j++) {
      compressed_set_single(x, i, keys[j], vals[j]);
    }
  }
  else {
    for (j=0; j<N; j++) {
      compressed_set_single(x, keys[j], i, vals[j]);
    }    
  }
#endif

  Py_DECREF(obj_k);
  Py_DECREF(obj_v);
  
  Py_RETURN_NONE;
}

static PyObject* setitem(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_i, *obj_j, *obj_k, *obj_v;

  if (!PyArg_ParseTuple(args, "OOOO", &obj, &obj_i, &obj_j, &obj_v)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  long axis = 0;

  long i = PyLong_AsLongLong(obj_i);
  long j = PyLong_AsLongLong(obj_j);

  if (i != -1 && j != -1) {
    /* Set a single item. */
    long val = PyLong_AsLongLong(obj_v);

    if (val < 0) {
      return NULL;
    }
   
    compressed_set_single(x, i, j, val);
    Py_RETURN_NONE;
  }

  /* Handle one dimension with multiple elements. */
  if (i == -1 && j != -1) {
    obj_k = PyArray_FROM_OTF(obj_i, NPY_LONG, NPY_IN_ARRAY);
    axis = 0;
  }
  else if (j == -1 && i != -1) {
    obj_k = PyArray_FROM_OTF(obj_j, NPY_LONG, NPY_IN_ARRAY);
    axis = 1;
  }
  else {
    return NULL;
  }

  PyErr_Restore(NULL, NULL, NULL); /* clear the exception */  

  const long *keys = (const long *) PyArray_DATA(obj_k);
  const long *vals = (const long *) PyArray_DATA(obj_v);
  long k, N = (long) PyArray_DIM(obj_k, 0);

  if (axis == 0) {
    for (k=0; k<N; k++) {
      compressed_set_single(x, keys[k], j, vals[k]);
    }
  }
  else {
    for (k=0; k<N; k++) {
      compressed_set_single(x, i, keys[k], vals[k]);
    }
  }

  Py_DECREF(obj_k);
  Py_DECREF(obj_v);
  
  Py_RETURN_NONE;
}


static PyObject* getitem(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_i, *obj_j, *py_arr;  
  long i, j;

  if (!PyArg_ParseTuple(args, "OOO", &obj, &obj_i, &obj_j)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  long axis = 0;

  i = PyLong_AsLongLong(obj_i);
  j = PyLong_AsLongLong(obj_j);

  if (i != -1 && j != -1) {
    /* Return a single item. */
    int64_t val = 0;
    compressed_get_single(x, i, j, &val);
    PyObject *ret = Py_BuildValue("k", val);
    return ret;
  }

  /* Handle one dimension with multiple elements. */
  if (i == -1 && j != -1) {
    py_arr = PyArray_FROM_OTF(obj_i, NPY_LONG, NPY_IN_ARRAY);
    axis = 0;
  }
  else if (j == -1 && i != -1) {
    py_arr = PyArray_FROM_OTF(obj_j, NPY_LONG, NPY_IN_ARRAY);
    axis = 1;
  }
  else {
    return NULL;
  }

  PyErr_Restore(NULL, NULL, NULL); /* clear the exception */  

  const long *arr = (const long *) PyArray_DATA(py_arr);
  long k, N = (long) PyArray_DIM(py_arr, 0);
  int64_t *vals = malloc(N * sizeof(int64_t));

  if (axis == 0) {
    for (k=0; k<N; k++) {
      compressed_get_single(x, arr[k], j, &vals[k]);
    }
  }
  else {
    for (k=0; k<N; k++) {
      compressed_get_single(x, i, arr[k], &vals[k]);
    }
  }
  
  npy_intp dims[] = {N};
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  

  Py_DECREF(py_arr);
  
  PyObject *ret = Py_BuildValue("N", vals_obj);
  return ret;
}

static PyObject* take(PyObject *self, PyObject *args)
{
  PyObject *obj;
  long i, axis;

  if (!PyArg_ParseTuple(args, "Oll", &obj, &i, &axis))
    return NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  long cnt;
  uint64_t *keys;
  int64_t *vals;

#if 0
  if (axis == 0) {
    hash_print(x->rows[i]);
  }
  else {
    hash_print(x->cols[i]);
  }
#endif

  compressed_take_keys_values(x, i, axis, &keys, &vals, &cnt);

#if 0
  fprintf(stderr, "Returnng %ld keys and values:", cnt);
  long j;
  for (j=0; j<cnt; j++) {
    fprintf(stderr, " %ld : %ld ", keys[j], vals[j]);
  }
  fprintf(stderr, "\n");
#endif

  if (cnt == 0) {
    Py_RETURN_NONE;
  }
  
  npy_intp dims[] = {cnt};

  PyObject *keys_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, keys);
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, vals);

  PyArray_ENABLEFLAGS((PyArrayObject*) keys_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);  

  PyObject *ret = Py_BuildValue("NN", keys_obj, vals_obj);
  return ret;
}

static PyObject* sanity_check(PyObject *self, PyObject *args)
{
  abort();
  
  PyObject *obj;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  size_t i, j;

  if (!x) {
    PyErr_SetString(PyExc_RuntimeError, "NULL pointer to compresed array");
    return NULL;
  }
  
  for (i=0; i<x->n_row; i++) {
    if (!x->rows[i] || !x->rows[i]->keys || !x->rows[i]->vals) {
      PyErr_SetString(PyExc_RuntimeError, "Invalid rows found");
      return NULL;
    }

    if (!x->cols[i] || !x->cols[i]->keys || !x->cols[i]->vals) {
       PyErr_SetString(PyExc_RuntimeError, "Invalid cols found");
       return NULL;       
    }

    for (i=0; i<x->n_row; i++) {
      for (j=0; j<x->rows[i]->width; j++) {
	if ((x->rows[i]->keys[j] & EMPTY_FLAG) == 0) {
	  if (x->rows[i]->keys[j] > 999) {
	    PyErr_SetString(PyExc_RuntimeError, "Invalid key value found");
	    return NULL;       	    
	  }
	}
      }
    }
  }
  Py_RETURN_NONE;
}


static PyMethodDef compressed_array_methods[] =
  {
   { "create", create, METH_VARARGS, "Create a new object." },
   { "copy", copy, METH_VARARGS, "Copy an existing object." },   
   { "setitem", setitem, METH_VARARGS, "Set an item." },
   { "setaxis", setaxis, METH_VARARGS, "Set items along an axis." },   
   { "getitem", getitem, METH_VARARGS, "Get an item." },
   { "take", take, METH_VARARGS, "Take items along an axis." },
   { "take_dict", take_dict, METH_VARARGS, "Take items along an axis in dict form." },
   { "accum_dict", accum_dict, METH_VARARGS, "Add to items in a dict slice." },
   { "keys_values_dict", keys_values_dict, METH_VARARGS, "Get keys and values from a dict slice." },      
   { "print_dict", print_dict, METH_VARARGS, "Print items along an axis in dict form." },
   { "empty_dict", empty_dict, METH_VARARGS, "New row dict." },
   { "getitem_dict", getitem_dict, METH_VARARGS, "Look up in a row dict." },
   { "set_dict", set_dict, METH_VARARGS, "Set a row dict." },
   { "copy_dict", copy_dict, METH_VARARGS, "Copy a row dict." },
   { "sum_dict", sum_dict, METH_VARARGS, "Sum the values of a dict." },   
   { "select_copy", select_copy, METH_VARARGS, "Selectively copy row and colummns." },
   { "sanity_check", sanity_check, METH_VARARGS, "Run a sanity check." },   
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
