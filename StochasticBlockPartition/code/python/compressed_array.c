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
#include <stdatomic.h>
#include <stdbool.h>
#include "shared_mem.h"

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
  size_t alloc_size;
  void *(*fn_malloc)(size_t);  
  void (*fn_free)(void *, size_t);
};

static void private_free(void *p, size_t len)
{
  free(p);
}

int hash_sanity_count(const char *msg, const struct hash *h)
{
  size_t i, sanity_cnt = 0;
  for (i=0; i<h->width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      sanity_cnt++;
    }
  }
  if (h->cnt != sanity_cnt) {
    fprintf(stderr, "Sanity count at %s for hash %p failed: cnt %ld h->cnt %ld\n", msg, h, sanity_cnt, h->cnt);
    return -1;
  }

  return 0;
}

struct hash *hash_create(size_t initial_size, int shared_mem)
{
  void (*fn_free)(void *, size_t);
  void *(*fn_malloc)(size_t);  
  
  if (shared_mem) {
    fn_malloc = shared_malloc;
    fn_free = shared_free;
  }
  else {
    fn_malloc = malloc;
    fn_free = private_free;
  }
  
  size_t buf_size = sizeof(struct hash) + 2 * initial_size * sizeof(uint64_t);

  char *buf = fn_malloc(buf_size);
  if (!buf) {
    fprintf(stderr, "hash_create(%ld): return NULL\n", initial_size);
    return NULL;
  }

  struct hash *h = (struct hash *) buf;
  h->width = initial_size;
  h->cnt = 0;
  h->limit = h->width * 0.70;
  h->keys = (uint64_t *) (buf + sizeof(struct hash) + 0 * h->width * sizeof(uint64_t));
  h->vals = (int64_t *) (buf + sizeof(struct hash) + 1 * h->width * sizeof(int64_t));
  h->alloc_size = buf_size;
  h->fn_malloc = fn_malloc;
  h->fn_free = fn_free;

  size_t i;
  for (i=0; i<h->width; i++) {
    h->keys[i] = EMPTY_FLAG;
    h->vals[i] = 0;
  }

  return h;
}

void hash_destroy(struct hash *h)
{
  if (h) {
    h->fn_free(h, h->alloc_size);
  }
}

struct hash *hash_copy(const struct hash *y, int shared_mem)
{
  void (*fn_free)(void *, size_t);
  void *(*fn_malloc)(size_t);  
  
  if (shared_mem) {
    fn_malloc = shared_malloc;
    fn_free = shared_free;
  }
  else {
    fn_malloc = malloc;
    fn_free = private_free;
  }
  
  size_t buf_size = sizeof(struct hash) + 2 * y->width * sizeof(uint64_t);

  char *buf = fn_malloc(buf_size);

  if (!buf) {
    fprintf(stderr, "hash_copy: return NULL\n");    
    return NULL;
  }

  struct hash *h = (struct hash *) buf;
  h->width = y->width;
  h->cnt = y->cnt;
  h->limit = y->limit;
  h->keys = (uint64_t *) (buf + sizeof(struct hash) + 0 * h->width * sizeof(uint64_t));
  h->vals = (int64_t *) (buf + sizeof(struct hash) + 1 * h->width * sizeof(int64_t));
  h->alloc_size = buf_size;
  h->fn_malloc = fn_malloc;
  h->fn_free = fn_free;

  memcpy(h->keys, y->keys, y->width * sizeof(uint64_t));
  memcpy(h->vals, y->vals, y->width * sizeof(int64_t));

  return h;
}

void hash_print(struct hash *h);
struct hash *hash_set_single(struct hash *h, uint64_t k, int64_t v);
#define RESIZE_DEBUG (0)

static inline struct hash *hash_resize(struct hash *h)
{
  size_t i;
  struct hash *h2;

  if (h->cnt == h->limit) {
    /* Resize needed */

    /* XXX Remove sanity check */
    hash_sanity_count("hash_resize", h);
    
#if RESIZE_DEBUG
    fprintf(stderr, "Before resize cnt is %ld\n", h->cnt);
    hash_print(h);
#endif

    int shared_mem = (h->fn_malloc == malloc) ? 0 : 1;
    
    h2 = hash_create(h->width * 2, shared_mem);
    if (!h2) {
      fprintf(stderr, "hash_resize: hash_create to width %ld from %ld failed\n", h->width * 2, h->limit);
      return NULL;
    }
#if RESIZE_DEBUG
    fprintf(stderr, "insert: ");
#endif

    long ins = 0;
    for (i=0; i<h->width; i++) {
      if ((h->keys[i] & EMPTY_FLAG) == 0) {
	//fprintf(stderr, " %ld ", h->keys[i]);	
	hash_set_single(h2, h->keys[i], h->vals[i]);
	ins++;
      }
    }
#if RESIZE_DEBUG
    fprintf(stderr, "\nAfter resize inserted %ld\n", ins);
    hash_print(h2);
    fprintf(stderr, "\n\n");
#endif        

    if (h->cnt != h2->cnt) {
      fprintf(stderr, "Mismatch found in hash %p h1->cnt %ld h2->cnt %ld ins %ld\n", h, h->cnt, h2->cnt, ins);
      return NULL;
    }

    hash_destroy(h);
    h = h2;
  }

  return h;
}


struct hash *hash_set_single(struct hash *h, uint64_t k, int64_t v)
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

  if (!h) {
    fprintf(stderr, "Warning in hash_set_single: hash_resize failed\n");
  }
  
  return h;
}


int hash_accum_single(struct hash *h, uint64_t k, int64_t c)
{
  /* To avoid a subtle logic bug, first check for existance. 
   * Beacuse not every insertion will cause an increase in cnt.
   */
  size_t i, width = h->width;
  uint64_t kh = hash(k);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
#if 0
    if (h->keys[idx] == k) {
      h->vals[idx] += c;
      break;      
    }
    else if (h->keys[idx] & EMPTY_FLAG) {
      h->keys[idx] = k & ~EMPTY_FLAG;
      h->vals[idx] = c;
      h->cnt++;
      break;
    }
#else
    /* Try experimental lock-free approach */
    atomic_long empty = EMPTY_FLAG;
    _Bool rc = atomic_compare_exchange_strong((atomic_ulong *) &h->keys[idx], &empty, (atomic_ulong) k);

    if (rc) {
      /* Was empty, and new key inserted. It is safe to add instead of assign because vals were all initialized to zero.*/
      atomic_fetch_add_explicit((atomic_ulong *) &h->vals[idx], c, memory_order_relaxed);
      /* And also do increase the count by 1. */
      atomic_fetch_add_explicit((atomic_ulong *) &h->cnt, 1, memory_order_relaxed);
      break;
    }
    else if (h->keys[idx] == k) {
      /* Was not empty. Check the existing key. */
      atomic_fetch_add_explicit((atomic_ulong *) &h->vals[idx], c, memory_order_relaxed);
      break;
    }
#endif
  }


  if (h->cnt >= h->limit) {
	  return 1; /* resize needed */
  }
  else {
	  return 0;
  }
}


static inline int hash_search(const struct hash *h, uint64_t k, int64_t *v)
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

static inline void hash_search_multi(const struct hash *h, const uint64_t *keys, int64_t *vals, size_t n)
{
  size_t i;

  for (i=0; i<n; i++) {
    hash_search(h, keys[i], &vals[i]);
  }
}

static inline int64_t hash_sum(const struct hash *h)
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


static inline size_t hash_keys(const struct hash *h, uint64_t *keys, size_t max_cnt)
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

size_t hash_vals(const struct hash *h, int64_t *vals, size_t max_cnt)
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

int hash_eq(const struct hash *h, const struct hash *h2)
{
  size_t i;

  for (i=0; i<h->width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      int64_t v2 = 0;
      hash_search(h2, h->keys[i], &v2);
      if (v2 != h->vals[i]) {
	fprintf(stderr, "Mismatch at key %lu\n", h->keys[i]);
	return 1;
      }
    }
  }

  for (i=0; i<h2->width; i++) {
    if ((h2->keys[i] & EMPTY_FLAG) == 0) {
      int64_t v = 0;
      hash_search(h, h2->keys[i], &v);
      if (v != h2->vals[i]) {
	fprintf(stderr, "Mismatch at key %lu\n", h2->keys[i]);	
	return 1;
      }
    }
  }

  return 0;
}

static inline void hash_accum_constant(const struct hash *h, size_t C)
{
  size_t i, width = h->width;

  for (i=0; i<width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      h->vals[i] += C;
    }
  }
}

static inline struct hash *hash_accum_multi(struct hash *h, const uint64_t *keys, const int64_t *vals, size_t n_keys)
{
  size_t j, i;

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

    if (!h) {
      fprintf(stderr, "Warning in hash_accum_multi: hash_resize failed\n");
      return NULL;
    }
  }

#if 1  
  int flag = 0;
  if (h->cnt == h->limit) {
    fprintf(stderr, "Resize on accum before %ld %ld\n", h->cnt, h->limit);
    flag = 1;
  }
#endif  

#if 1
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
  size_t i;
  
  x->n_row = n_nodes;
  x->n_col = n_nodes;

  x->rows = shared_calloc(x->n_row, sizeof(struct hash *));

  if (!x->rows) {
    free(x);
    return NULL;
  }

  x->cols = shared_calloc(x->n_col, sizeof(struct hash *));

  if (!x->cols) {
    shared_free(x->rows, x->n_row * sizeof(struct hash *));
    free(x);
    return NULL;
  }
  
  for (i=0; i<n_nodes; i++) {
    x->rows[i] = hash_create(initial_width * 2, 1);
    x->cols[i] = hash_create(initial_width * 2, 1);
    if (!x->rows[i] || !x->cols[i]) {
      fprintf(stderr, "compressed_array_create: hash_create failed\n");
      return NULL;
      break;
    }
  }

  if (i != n_nodes) {
    for (; i>=0; i--) {
      hash_destroy(x->rows[i]);
      hash_destroy(x->cols[i]);
    }
    shared_free(x->rows, x->n_row * sizeof(struct hash *));
    shared_free(x->cols, x->n_col * sizeof(struct hash *));
    free(x);
    fprintf(stderr, "compressed_array_create return NULL\n");
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

  shared_free(x->rows, x->n_row * sizeof(struct hash *));
  shared_free(x->cols, x->n_col * sizeof(struct hash *));
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
  x->rows = shared_calloc(x->n_row, sizeof(struct hash *));

  if (!x->rows) {
    free(x);
    return NULL;
  }

  x->cols = shared_calloc(x->n_col, sizeof(struct hash *));

  if (!x->cols) {
    shared_free(x->rows, x->n_row * sizeof(struct hash *));
    free(x);
    return NULL;
  }
  
  for (i=0; i<x->n_row; i++) {
    x->rows[i] = hash_copy(y->rows[i], 1);
    x->cols[i] = hash_copy(y->cols[i], 1);
    if (!x->rows[i] || !x->cols[i]) {
      fprintf(stderr, "compressed_copy failed: hash_copy returned NULL\n");
      break;
    }
  }

  if (i != x->n_row) {
    do {
      hash_destroy(x->rows[i]);
      hash_destroy(x->cols[i]);
    }
    while (i-- != 0);

    shared_free(x->rows, x->n_row * sizeof(struct hash *));
    shared_free(x->cols, x->n_col * sizeof(struct hash *));    
    free(x);
    return NULL;
  }

  return x;
}

static inline int compressed_get_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t *val)
{
  /* Just get from row[i][j] */
  return hash_search(x->rows[i], j, val);
}

static inline void compressed_set_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t val)
{
  /* XXX There is a bug in this logic if exactly one fails. */
  x->rows[i] = hash_set_single(x->rows[i], j, val);
  x->cols[j] = hash_set_single(x->cols[j], i, val);
}

/* XXX TODO Merge old and new */
static inline int compressed_accum_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t C)
{
#if 0
  int rc;
  rc = hash_accum_single(x->rows[i], j, C);
  if (rc) {
    if ((x->rows[i] = hash_resize(x->rows[i])) < 0) { return -1; }
    fprintf(stderr, "Done row resize\n");
  }
  rc = hash_accum_single(x->cols[j], i, C);
  if (rc) {
    if ((x->cols[j] = hash_resize(x->cols[j])) < 0) { return -1; }
    fprintf(stderr, "Done col resize\n");
  }
  return 0;
#elif 0
  int rc;
  rc = hash_accum_single(x->rows[i], j, C);
#if 0
  static int cnt = 0;
  if (cnt++ < 2) {
    struct hash *old = x->rows[i];
    struct hash *new = hash_copy(old, 1);
    fprintf(stderr, "copied %p into row %ld!\n", new, i);
    x->rows[i] = new;
    hash_destroy(old);
  }
#endif
		       
  if (rc) {
    if ((x->rows[i] = hash_resize(x->rows[i])) < 0) { return -1; }
    fprintf(stderr, "Done row resize\n");
    return -1;
  }
  rc = hash_accum_single(x->cols[j], i, C);
  if (rc) {
    if ((x->cols[j] = hash_resize(x->cols[j])) < 0) { return -1; }
    fprintf(stderr, "Done col resize\n");
    return -1;
  }
  return 0;	
#else
  size_t ii;
  int rc;

  struct hash *old_row = x->rows[i], *new_row;
  rc = hash_accum_single(x->rows[i], j, C);

  if (rc == 1) {
    new_row = hash_create(old_row->width * 2, 1);

    if (!new_row) {
      fprintf(stderr, "compressed_accum_single: hash_create row failed\n");
      return -1;
    }

    _Bool rc = false;
    rc = atomic_compare_exchange_strong((atomic_uintptr_t *) &x->rows[i], &old_row, (atomic_uintptr_t) new_row);
    if (!rc) { /* Someone else won the race */
      hash_destroy(new_row);
    }
    else { /* We won the row race. */
      fprintf(stderr, "We won the row race! Merge into hash %p from %p\n", new_row, old_row);
      long ins = 0;
      for (ii=0; ii<old_row->width; ii++) {
	uint64_t k = atomic_load_explicit((atomic_ulong *) &old_row->keys[ii], memory_order_relaxed);
	int64_t v = atomic_load_explicit((atomic_long *) &old_row->vals[ii], memory_order_relaxed);
	if ((k != EMPTY_FLAG)) {
	  hash_accum_single(new_row, k, v);					
	  ins++;
	}
      }
    }
  }

  struct hash *old_col = x->cols[j], *new_col;
  rc = hash_accum_single(x->cols[j], i, C);

  if (rc == 1) {
    new_col = hash_create(old_col->width * 2, 1);

    if (!new_col) {
      fprintf(stderr, "compressed_accum_single: hash_create col failed\n");
      //hash_destroy(new_row);
      return -1;
    }

    _Bool rc = false;
    rc = atomic_compare_exchange_strong((atomic_uintptr_t *) &x->cols[j], &old_col, (atomic_uintptr_t) new_col);
    if (!rc) { /* Someone else won the race */
      hash_destroy(new_col);
    }
    else { /* We won the race. */
      fprintf(stderr, "We won the col race! Merge into hash %p from %p\n", new_col, old_col);
      long ins = 0;
      for (ii=0; ii<old_col->width; ii++) {
	uint64_t k = atomic_load_explicit((atomic_ulong *) &old_col->keys[ii], memory_order_relaxed);
	int64_t v = atomic_load_explicit((atomic_long *) &old_col->vals[ii], memory_order_relaxed);
	if ((k != EMPTY_FLAG)) {
	  hash_accum_single(new_col, k, v);					
	  ins++;
	}
      }
    }
  }
#endif

  return 0;
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

static inline struct hash *compressed_take(struct compressed_array *x, long idx, long axis)
{
  return (axis == 0 ? x->rows[idx] : x->cols[idx]);
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
  ret = PyCapsule_New(x, "compressed_array", destroy);
  return ret;
}

/* xxx should be shared or not? */
static inline struct hash **create_dict(struct hash *p)
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

/* Return keys and values from a dict. */
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

  hash_sanity_count("accum_dict", h);
  
  h = hash_accum_multi(h, keys, vals, N);

  if (!h) {
    PyErr_SetString(PyExc_RuntimeError, "hash_accum_multi failed");
    return NULL;
  }

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

  struct hash *ent = hash_create(N, 0);

  if (!ent) {
    PyErr_SetString(PyExc_RuntimeError, "empty_dict: hash_create failed");    
    return NULL;
  }
  
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
  struct hash *ent = hash_copy(orig, 0);

  int ok1 = hash_sanity_count("take_dict orig", orig);
  int ok2 = hash_sanity_count("take_dict ent", ent);

  if (ok1 != ok2) {
    fprintf(stderr, "take_dict found different sanity for orig (cnt %ld) and copy ent (cnt %ld). Major weirdness!\n", orig->cnt, ent->cnt);
  }
  
  
  if (!ent) {
    PyErr_SetString(PyExc_RuntimeError, "take_dict: hash_copy failed");    
    return NULL;
  }
  
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
    x->rows[idx] = hash_copy(ent, 1);
  }
  else {
    hash_destroy(x->cols[idx]);
    x->cols[idx] = hash_copy(ent, 1);
  }
  
#endif

  Py_RETURN_NONE;
}


static PyObject* eq_dict(PyObject *self, PyObject *args)
{
  PyObject *obj_h1, *obj_h2;

  if (!PyArg_ParseTuple(args, "OO", &obj_h1, &obj_h2))
    return NULL;

  struct hash **ph1 = PyCapsule_GetPointer(obj_h1, "compressed_array_dict");

  if (!ph1) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  struct hash **ph2 = PyCapsule_GetPointer(obj_h2, "compressed_array_dict");  

  if (!ph2) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }

  long val = hash_eq(*ph1, *ph2);
  
  PyObject *ret = Py_BuildValue("k", val);
  return ret;
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

  struct hash *h2 = hash_copy(h, 0);

  // fprintf(stderr, "    pid %d copied into hash %p h count %ld h2 count %ld\n", getpid(), h2, h->cnt, h2->cnt);

  hash_sanity_count("copy_dict h", h);
  hash_sanity_count("copy_dict h2", h2);
  
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

  if (!y) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid reference to compressed_array.");    
    return NULL;
  }

  struct compressed_array *x = compressed_copy(y);

  if (!x) {
    PyErr_SetString(PyExc_RuntimeError, "compresed array copy failed");
    return NULL;
  }

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
    dst->rows[idx] = hash_copy(src->rows[idx], 1);
    dst->cols[idx] = hash_copy(src->cols[idx], 1);
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

static PyObject* setaxis_from_dict(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_d;
  long i, axis = 0;

  if (!PyArg_ParseTuple(args, "OllO", &obj, &i, &axis, &obj_d)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");
  struct hash **ph = PyCapsule_GetPointer(obj_d, "compressed_array_dict");
  struct hash *h = *ph;

  unsigned long j;
  for (j=0; j<h->width; j++) {
    if ((h->keys[j] & EMPTY_FLAG) == 0) {
      if (axis == 0) {
	compressed_set_single(x, i, h->keys[j], h->vals[j]);
      }
      else {
	compressed_set_single(x, h->keys[j], i, h->vals[j]);
      }
    }
  }

  /* Handle one dimension with multiple elements. */
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

#if 0
  /* Large sized graph seems to trigger the empty case. */
  if (cnt == 0) {
    Py_RETURN_NONE;
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


/*
 * Compute the delta entropy for a row using a compressed hash only.
 A typical call in Python looks like this:
 d0 = entropy_row_nz(M_r_row_v, d_in_new[M_r_row_i], d_out_new[r])
*/

static inline double entropy_row(struct hash *h, const int64_t *restrict deg, long N, int64_t c)
{
  double sum = 0, log_c;
  size_t i;

  if (c == 0) {
    return 0.0;
  }

  log_c = log(c);

  /* Iterate over keys and values */
  for (i=0; i<h->width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0) {
      int64_t xi = h->vals[i];
      int64_t yi = deg[h->keys[i]];
      if (xi > 0 && yi > 0) {
	sum += xi * (log(xi) - log(yi) - log_c);
      }
    }
  }

  return sum;
}

static PyObject* dict_entropy_row(PyObject *self, PyObject *args)
{
  PyObject *hash_obj, *deg_obj;
  double c;

  if (!PyArg_ParseTuple(args, "OOd", &hash_obj, &deg_obj, &c))
    return NULL;


  struct hash **ph = PyCapsule_GetPointer(hash_obj, "compressed_array_dict");
  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }
  struct hash *h = *ph;

  PyObject *deg_array = PyArray_FROM_OTF(deg_obj, NPY_LONG, NPY_IN_ARRAY);
  long N = (long) PyArray_DIM(deg_array, 0);

  const int64_t *deg = (const int64_t *) PyArray_DATA(deg_array);

  double val = entropy_row(h, deg, N, c);

  Py_DECREF(deg_array);

  PyObject *ret = Py_BuildValue("d", val);
  return ret;
}


static inline double entropy_row_excl(struct hash *h, const int64_t *restrict deg, long N, int64_t c, uint64_t r, uint64_t s)
{
  double sum = 0, log_c;
  size_t i;

  if (c == 0) {
    return 0.0;
  }

  log_c = log(c);

  /* Iterate over keys and values */
  for (i=0; i<h->width; i++) {
    if ((h->keys[i] & EMPTY_FLAG) == 0 && h->keys[i] != r && h->keys[i] != s) {
      int64_t xi = h->vals[i];
      int64_t yi = deg[h->keys[i]];
      if (xi > 0 && yi > 0) {
	sum += xi * (log(xi) - log(yi) - log_c);
      }
    }
  }

  return sum;
}

static PyObject* dict_entropy_row_excl(PyObject *self, PyObject *args)
{
  PyObject *hash_obj, *deg_obj;
  double c;
  uint64_t r, s;

  if (!PyArg_ParseTuple(args, "OOdll", &hash_obj, &deg_obj, &c, &r, &s))
    return NULL;


  struct hash **ph = PyCapsule_GetPointer(hash_obj, "compressed_array_dict");
  if (!ph) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid compressed_array_dict object");
    return NULL;
  }
  struct hash *h = *ph;

  PyObject *deg_array = PyArray_FROM_OTF(deg_obj, NPY_LONG, NPY_IN_ARRAY);
  long N = (long) PyArray_DIM(deg_array, 0);

  const int64_t *deg = (const int64_t *) PyArray_DATA(deg_array);

  double val = entropy_row_excl(h, deg, N, c, r, s);

  Py_DECREF(deg_array);

  PyObject *ret = Py_BuildValue("d", val);
  return ret;
}


/* 
 * Args: M, r, s, b_out, count_out, b_in, count_in
 */
static PyObject* inplace_compute_new_rows_cols_interblock_edge_count_matrix(PyObject *self, PyObject *args)
{
  PyObject *obj_M, *obj_b_out, *obj_count_out, *obj_b_in, *obj_count_in, *obj_d_out, *obj_d_in, *obj_d;
  uint64_t r, s;

  if (!PyArg_ParseTuple(args, "OllOOOOOOO", &obj_M, &r, &s, &obj_b_out, &obj_count_out, &obj_b_in, &obj_count_in, &obj_d_out, &obj_d_in, &obj_d)) {
    return NULL;
  }

  struct compressed_array *M = PyCapsule_GetPointer(obj_M, "compressed_array");

  if (!M) {
    PyErr_SetString(PyExc_RuntimeError, "NULL pointer to compresed array");
    return NULL;
  }

  const PyObject *ar_b_out = PyArray_FROM_OTF(obj_b_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_out = PyArray_FROM_OTF(obj_count_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_b_in = PyArray_FROM_OTF(obj_b_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_in = PyArray_FROM_OTF(obj_count_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_out = PyArray_FROM_OTF(obj_d_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_in = PyArray_FROM_OTF(obj_d_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d = PyArray_FROM_OTF(obj_d, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *b_out = (const uint64_t *) PyArray_DATA(ar_b_out);
  const int64_t *count_out = (const int64_t *) PyArray_DATA(ar_count_out);
  const uint64_t *b_in = (const uint64_t *) PyArray_DATA(ar_b_in);
  const int64_t *count_in = (const int64_t *) PyArray_DATA(ar_count_in);
  atomic_long * d_out = (atomic_long *) PyArray_DATA(ar_d_out);
  atomic_long * d_in = (atomic_long *) PyArray_DATA(ar_d_in);
  atomic_long * d = (atomic_long *) PyArray_DATA(ar_d);

  long n_out= (long) PyArray_DIM(ar_b_out, 0);
  long n_in = (long) PyArray_DIM(ar_b_in, 0);

  long i;
  int64_t dM_r_row_sum = 0, dM_r_col_sum = 0;
  
  for (i=0; i<n_out; i++) {
    /* M[r, b_out[i]] -= count_out[i] */
    /* M[s, b_out[i]] += count_out[i] */
    dM_r_row_sum -= count_out[i];
    if (compressed_accum_single(M, r, b_out[i], -count_out[i])) { goto bad; }
    if (compressed_accum_single(M, s, b_out[i], +count_out[i])) { goto bad; }
  }
  
  for (i=0; i<n_in; i++) {
    /* M[b_in[i], r] -= count_in[i] */
    /* M[b_in[i], s] += count_in[i] */
    dM_r_col_sum -= count_in[i];
    if (compressed_accum_single(M, b_in[i], r, -count_in[i])) { goto bad; }
    if (compressed_accum_single(M, b_in[i], s, +count_in[i])) { goto bad; }

#if 0
    /* XXX */
    struct hash *hh = M->rows[b_in[i]];
    M->rows[b_in[i]] = hash_copy(hh, 1);
    hash_destroy(hh);
#endif    
  }
  
  atomic_fetch_add_explicit(&d_out[r], dM_r_row_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_out[s], -dM_r_row_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_in[r], dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_in[s], -dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d[r], dM_r_row_sum + dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d[s], -dM_r_row_sum - dM_r_col_sum, memory_order_relaxed);
  
  Py_DECREF(ar_b_out);
  Py_DECREF(ar_count_out);
  Py_DECREF(ar_b_in);
  Py_DECREF(ar_count_in);
  Py_DECREF(ar_d_out);
  Py_DECREF(ar_d_in);
  Py_DECREF(ar_d);

  PyObject *ret = Py_BuildValue("llll", dM_r_row_sum, dM_r_col_sum, -dM_r_row_sum, -dM_r_col_sum);
  return ret;

bad:
  PyErr_SetString(PyExc_RuntimeError, "Update interblock edge count failed.");
  return NULL;
}


/* 
 * Args: partition, neighbors, weights
 */
static PyObject* blocks_and_counts(PyObject *self, PyObject *args)
{
  PyObject *obj_partition, *obj_neighbors, *obj_weights;

  if (!PyArg_ParseTuple(args, "OOO", &obj_partition, &obj_neighbors, &obj_weights)) {
    return NULL;
  }

  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);  
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_weights = PyArray_FROM_OTF(obj_weights, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *partition = (const uint64_t *) PyArray_DATA(ar_partition);
  const uint64_t *neighbors = (const uint64_t *) PyArray_DATA(ar_neighbors);  
  const uint64_t *weights = (const uint64_t *) PyArray_DATA(ar_weights);

  long i;
  long n = (long) PyArray_DIM(ar_neighbors, 0);

  /* Reserve at least 2x elements to avoid resizes. */
  struct hash *h = hash_create(2 * n, 0);

  for (i=0; i<n; i++) {
    uint64_t k = partition[neighbors[i]];
    uint64_t w = weights[i];
    if (hash_accum_single(h, k, w) == 1) {
	    h = hash_resize(h);
	    if (!h) {
	      return NULL;
	    }
    }
  }
  
  Py_DECREF(ar_partition);
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_weights);

  uint64_t *blocks = malloc(n * sizeof(uint64_t));
  int64_t *counts = malloc(n * sizeof(int64_t));  

  long cnt;
  cnt = hash_keys(h, blocks, n);
  hash_vals(h, counts, n);

  hash_destroy(h);

  npy_intp dims[] = {cnt};
  PyObject *blocks_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, blocks);
  PyObject *counts_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, counts);

  PyArray_ENABLEFLAGS((PyArrayObject*) blocks_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) counts_obj, NPY_ARRAY_OWNDATA);  

  PyObject *ret = Py_BuildValue("NN", blocks_obj, counts_obj);
  return ret;
}

/* 
 * Args: M, r, s, b_out, count_out, b_in, count_in
 * Version for an uncompressed array, using atomic operations.
 */
static PyObject* inplace_atomic_new_rows_cols_M(PyObject *self, PyObject *args)
{
  PyObject *obj_M, *obj_b_out, *obj_count_out, *obj_b_in, *obj_count_in, *obj_d_out, *obj_d_in, *obj_d;
  uint64_t r, s;

  if (!PyArg_ParseTuple(args, "OllOOOOOOO", &obj_M, &r, &s, &obj_b_out, &obj_count_out, &obj_b_in, &obj_count_in, &obj_d_out, &obj_d_in, &obj_d)) {
    return NULL;
  }

  const PyObject *M = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_b_out = PyArray_FROM_OTF(obj_b_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_out = PyArray_FROM_OTF(obj_count_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_b_in = PyArray_FROM_OTF(obj_b_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_in = PyArray_FROM_OTF(obj_count_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_out = PyArray_FROM_OTF(obj_d_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_in = PyArray_FROM_OTF(obj_d_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d = PyArray_FROM_OTF(obj_d, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *b_out = (const uint64_t *) PyArray_DATA(ar_b_out);
  const int64_t *count_out = (const int64_t *) PyArray_DATA(ar_count_out);
  const uint64_t *b_in = (const uint64_t *) PyArray_DATA(ar_b_in);
  const int64_t *count_in = (const int64_t *) PyArray_DATA(ar_count_in);
  atomic_long * d_out = (atomic_long *) PyArray_DATA(ar_d_out);
  atomic_long * d_in = (atomic_long *) PyArray_DATA(ar_d_in);
  atomic_long * d = (atomic_long *) PyArray_DATA(ar_d);    


  long n_out= (long) PyArray_DIM(ar_b_out, 0);
  long n_in = (long) PyArray_DIM(ar_b_in, 0);  
  long i;

  int64_t dM_r_row_sum = 0, dM_r_col_sum = 0;

  for (i=0; i<n_out; i++) {
    /* M[r, b_out[i]] -= count_out[i] */
    /* M[s, b_out[i]] += count_out[i] */
    dM_r_row_sum -= count_out[i];
    atomic_fetch_add_explicit((atomic_long *) PyArray_GETPTR2(M, r, b_out[i]), -count_out[i], memory_order_relaxed);
    atomic_fetch_add_explicit((atomic_long *) PyArray_GETPTR2(M, s, b_out[i]), +count_out[i], memory_order_relaxed);
  }
  
  for (i=0; i<n_in; i++) {
    /* M[b_in[i], r] -= count_in[i] */
    /* M[b_in[i], s] += count_in[i] */
    dM_r_col_sum -= count_in[i];
    atomic_fetch_add_explicit((atomic_long *) PyArray_GETPTR2(M, b_in[i], r), -count_in[i], memory_order_relaxed);
    atomic_fetch_add_explicit((atomic_long *) PyArray_GETPTR2(M, b_in[i], s), +count_in[i], memory_order_relaxed);
  }

  atomic_fetch_add_explicit(&d_out[r], dM_r_row_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_out[s], -dM_r_row_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_in[r], dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d_in[s], -dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d[r], dM_r_row_sum + dM_r_col_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&d[s], -dM_r_row_sum - dM_r_col_sum, memory_order_relaxed);

  Py_DECREF(M);
  Py_DECREF(ar_b_out);
  Py_DECREF(ar_count_out);
  Py_DECREF(ar_b_in);
  Py_DECREF(ar_count_in);
  Py_DECREF(ar_d_out);
  Py_DECREF(ar_d_in);
  Py_DECREF(ar_d);  

  PyObject *ret = Py_BuildValue("llll", dM_r_row_sum, dM_r_col_sum, -dM_r_row_sum, -dM_r_col_sum);
  return ret;
}



/* 
 * Args: partition, vertex_id_start, vertex_id_end, neighbors[vid], weights[vid]
 * Version for an uncompressed array, using atomic operations.
 */
static PyObject* rebuild_M(PyObject *self, PyObject *args)
{

  PyObject *obj_partition, *obj_neighbors, *obj_weights, *obj_M;
  uint64_t vid_start, vid_end;

  if (!PyArg_ParseTuple(args, "OllOOO", &obj_partition, &vid_start, &vid_end, &obj_neighbors, &obj_weights, &obj_M)) {
    return NULL;
  }

  const PyObject *M = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);  
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_weights = PyArray_FROM_OTF(obj_weights, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *partition = (const uint64_t *) PyArray_DATA(ar_partition);
  const uint64_t *neighbors = (const uint64_t *) PyArray_DATA(ar_neighbors);  
  const uint64_t *weights = (const uint64_t *) PyArray_DATA(ar_weights);

  long i;
  long n = (long) PyArray_DIM(ar_neighbors, 0);

  uint64_t k1 = partition[vid_start];
  
  for (i=0; i<n; i++) {
    uint64_t k2 = partition[neighbors[i]];
    uint64_t w = weights[i];
    atomic_fetch_add_explicit((atomic_long *) PyArray_GETPTR2(M, k1, k2), w, memory_order_relaxed);
  }

  Py_DECREF(M);
  Py_DECREF(ar_partition);
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_weights);
  Py_RETURN_NONE;
}

static PyObject* shared_memory_report(PyObject *self, PyObject *args)
{
  shared_print_report();
  Py_RETURN_NONE;
}

static PyObject* shared_memory_query(PyObject *self, PyObject *args)
{
  long i;
  size_t n_items;

  long *used = calloc(SHARED_MAX_POOLS, sizeof(long));
  long *avail = calloc(SHARED_MAX_POOLS, sizeof(long));

  for (i=0; i<SHARED_MAX_POOLS; i++) {
    shared_query(i, &used[i], &avail[i]);
  }

  npy_intp dims[] = {SHARED_MAX_POOLS};
  PyObject *used_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, used);
  PyObject *avail_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, avail);  

  PyArray_ENABLEFLAGS((PyArrayObject*) used_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) avail_obj, NPY_ARRAY_OWNDATA);

  PyObject *ret = Py_BuildValue("NN", used_obj, avail_obj);
  return ret;
}

static PyObject* shared_memory_reserve(PyObject *self, PyObject *args)
{
  PyObject *ret;
  long pool_id, n_items;

  if (!PyArg_ParseTuple(args, "ll", &pool_id, &n_items)) {
    return NULL;
  }

  shared_reserve(pool_id, n_items);
  Py_RETURN_NONE;
}

static PyMethodDef compressed_array_methods[] =
  {
   { "create", create, METH_VARARGS, "Create a new object." },
   { "copy", copy, METH_VARARGS, "Copy an existing object." },   
   { "setitem", setitem, METH_VARARGS, "Set an item." },
   { "setaxis", setaxis, METH_VARARGS, "Set items along an axis from key and value arrays." },
   { "setaxis_from_dict", setaxis_from_dict, METH_VARARGS, "Set items along an axis from another dict." },
   { "getitem", getitem, METH_VARARGS, "Get an item." },
   { "take", take, METH_VARARGS, "Take items along an axis." },
   { "take_dict", take_dict, METH_VARARGS, "Take items along an axis in dict form." },
   { "accum_dict", accum_dict, METH_VARARGS, "Add to items in a dict slice." },
   { "keys_values_dict", keys_values_dict, METH_VARARGS, "Get keys and values from a dict slice." },      
   { "print_dict", print_dict, METH_VARARGS, "Print items along an axis in dict form." },
   { "empty_dict", empty_dict, METH_VARARGS, "New row dict." },
   { "getitem_dict", getitem_dict, METH_VARARGS, "Look up in a row dict." },
   { "set_dict", set_dict, METH_VARARGS, "Set a row dict." },
   { "eq_dict", eq_dict, METH_VARARGS, "Compare two dicts." },
   { "copy_dict", copy_dict, METH_VARARGS, "Copy a row dict." },
   { "sum_dict", sum_dict, METH_VARARGS, "Sum the values of a dict." },   
   { "select_copy", select_copy, METH_VARARGS, "Selectively copy row and colummns." },
   { "sanity_check", sanity_check, METH_VARARGS, "Run a sanity check." },   
   { "dict_entropy_row", dict_entropy_row, METH_VARARGS, "Compute part of delta entropy for a row entry." },
   { "dict_entropy_row_excl", dict_entropy_row_excl, METH_VARARGS, "Compute part of delta entropy for a row entry." },
   { "inplace_compute_new_rows_cols_interblock_edge_count_matrix", inplace_compute_new_rows_cols_interblock_edge_count_matrix, METH_VARARGS, "Move node from block r to block s and apply changes to interblock edge count matrix, and other algorithm state." },
   { "blocks_and_counts", blocks_and_counts, METH_VARARGS, "" },
   { "inplace_atomic_new_rows_cols_M", inplace_atomic_new_rows_cols_M, METH_VARARGS, "" },
   { "rebuild_M", rebuild_M, METH_VARARGS, "" },
   { "shared_memory_report", shared_memory_report, METH_VARARGS, "" },
   { "shared_memory_query", shared_memory_query, METH_VARARGS, "" },   
   { "shared_memory_reserve", shared_memory_reserve, METH_VARARGS, "" },
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
