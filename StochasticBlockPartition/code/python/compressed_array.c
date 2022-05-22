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
  size_t l; /* allocated size of each table */
  size_t m; /* m <= l */
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

  size_t table_size = (n_nodes * max_degree * 1.43);
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
  x->rows.l = max_degree;
  x->rows.m = max_degree;
  x->rows.n = n_nodes;
  
  x->cols.keys = (uint64_t *) x->buf + 2 * table_size;
  x->cols.vals = (uint64_t *) x->buf + 3 * table_size;
  x->cols.l = max_degree;
  x->cols.m = max_degree;
  x->cols.n = n_nodes;

  memset(x->buf, 0xff, buf_size);
  return x;
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

static void compressed_destroy(PyObject *obj)
{
  void *x = PyCapsule_GetPointer(obj, "compressed_array");
  fprintf(stderr, "Destroying %p\n", x);
  compressed_array_destroy(x);
}


static PyObject* compressed_create(PyObject *self, PyObject *args)
{
  PyObject *ret;
  long n_nodes, max_degree;

  if (!PyArg_ParseTuple(args, "ll", &n_nodes, &max_degree)) {
    return NULL;
  }

  struct compressed_array *x = compressed_array_create(n_nodes, max_degree);

  uint64_t flag = EMPTY_FLAG;
  fprintf(stderr, "flag is 0x%lx\n", flag);
  fprintf(stderr, "Created tables for n_nodes %ld with degree %ld = %p\n", n_nodes, max_degree, x);

  ret = PyCapsule_New(x, "compressed_array", compressed_destroy);
  fprintf(stderr, "New object ptr is %p\n", ret);
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

  PyObject *ret = Py_BuildValue("l", 0);
  return ret;
}

static PyMethodDef compressed_array_methods[] =
  {
   {
    "compressed_create",
    compressed_create,
    METH_VARARGS,
    "Create a new object."
   },
   {
    "get_magic",
    get_magic,
    METH_VARARGS,
    "Get magic value."
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

