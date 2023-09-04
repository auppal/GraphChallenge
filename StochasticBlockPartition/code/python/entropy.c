#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API (1)
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

double entropy_row_nz(const int64_t* restrict x, const int64_t* restrict y, long n, int64_t c)
{
  double sum = 0, log_c;
  long i;

  if (c == 0) {
    return 0.0;
  }

  log_c = log(c);

  for (i=0; i<n; i++) {
    if (x[i] > 0 && y[i] > 0) {
      sum += x[i] * (log(x[i]) - log(y[i]) - log_c);
    }
  }
  return sum;
}

double entropy_row_nz_ignore(const int64_t* restrict x, const int64_t* restrict y, long n, int64_t c, const int64_t* restrict x_idx, long r, long s)
{
  double sum = 0, log_c;
  long i;

  if (c == 0) {
    return 0.0;
  }

  log_c = log(c);

  for (i=0; i<n; i++) {
    if (x_idx[i] != r && x_idx[i] != s && x[i] > 0 && y[i] > 0) {
      sum += x[i] * (log(x[i]) - log(y[i]) - log_c);
    }
  }
  return sum;
}

double entropy_dense_row_ignore(const int64_t* restrict x, const int64_t* restrict y, long n, int64_t c, long r, long s)
{
  double sum = 0, log_c;
  long i;

  if (c == 0) {
    return 0.0;
  }

  log_c = log(c);

  for (i=0; i<n; i++) {
    if (i != r && i != s && x[i] > 0 && y[i] > 0) {
      sum += x[i] * (log(x[i]) - log(y[i]) - log_c);
    }
  }
  return sum;
}

static PyObject* module_entropy_row_nz_ignore(PyObject *self, PyObject *args)
{
  PyObject *x_obj, *y_obj, *x_idx_obj;
  double c;
  long r, s;

  if (!PyArg_ParseTuple(args, "OOdOll", &x_obj, &y_obj, &c, &x_idx_obj, &r, &s))
    return NULL;

  PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_LONG, NPY_IN_ARRAY);
  PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_LONG, NPY_IN_ARRAY);
  PyObject *x_idx_array = PyArray_FROM_OTF(x_idx_obj, NPY_LONG, NPY_IN_ARRAY);

  if (x_array == NULL || y_array == NULL || x_idx_array == NULL) {
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(x_idx_array);
    return NULL;
  }

  int N = (int) PyArray_DIM(x_array, 0);
  const int64_t *x = (const int64_t *) PyArray_DATA(x_array);
  const int64_t *y = (const int64_t *) PyArray_DATA(y_array);
  const int64_t *x_idx = (const int64_t *) PyArray_DATA(x_idx_array);

  double val = entropy_row_nz_ignore(x, y, N, c, x_idx, r, s);

  Py_DECREF(x_array);
  Py_DECREF(y_array);
  Py_DECREF(x_idx_array);

  PyObject *ret = Py_BuildValue("d", val);
  return ret;
}

static PyObject* module_entropy_dense_row_ignore(PyObject *self, PyObject *args)
{
  PyObject *x_obj, *y_obj;
  double c;
  long r, s;

  if (!PyArg_ParseTuple(args, "OOdll", &x_obj, &y_obj, &c, &r, &s))
    return NULL;

  PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_LONG, NPY_IN_ARRAY);
  PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_LONG, NPY_IN_ARRAY);

  if (x_array == NULL || y_array == NULL) {
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
  }

  int N = (int) PyArray_DIM(x_array, 0);
  const int64_t *x = (const int64_t *) PyArray_DATA(x_array);
  const int64_t *y = (const int64_t *) PyArray_DATA(y_array);

  double val = entropy_dense_row_ignore(x, y, N, c, r, s);

  Py_DECREF(x_array);
  Py_DECREF(y_array);

  PyObject *ret = Py_BuildValue("d", val);
  return ret;
}

static PyObject* module_entropy_row_nz(PyObject *self, PyObject *args)
{
  PyObject *x_obj, *y_obj;
  double c;

  if (!PyArg_ParseTuple(args, "OOd", &x_obj, &y_obj, &c))
    return NULL;

  PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_LONG, NPY_IN_ARRAY);
  PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_LONG, NPY_IN_ARRAY);

  if (x_array == NULL || y_array == NULL) {
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
  }

  int N = (int) PyArray_DIM(x_array, 0);
  const int64_t *x = (const int64_t *) PyArray_DATA(x_array);
  const int64_t *y = (const int64_t *) PyArray_DATA(y_array);

  double val = entropy_row_nz(x, y, N, c);

  Py_DECREF(x_array);
  Py_DECREF(y_array);

  PyObject *ret = Py_BuildValue("d", val);
  return ret;
}


static PyMethodDef entropy_module_methods[] =
  {
   {
    "entropy_row_nz",
    module_entropy_row_nz,
    METH_VARARGS,
    "Return an entropy row computation."
   },
   {
    "entropy_row_nz_ignore",
    module_entropy_row_nz_ignore,
    METH_VARARGS,
    "Return an entropy row computation, ignoring two indices."
   },
   {
    "entropy_dense_row_ignore",
    module_entropy_dense_row_ignore,
    METH_VARARGS,
    "Return an entropy row computation, ignoring two indices."
   },
   {NULL, NULL, 0, NULL}
  };

static struct PyModuleDef entropy_module_definition =
  {
   PyModuleDef_HEAD_INIT,
   "entropy_module",
   "A Python module that supports natively-compiled entropy computations.",
   -1,
   entropy_module_methods
  };


PyMODINIT_FUNC PyInit_entropy_module(void)
{
  Py_Initialize();
  import_array();
  return PyModule_Create(&entropy_module_definition);
}
