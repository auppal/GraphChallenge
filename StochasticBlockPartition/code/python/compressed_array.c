#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API (1)
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

#define DEBUG_SANITY_COUNTS (0)
#define DEBUG_RESIZE_RACE (0)
#define USE_32_BITS (1)
#define HASH_FLAG_SHARED_MEM (1)

#if DEBUG_RESIZE_RACE
static int debug_resize_race_enabled = 0;
#endif

#if USE_32_BITS
typedef uint32_t hash_key_t;
typedef int32_t hash_val_t;
#define EMPTY_KEY (1U << 31)
#define hash(x) ((((uint64_t) x) ^ ((uint64_t) x) << 11))

#define PRI_HASH_KEY "%u"
#define PRI_HASH_VAL "%d"
#define HASH_IMPL_DESCRIPTION "32 bit"
#else
typedef uint64_t hash_key_t;
typedef int64_t hash_val_t;
#define EMPTY_KEY (1UL << 63)
/* From khash */
#define kh_int64_hash_func(key) (khint32_t)((key)>>33^(key)^(key)<<11)
#define hash(x) (((x) >> 33 ^ (x) ^ (x) << 11))
#define PRI_HASH_KEY "%lu"
#define PRI_HASH_VAL "%ld"
#define HASH_IMPL_DESCRIPTION "64 bit"
#endif

struct hash {
  uint32_t flags;
  uint32_t width;  /* width of each hash table */
  _Atomic(uint32_t) cnt;
  _Atomic(int32_t) internal_refcnt;
};

static inline size_t hash_get_alloc_size(const struct hash *h)
{
  return sizeof(struct hash) + h->width * sizeof(hash_key_t) + h->width * sizeof(hash_val_t);
}

static inline hash_key_t *hash_get_keys(struct hash *h)
{
  return (hash_key_t *) ((uintptr_t) h + sizeof(struct hash));
}

static inline hash_val_t *hash_get_vals(struct hash *h)
{
  return (hash_val_t *) ((uintptr_t) h + sizeof(struct hash) + h->width * sizeof(hash_key_t));
}

static inline size_t hash_get_limit(const struct hash *h)
{
  return h->width * 7 / 10;
}

struct hash_outer {
	long external_refcnt;
	struct hash *h;
};

static void private_free(void *p, size_t len)
{
  free(p);
}

int hash_sanity_count(const char *msg, const struct hash *h)
{
  size_t i, sanity_cnt = 0;

  const hash_key_t *keys = hash_get_keys((struct hash *) h);

  for (i=0; i<h->width; i++) {
    if (keys[i] != EMPTY_KEY) {
      sanity_cnt++;
    }
  }
  if (h->cnt != sanity_cnt) {
    fprintf(stderr, "Sanity count at %s for hash %p failed: cnt %lu h->cnt %u\n", msg, h, sanity_cnt, h->cnt);
    return -1;
  }

  return 0;
}

struct hash *hash_create(size_t initial_size, int shared_mem)
{
  size_t buf_size = sizeof(struct hash) + initial_size * sizeof(hash_key_t) + initial_size * sizeof(hash_val_t);
  void *(*fn_malloc)(size_t);

  if (shared_mem) {
    fn_malloc = shared_malloc;
  }
  else {
    fn_malloc = malloc;
  }

  char *buf = fn_malloc(buf_size);
  if (!buf) {
    fprintf(stderr, "hash_create(%ld): return NULL\n", initial_size);
    return NULL;
  }

  struct hash *h = (struct hash *) buf;
  h->flags = 0;
  if (shared_mem) {
    h->flags |= HASH_FLAG_SHARED_MEM;
  }
  h->width = initial_size;
  h->cnt = 0;
  h->internal_refcnt = 0;

  size_t i;
  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);
  for (i=0; i<h->width; i++) {
    keys[i] = EMPTY_KEY;
    vals[i] = 0;
  }

  return h;
}

int hash_outer_init(struct hash_outer *ho, size_t initial_size)
{
  ho->external_refcnt = 0;
  ho->h = hash_create(initial_size, 1 /* use shared mem */);

  if (!ho->h) {
    return -1;
  }

  return 0;
}

void hash_destroy(struct hash *h)
{
  if (h) {
    size_t alloc_size = hash_get_alloc_size(h);
    if (h->flags & HASH_FLAG_SHARED_MEM) {
      shared_free(h, alloc_size);
    }
    else {
      private_free(h, alloc_size);
    }
  }
}

void hash_outer_destroy(struct hash_outer *ho)
{
  // fprintf(stderr, "hash_outer_destroy %p\n", ho);
  hash_destroy(ho->h);
}

struct hash *hash_copy(const struct hash *y, int shared_mem)
{
  void *(*fn_malloc)(size_t);

  if (shared_mem) {
    fn_malloc = shared_malloc;
  }
  else {
    fn_malloc = malloc;
  }

  size_t buf_size = hash_get_alloc_size(y);
  char *buf = fn_malloc(buf_size);

  if (!buf) {
    fprintf(stderr, "hash_copy: return NULL\n");
    return NULL;
  }

  struct hash *h = (struct hash *) buf;

  h->flags = 0;
  if (shared_mem) {
    h->flags |= HASH_FLAG_SHARED_MEM;
  }

  h->width = y->width;
  h->cnt = y->cnt;
  hash_key_t *y_keys = hash_get_keys((struct hash *) y);
  hash_val_t *y_vals = hash_get_vals((struct hash *) y);
  h->internal_refcnt = 0;

  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);
  memcpy(keys, y_keys, y->width * sizeof(hash_key_t));
  memcpy(vals, y_vals, y->width * sizeof(hash_val_t));
  return h;
}

int hash_outer_copy(struct hash_outer *to, const struct hash_outer *from)
{
  to->external_refcnt = 0;
  to->h = hash_copy(from->h, 1 /* shared mem */);
  return to->h ? 0 : -1;
}

void hash_print(struct hash *h);
int hash_set_single(struct hash *h, hash_key_t k, hash_val_t v);
#define RESIZE_DEBUG (0)

static inline struct hash *hash_resize(struct hash *h)
{
  size_t i;
  struct hash *h2;
  size_t limit = hash_get_limit(h);

  if (h->cnt == limit) {
    /* Resize needed */

#if RESIZE_DEBUG
    fprintf(stderr, "Before resize cnt is %ld\n", h->cnt);
    hash_print(h);
#endif

    h2 = hash_create(h->width * 2, ((h->flags & HASH_FLAG_SHARED_MEM) != 0));
    if (!h2) {
      fprintf(stderr, "hash_resize: hash_create to width %u from %ld failed\n", h->width * 2, limit);
      return NULL;
    }
#if RESIZE_DEBUG
    fprintf(stderr, "insert: ");
#endif

    long ins = 0;
    hash_key_t *keys = hash_get_keys((struct hash *) h);
    hash_val_t *vals = hash_get_vals((struct hash *) h);
    for (i=0; i<h->width; i++) {
      if (keys[i] != EMPTY_KEY) {
	//fprintf(stderr, " %ld ", keys[i]);
	hash_set_single(h2, keys[i], vals[i]);
	ins++;
      }
    }
#if RESIZE_DEBUG
    fprintf(stderr, "\nAfter resize inserted %ld\n", ins);
    hash_print(h2);
    fprintf(stderr, "\n\n");
#endif

    if (h->cnt != h2->cnt) {
      fprintf(stderr, "Mismatch found in hash %p h1->cnt %u h2->cnt %u ins %ld\n", h, h->cnt, h2->cnt, ins);
      return NULL;
    }

    hash_destroy(h);
    h = h2;
  }

  return h;
}

static int hash_resize_needed(const struct hash *h)
{
  /* Return whether a resize is needed. */
  size_t limit = hash_get_limit(h);
  return h->cnt >= limit ? 1 : 0;
}

int hash_set_single(struct hash *h, hash_key_t k, hash_val_t v)
{
  /* To avoid a subtle logic bug, first check for existance.
   * Beacuse not every insertion will cause an increase in cnt.
   */
  size_t i, width = h->width;
  hash_key_t kh = hash(k);
  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
    if (keys[idx] == k) {
      vals[idx] = v;
      break;
    }
    else if (keys[idx] == EMPTY_KEY) {
      keys[idx] = k;
      vals[idx] = v;
      h->cnt++;
      break;
    }
  }

  return hash_resize_needed(h);
}


int hash_accum_single(struct hash *h, hash_key_t k, hash_val_t c)
{
  /* To avoid a subtle logic bug, first check for existance.
   * Beacuse not every insertion will cause an increase in cnt.
   */
  size_t i, width = h->width;
  hash_key_t kh = hash(k);

  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
#if 0
    if (keys[idx] == k) {
      vals[idx] += c;
      break;
    }
    else if (keys[idx] == EMPTY_KEY) {
      keys[idx] = k;
      vals[idx] = c;
      h->cnt++;
      break;
    }
#else
    /* Try experimental lock-free approach */
    hash_key_t empty = EMPTY_KEY;
    _Bool rc = atomic_compare_exchange_strong((_Atomic(hash_key_t) *) &keys[idx], &empty, k);

    if (rc) {
      /* Was empty, and new key inserted.
       * It is safe to add instead of assign because vals were all
       * initialized to zero.
       */
      atomic_fetch_add_explicit((_Atomic(hash_val_t) *) &vals[idx], c, memory_order_relaxed);
      /* And also do increase the count by 1. */
      atomic_fetch_add_explicit(&h->cnt, 1, memory_order_seq_cst);
      break;
    }
    else if (keys[idx] == k) {
      /* Was not empty. Check the existing key. */
      atomic_fetch_add_explicit((_Atomic(hash_val_t) *) &vals[idx], c, memory_order_relaxed);
      break;
    }
#endif
  }

  return hash_resize_needed(h);
}


static inline int hash_search(const struct hash *h, hash_key_t k, hash_val_t *v)
{
  size_t i, width = h->width;
  hash_key_t kh = hash(k);

  const hash_key_t *keys = hash_get_keys((struct hash *) h);
  const hash_val_t *vals = hash_get_vals((struct hash *) h);

  for (i=0; i<width; i++) {
    size_t idx = (kh + i) % width;
    if (keys[idx] == k) {
      *v = vals[idx];
      return 0;
    }
    else if (keys[idx] == EMPTY_KEY) {
      *v = 0; /* Default value */
      return -1;
    }
  }

  return -1;
}

static inline void hash_search_multi(const struct hash *h, const unsigned long *keys, long *vals, size_t n)
{
  size_t i;
  for (i=0; i<n; i++) {
#if USE_32_BITS
    hash_val_t v;
    hash_search(h, keys[i], &v);
    vals[i] = v;
#else
    hash_search(h, keys[i], &vals[i]);
#endif
  }
}

static inline hash_val_t hash_sum(const struct hash *h)
{
  size_t i;
  hash_val_t s = 0;
  const hash_key_t *keys = hash_get_keys((struct hash *) h);
  const hash_val_t *vals = hash_get_vals((struct hash *) h);

  for (i=0; i<h->width; i++) {
    if (keys[i] != EMPTY_KEY) {
      s += vals[i];
    }
  }

  return s;
}


static inline size_t hash_keys(const struct hash *h, unsigned long *keys, size_t max_cnt)
{
  const hash_key_t *h_keys = hash_get_keys((struct hash *) h);
  size_t i, width = h->width, cnt = 0;

  for (i=0; i<width; i++) {
    if (h_keys[i] != EMPTY_KEY) {
      if (cnt == max_cnt) {
	break;
      }
      *keys++ = h_keys[i];
      cnt++;
    }
  }

  return cnt;
}

size_t hash_vals(const struct hash *h, long *vals, size_t max_cnt)
{
  size_t i, width = h->width, cnt = 0;
  hash_key_t *h_keys = hash_get_keys((struct hash *) h);
  hash_val_t *h_vals = hash_get_vals((struct hash *) h);

  for (i=0; i<width; i++) {
    if (h_keys[i] != EMPTY_KEY) {
      if (cnt == max_cnt) {
	break;
      }
      *vals++ = h_vals[i];
      cnt++;
    }
  }

  return cnt;
}

void hash_print(struct hash *h)
{
  size_t i, width = h->width;
  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);

  fprintf(stderr, "Print dict %p with %u items\n", h, h->cnt);
  fprintf(stderr, "{ ");
  for (i=0; i<width; i++) {
    if (keys[i] != EMPTY_KEY) {
      fprintf(stderr, PRI_HASH_KEY ":" PRI_HASH_VAL " ", keys[i], vals[i]);
    }
  }
  fprintf(stderr, "}\n");
}

int hash_eq(const struct hash *x, const struct hash *y)
{
  size_t i;
  hash_key_t *x_keys = hash_get_keys((struct hash *) x);
  hash_val_t *x_vals = hash_get_vals((struct hash *) x);

  for (i=0; i<x->width; i++) {
    if (x_keys[i] != EMPTY_KEY) {
      hash_val_t v2 = 0;
      hash_search(y, x_keys[i], &v2);
      if (v2 != x_vals[i]) {
	fprintf(stderr, "Mismatch at key " PRI_HASH_KEY "\n", x_keys[i]);
	return 0;
      }
    }
  }

  hash_key_t *y_keys = hash_get_keys((struct hash *) y);
  hash_val_t *y_vals = hash_get_vals((struct hash *) y);

  for (i=0; i<y->width; i++) {
    if (y_keys[i] != EMPTY_KEY) {
      hash_val_t v = 0;
      hash_search(x, y_keys[i], &v);
      if (v != y_vals[i]) {
	fprintf(stderr, "Mismatch at key " PRI_HASH_KEY "\n", y_keys[i]);
	return 0;
      }
    }
  }

  return 1; /* 1 if equal */
}

#if 0
/* Unused */
static inline void hash_accum_constant(const struct hash *h, size_t C)
{
  size_t i, width = h->width;
  hash_key_t *keys = hash_get_keys((struct hash *) h);
  hash_val_t *vals = hash_get_vals((struct hash *) h);

  for (i=0; i<width; i++) {
    if (keys[i] != EMPTY_KEY) {
      vals[i] += C;
    }
  }
}
#endif

/*
 * It is not strictly correct to use int64_t instead of hash_key_t.
 * But it is done here for expedience.
 */
static inline struct hash *hash_accum_multi(struct hash *h, const unsigned long *keys, const long *vals, size_t n_keys, int scale)
{
  size_t j, i;

#if 0
  fprintf(stderr, "Before accum_multi\n");
  hash_print(h);
#endif

  for (j=0; j<n_keys; j++) {
    hash_key_t *h_keys = hash_get_keys(h);
    hash_val_t *h_vals = hash_get_vals(h);

    hash_key_t kh = hash(keys[j]);
#if 0
    fprintf(stderr, " Insert %ld +%ld\n", keys[j], vals[j]);
#endif
    for (i=0; i<h->width; i++) {
      size_t idx = (kh + i) % h->width;
      if (h_keys[idx] == keys[j]) {
	h_vals[idx] += scale * vals[j];
	break;
      }
      else if (h_keys[idx] == EMPTY_KEY) {
	/* Not found assume the previous default value of zero and set a new entry. */
	h_keys[idx] = keys[j];
	h_vals[idx] = scale * vals[j];
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

#if 0
  int flag = 0;
  size_t limit = hash_get_limit(h);
  if (h->cnt == limit) {
    fprintf(stderr, "Resize on accum before %u %ld\n", h->cnt, limit);
    flag = 1;
  }

  if (flag) {
    fprintf(stderr, "Resize on accum after %u %ld\n", h->cnt, limit);
  }
#endif

  return h;
}

struct compressed_array {
  size_t n_row, n_col;
  struct hash_outer *rows;
  struct hash_outer *cols;
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

  x->rows = shared_calloc(x->n_row, sizeof(x->rows[0]));

  if (!x->rows) {
    goto bad;
  }

  x->cols = shared_calloc(x->n_col, sizeof(x->cols[0]));

  if (!x->cols) {
    goto bad;
  }
  
  for (i=0; i<n_nodes; i++) {
    int rc1 = hash_outer_init(&x->rows[i], initial_width);
    int rc2 = hash_outer_init(&x->cols[i], initial_width);

    if (rc1 || rc2) {
      fprintf(stderr, "compressed_array_create: hash_create failed\n");
      break;
    }
  }

  if (i != n_nodes) {
    for (; i>=0; i--) {
      hash_outer_destroy(&x->rows[i]);
      hash_outer_destroy(&x->cols[i]);
    }
    goto bad;
  }

  return x;
  
bad:
  if (x) {
    shared_free(x->rows, x->n_row * sizeof(x->rows[0]));
    shared_free(x->cols, x->n_col * sizeof(x->cols[0]));
    free(x);
  }
  fprintf(stderr, "compressed_array_create return NULL\n");
  return NULL;
}

void compressed_array_destroy(struct compressed_array *x)
{
  size_t i;

  //fprintf(stderr, "Destroy %p\n", x);

  for (i=0; i<x->n_col; i++) {
    //fprintf(stderr, "Destroy hash %p\n", x->cols[i]);
    hash_outer_destroy(&x->cols[i]);
  }

  for (i=0; i<x->n_row; i++) {
    hash_outer_destroy(&x->rows[i]);
  }

  shared_free(x->rows, x->n_row * sizeof(x->rows[0]));
  shared_free(x->cols, x->n_col * sizeof(x->cols[0]));
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

  x->rows = shared_calloc(x->n_row, sizeof(x->rows[0]));

  if (!x->rows) {
    free(x);
    return NULL;
  }

  for (i=0; i<x->n_row; i++) {
    int rc = hash_outer_copy(&x->rows[i], &y->rows[i]);

    if (rc < 0) {
      do { hash_outer_destroy(&x->rows[i]); } while (i-- != 0);
      shared_free(x->rows, x->n_row * sizeof(x->rows[0]));
      return NULL;
    }
  }

  x->cols = shared_calloc(x->n_col, sizeof(x->cols[0]));

  if (!x->cols) {
    for (i=0; i<x->n_row; i++) {
      hash_outer_destroy(&x->rows[i]);
    }
    free(x);
    return NULL;
  }

  for (i=0; i<x->n_col; i++) {
    int rc = hash_outer_copy(&x->cols[i], &y->cols[i]);

    if (rc < 0) {
      do { hash_outer_destroy(&x->cols[i]); } while (i-- != 0);
      for (i=0; i<x->n_row; i++) {
	hash_outer_destroy(&x->rows[i]);
      }
      shared_free(x->rows, x->n_row * sizeof(x->rows[0]));
      shared_free(x->cols, x->n_col * sizeof(x->cols[0]));
      return NULL;
    }
  }

  return x;
}

static inline int compressed_get_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t *val)
{
  /* Just get from row[i][j] */
#if USE_32_BITS
  hash_val_t v;
  int rc = hash_search(x->rows[i].h, j, &v);
  *val = v;
  return rc;
#else
  return hash_search(x->rows[i].h, j, val);
#endif
}


/* Unlike compressed_accum_single, this is NOT safe for lock-free use. */
static inline void compressed_set_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t val)
{
  hash_set_single(x->rows[i].h, j, val);
  hash_set_single(x->cols[j].h, i, val);

  if (hash_resize_needed(x->rows[i].h)) {
    x->rows[i].h = hash_resize(x->rows[i].h);

  }

  if (hash_resize_needed(x->cols[j].h)) {
    x->cols[j].h = hash_resize(x->cols[j].h);
  }
}

int hash_accum_resize(struct hash_outer *ho, hash_key_t k, hash_val_t C)
{
  struct hash *oldh, *newh;

  struct hash_outer hoa_cur;
  struct hash_outer hoa_new, cur;

  /* Atomically both grab the pointer and increment the counter. */
  do {
    hoa_cur = *ho;
    hoa_new = hoa_cur;
    hoa_new.external_refcnt++;
  } while(!atomic_compare_exchange_strong((_Atomic(struct hash_outer) *) ho, &hoa_cur, hoa_new));

  cur = hoa_cur;
  oldh = cur.h;

#if DEBUG_RESIZE_RACE
  if (debug_resize_race_enabled) {
    fprintf(stderr, "Outer %p inner %p %ld external_refcnt was %ld now %ld\n", ho, ho->h, ho->h->internal_refcnt, cur.external_refcnt, hoa_new.external_refcnt);
  }
#endif

  if (1 == hash_accum_single(oldh, k, C)) {
    newh = hash_create(oldh->width * 2, 1);

    if (!newh) {
      fprintf(stderr, "hash_accum_resize: hash_create failed\n");
      return -1;
    }

    _Bool rc = false;
    hoa_cur = *ho; /* Re-read because external_refcnt may have changed. */
    cur = hoa_cur;
    if (cur.h == oldh) {
      hoa_new.h = newh;
      hoa_new.external_refcnt = 0;

#if DEBUG_RESIZE_RACE
      if (debug_resize_race_enabled) {
	fprintf(stderr, "Pid %d before CAS outer %p inner %p external_refcnt %ld (oldh %p)\n", getpid(), ho, cur.h, cur.external_refcnt, oldh);
      }
#endif
      rc = atomic_compare_exchange_strong((_Atomic(struct hash_outer) *) ho, &hoa_cur, hoa_new);

      cur = hoa_cur;

#if DEBUG_RESIZE_RACE
      if (debug_resize_race_enabled) {
	fprintf(stderr, "Pid %d after CAS outer %p inner %p external_refcnt %ld (rc %d oldh %p)\n", getpid(), ho, cur.h, cur.external_refcnt, rc, oldh);
      }
#endif
    }

    if (!rc) {
      /* Someone else won the race */
#if DEBUG_RESIZE_RACE
      if (debug_resize_race_enabled) {
	fprintf(stderr, "Pid %d Someone else won the race for %p.\n", getpid(), ho);
      }
#endif
      hash_destroy(newh);
    }
    else {
      /* We won the race. */

      atomic_fetch_sub_explicit(&oldh->internal_refcnt,
				cur.external_refcnt - 1,
				memory_order_seq_cst);
#if DEBUG_RESIZE_RACE
      if (debug_resize_race_enabled) {
	fprintf(stderr, "Pid %d We won the race (rc %d) for %p ! Subtract %ld from oldh %p (refcnt %ld) and wait.\n", getpid(), rc, ho, cur.external_refcnt, oldh, oldh->internal_refcnt);
      }
#endif

      /* Wait for other writers to finish. Minus 1 because WE are
       * still using it.
       */
      do {

#if DEBUG_RESIZE_RACE
	if (debug_resize_race_enabled) {
	  fprintf(stderr, "Pid %d Outer %p Wait for oldh %p oldh->internal_refcnt = %ld\n", getpid(), ho, oldh, oldh->internal_refcnt);
	  usleep(100000);
	}
#endif
      } while (oldh->internal_refcnt < 0);

      atomic_thread_fence(memory_order_acquire);

#if DEBUG_RESIZE_RACE
      if (debug_resize_race_enabled) {
	fprintf(stderr, "Pid %d Outer %p Done waiting for %p ! Merge %ld items into hash %p\n", getpid(), ho, oldh, oldh->cnt, newh);
      }
#endif
      // long ins = 0;
      size_t ii;


      hash_key_t *keys = hash_get_keys(oldh);
      hash_val_t *vals = hash_get_vals(oldh);

      for (ii=0; ii<oldh->width; ii++) {
	hash_key_t k = keys[ii];
	hash_val_t v = vals[ii];
	if (k != EMPTY_KEY) {
	  if (1 == hash_accum_single(newh, k, v)) {
	    fprintf(stderr, "Pid %d Error: Needed ANOTHER resize while resizing!\n", getpid());
	    return -1;
	  }
	  // ins++;
	}
      }

      hash_destroy(oldh);
      return 0;
    }
  }

#if DEBUG_RESIZE_RACE
  if (debug_resize_race_enabled) {
    fprintf(stderr, "Release oldh %p %ld\n", oldh, oldh->internal_refcnt);
  }
#endif

  atomic_thread_fence(memory_order_release);
  oldh->internal_refcnt++;

  return 0;
}

static inline int compressed_accum_single(struct compressed_array *x, uint64_t i, uint64_t j, int64_t C)
{
  if (hash_accum_resize(&x->rows[i], j, C)) { return -1; }
  if (hash_accum_resize(&x->cols[j], i, C)) { return -1; }

  return 0;
}

/* Take values along a particular axis */
int compressed_take_keys_values(struct compressed_array *x, long idx, long axis, unsigned long **p_keys, long **p_vals, long *p_cnt)
{
  size_t cnt;
  struct hash *h = (axis == 0 ? x->rows[idx].h : x->cols[idx].h);

  cnt = h->cnt;

  unsigned long *keys;
  long *vals;

  if (cnt == 0) {
    *p_keys = NULL;
    *p_vals = NULL;
    *p_cnt = 0;
    return 0;
  }
  else {
    keys = malloc(cnt * sizeof(keys[0]));
    vals = malloc(cnt * sizeof(vals[0]));
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
  return (axis == 0 ? x->rows[idx].h : x->cols[idx].h);
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

static inline struct hash **create_dict(struct hash *p)
{
  struct hash **ph = malloc(sizeof(struct hash **));
  if (ph) {
    *ph = p;
  }
  return ph;
}

static void destroy_dict_copy(PyObject *obj)
{
  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");
  if (ph) {
    hash_destroy(*ph);
    free(ph);
  }
}

static void destroy_dict_ref(PyObject *obj)
{
  struct hash **ph = PyCapsule_GetPointer(obj, "compressed_array_dict");
  if (ph) {
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
  unsigned long *keys = malloc(cnt * sizeof(keys[0]));
  long *vals = malloc(cnt * sizeof(vals[0]));

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

  const long *keys = (const long *) PyArray_DATA(obj_k);
  const long *vals = (const long *) PyArray_DATA(obj_v);
  long N = (long) PyArray_DIM(obj_k, 0);

  h = hash_accum_multi(h, (unsigned long *) keys, vals, N, 1);

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
  
  ret = PyCapsule_New(ph, "compressed_array_dict", destroy_dict_copy);
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
  struct hash *orig = compressed_take(x, idx, axis);
  //hash_print(orig);
  struct hash *ent = hash_copy(orig, 0);

  if (!ent) {
    PyErr_SetString(PyExc_RuntimeError, "take_dict: hash_copy failed");
    return NULL;
  }

#if DEBUG_SANITY_COUNTS  
  int ok1 = hash_sanity_count("take_dict orig", orig);

  if (ok1 < 0) {
    char *msg;
    asprintf(&msg, "take_dict orig failed at idx %ld axis %ld\n", idx, axis);
    fputs(msg, stderr);
    PyErr_SetString(PyExc_RuntimeError, msg);
    free(msg);
    return NULL;    
  }
  
  int ok2 = hash_sanity_count("take_dict ent", ent);
  
  if (ok1 != ok2) {
    fprintf(stderr, "take_dict found different sanity for orig (cnt %ld) and copy ent (cnt %ld). Major weirdness!\n", orig->cnt, ent->cnt);
  }
#endif
  
  struct hash **ph = create_dict(ent);
  ret = PyCapsule_New(ph, "compressed_array_dict", destroy_dict_copy);
  return ret;
}

static PyObject* take_dict_ref(PyObject *self, PyObject *args)
{
  PyObject *obj, *ret;
  long idx, axis;

  if (!PyArg_ParseTuple(args, "Oll", &obj, &idx, &axis))
    return NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  /* Returns a reference. */
  struct hash *ent = compressed_take(x, idx, axis);

  if (!ent) {
    PyErr_SetString(PyExc_RuntimeError, "take_dict: hash_copy failed");
    return NULL;
  }
  
  struct hash **ph = create_dict(ent);
  ret = PyCapsule_New(ph, "compressed_array_dict", destroy_dict_ref);
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

  if (axis == 0) {
    hash_outer_destroy(&x->rows[idx]);
    x->rows[idx].h = hash_copy(ent, 1);
  }
  else {
    hash_outer_destroy(&x->cols[idx]);
    x->cols[idx].h = hash_copy(ent, 1);
  }

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

#if DEBUG_SANITY_COUNTS
  hash_sanity_count("copy_dict h", h);
  hash_sanity_count("copy_dict h2", h2);
#endif  
  
  struct hash **ph2 = create_dict(h2);

  ret = PyCapsule_New(ph2, "compressed_array_dict", destroy_dict_copy);
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

  hash_val_t val = hash_sum(*ph);
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
    hash_val_t val = 0;
    hash_search(h, k_int, &val);
    PyObject *ret = Py_BuildValue("k", val);
    return ret;
  }

  PyErr_Restore(NULL, NULL, NULL); /* clear the exception */  

  obj_k = PyArray_FROM_OTF(obj_k, NPY_LONG, NPY_IN_ARRAY);
  const long *keys = (const long *) PyArray_DATA(obj_k);
  long N = (long) PyArray_DIM(obj_k, 0);

  long *vals = malloc(N * sizeof(vals[0]));

  hash_search_multi(h, (const unsigned long *) keys, vals, N);

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
  hash_key_t *keys = hash_get_keys(h);
  hash_val_t *vals = hash_get_vals(h);  

  for (j=0; j<h->width; j++) {
    if (keys[j] != EMPTY_KEY) {
      if (axis == 0) {
	compressed_set_single(x, i, keys[j], vals[j]);
      }
      else {
	compressed_set_single(x, keys[j], i, vals[j]);
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

static PyObject* nonzero_count(PyObject *self, PyObject *args)
{
  PyObject *obj;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  if (!x) {
    PyErr_SetString(PyExc_RuntimeError, "Bad pointer to compresed array");
    return NULL;
  }

  size_t i;
  long count = 0;
  for (i=0; i<x->n_row; i++) {
    count += x->rows[i].h->cnt;
  }

  PyObject *ret = Py_BuildValue("l", count);
  return ret;
}

static PyObject* sanity_check(PyObject *self, PyObject *args)
{
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
    if (x->rows[i].external_refcnt != x->rows[i].h->internal_refcnt) {
      PyErr_SetString(PyExc_RuntimeError, "external_cnt != internal_refcnt at row");
      return NULL;
    }
  }

  for (i=0; i<x->n_col; i++) {
    if (x->cols[i].external_refcnt != x->cols[i].h->internal_refcnt) {
      PyErr_SetString(PyExc_RuntimeError, "external_cnt != internal_refcnt at col");
      return NULL;
    }
  }

  for (i=0; i<x->n_row; i++) {
    if (!x->rows[i].h) {
      PyErr_SetString(PyExc_RuntimeError, "Invalid rows found");
      return NULL;
    }
  }

  for (i=0; i<x->n_col; i++) {
    if (!x->cols[i].h) {
      PyErr_SetString(PyExc_RuntimeError, "Invalid cols found");
      return NULL;
    }
  }

  for (i=0; i<x->n_row; i++) {
    if (hash_sanity_count("sanity_check", x->rows[i].h) < 0) {
      char *msg;
      if (asprintf(&msg, "Invalid row count found at position %ld\n", i) > 0) {
	PyErr_SetString(PyExc_RuntimeError, msg);
      }
      return NULL;
    }
  }

  for (i=0; i<x->n_col; i++) {
    if (hash_sanity_count("sanity_check", x->cols[i].h) < 0) {
      char *msg;
      if (asprintf(&msg, "Invalid col count found at position %ld\n", i) > 0) {
	PyErr_SetString(PyExc_RuntimeError, msg);
      }
      return NULL;
    }
  }

  for (i=0; i<x->n_row; i++) {
    for (j=0; j<x->rows[i].h->width; j++) {
      hash_key_t *keys = hash_get_keys(x->rows[i].h);
      if (keys[j] != EMPTY_KEY) {
	if (keys[j] > 999999) {
	  char *msg;
	  if (asprintf(&msg, "Invalid key value "PRI_HASH_KEY" found in hash %p", keys[j], x->rows[i].h) > 0) {
	    PyErr_SetString(PyExc_RuntimeError, msg);
	  }
	  return NULL;
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
  double sum = 0.0;
  float log_c;
  size_t i;

  if (c == 0) {
    return 0.0;
  }

  log_c = logf(c);

  /* Iterate over keys and values */
  const hash_key_t *restrict keys = hash_get_keys(h);
  const hash_val_t *restrict vals = hash_get_vals(h);

  for (i=0; i<h->width; i++) {
    if (keys[i] != EMPTY_KEY) {
      int64_t xi = vals[i];
      int64_t yi = deg[keys[i]];
      if (xi > 0 && yi > 0) {
	sum += xi * (logf((float) xi / yi) - log_c);
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
  double sum = 0.0;
  float log_c;
  size_t i;

  if (c == 0) {
    return 0.0;
  }

  log_c = logf(c);

  const hash_key_t *restrict keys = hash_get_keys(h);
  const hash_val_t *restrict vals = hash_get_vals(h);

  /* Iterate over keys and values */
  for (i=0; i<h->width; i++) {
    if (keys[i] != EMPTY_KEY && keys[i] != r && keys[i] != s) {
      hash_val_t xi = vals[i];
      int64_t yi = deg[keys[i]];
      if (xi > 0 && yi > 0) {
	sum += xi * (logf((float) xi / yi) - log_c);
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
static PyObject* inplace_apply_movement_compressed_interblock_matrix(PyObject *self, PyObject *args)
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



static int blocks_and_counts(const long *partition, const long *neighbors, const long *weights, long n, long **p_blocks, long **p_counts, long *p_n_blocks, struct hash **p_h)
{
  /* Reserve at least 2x elements, to avoid resizes due to 0.70 max
   * load factor. But reserve at least 16.
   */
  long i;
  size_t initial_width = (2 * n > 16) ? 2 * n : 16;
  struct hash *h = hash_create(initial_width, 0);
  // size_t resize_cnt = 0;

  for (i=0; i<n; i++) {
    uint64_t k = partition[neighbors[i]];
    uint64_t w = weights[i];
    if (hash_accum_single(h, k, w) == 1) {
      // fprintf(stderr, "PID: %d blocks_and_counts of %ld neighbors needed resizing (cnt %ld) from width %u to width %u\n", getpid(), n, resize_cnt, h->width, 2 * h->width);
      h = hash_resize(h);
      if (!h) {
	return -1;
      }
    }
  }

  long *blocks = malloc(n * sizeof(long));
  long *counts = malloc(n * sizeof(long));

  long cnt;
  cnt = hash_keys(h, (unsigned long *) blocks, n);
  hash_vals(h, counts, n);

  if (p_h) {
    *p_h = h;
  }
  else {
    hash_destroy(h);
  }

  *p_blocks = blocks;
  *p_counts = counts;
  *p_n_blocks = cnt;
  return 0;
}
  

/*
 * Args: partition, neighbor nodes, weights
 */
static PyObject* blocks_and_counts_py(PyObject *self, PyObject *args)
{
  PyObject *obj_partition, *obj_neighbors, *obj_weights;

  if (!PyArg_ParseTuple(args, "OOO", &obj_partition, &obj_neighbors, &obj_weights)) {
    return NULL;
  }

  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_weights = PyArray_FROM_OTF(obj_weights, NPY_LONG, NPY_IN_ARRAY);

  const long *partition = (const long *) PyArray_DATA(ar_partition);
  const long *neighbors = (const long *) PyArray_DATA(ar_neighbors);
  const long *weights = (const long *) PyArray_DATA(ar_weights);

  long n = (long) PyArray_DIM(ar_neighbors, 0);

  long *blocks;
  long *counts, n_block;

  PyObject *ret = NULL;

  if (blocks_and_counts(partition, neighbors, weights, n, &blocks, &counts, &n_block, NULL) < 0) {
    	PyErr_SetString(PyExc_RuntimeError, "hash_resize failed");
  }
  else {
    npy_intp dims[] = {n_block};
    PyObject *blocks_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, blocks);
    PyObject *counts_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, counts);

    PyArray_ENABLEFLAGS((PyArrayObject*) blocks_obj, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) counts_obj, NPY_ARRAY_OWNDATA);

    ret = Py_BuildValue("NN", blocks_obj, counts_obj);
  }

  Py_DECREF(ar_partition);
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_weights);

  return ret;
}

/* Given two sets of key-alue pairs, combine them and return the
 * resulting key-value pairs.
 */
static inline int combine_key_value_pairs(const long *k0, const long *v0, long n0, const long *k1, const long *v1, long n1, long **pk2, long **pv2, long *pn2)
{
  /* Reserve at least 2x elements, to avoid resizes due to 0.70 max
   * load factor. But reserve at least 16.
   */
  long i;
  size_t initial_width = (2 * (n0 + n1) > 16) ? 2 * (n0 + n1) : 16;

  struct hash *h = hash_create(initial_width, 0);

  for (i=0; i<n0; i++) {
    if (hash_accum_single(h, k0[i], v0[i]) == 1) {
      h = hash_resize(h);
      return -1;
    }
  }

  for (i=0; i<n1; i++) {
    if (hash_accum_single(h, k1[i], v1[i]) == 1) {
      h = hash_resize(h);
      if (!h) {
	return -1;
      }
    }
  }

  long cnt = h->cnt;
  long *k2 = malloc(cnt * sizeof(long));
  long *v2 = malloc(cnt * sizeof(long));

  hash_keys(h, (unsigned long *) k2, cnt);
  hash_vals(h, v2, cnt);
  hash_destroy(h);

  *pk2 = k2;
  *pv2 = v2;
  *pn2 = cnt;

  return 0;
}

static PyObject* combine_key_value_pairs_py(PyObject *self, PyObject *args)
{
  PyObject *obj_k0, *obj_v0, *obj_k1, *obj_v1;

  if (!PyArg_ParseTuple(args, "OOOO", &obj_k0, &obj_v0, &obj_k1, &obj_v1)) {
    return NULL;
  }

  const PyObject *ar_k0 = PyArray_FROM_OTF(obj_k0, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_v0 = PyArray_FROM_OTF(obj_v0, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_k1 = PyArray_FROM_OTF(obj_k1, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_v1 = PyArray_FROM_OTF(obj_v1, NPY_LONG, NPY_IN_ARRAY);  

  const long *k0 = (const long *) PyArray_DATA(ar_k0);
  const long *v0 = (const long *) PyArray_DATA(ar_v0);
  const long *k1 = (const long *) PyArray_DATA(ar_k1);
  const long *v1 = (const long *) PyArray_DATA(ar_v1);  

  long n0 = (long) PyArray_DIM(ar_k0, 0);
  long n1 = (long) PyArray_DIM(ar_k1, 0);

  if (n0 != (long) PyArray_DIM(ar_v0, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Key-Value pair 0 dimension mismatch");
    return NULL;
  }

  if (n1 != (long) PyArray_DIM(ar_v1, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Key-Value pair 1 dimension mismatch");
    return NULL;
  }

  long *k2, *v2, n2;
  
  if (combine_key_value_pairs(k0, v0, n0, k1, v1, n1, &k2, &v2, &n2) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "combine_key_value_pairs_inner failed");
    return NULL;
  }

  Py_DECREF(ar_k0);
  Py_DECREF(ar_v0);
  Py_DECREF(ar_k1);
  Py_DECREF(ar_v1);  
  
  npy_intp dims[] = {n2};
  PyObject *keys_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, k2);
  PyObject *vals_obj = PyArray_SimpleNewFromData(1, dims, NPY_LONG, v2);

  PyArray_ENABLEFLAGS((PyArrayObject*) keys_obj, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*) vals_obj, NPY_ARRAY_OWNDATA);

  PyObject *ret = Py_BuildValue("NN", keys_obj, vals_obj);
  return ret;
}


static inline int hastings_correction(const long *b_out, const long *count_out, long n_out,
				      const long *b_in, const long *count_in, long n_in,
				      const void *p_cur_M_s_row,
				      const void *p_cur_M_s_col,
				      const void *p_new_M_r_row,
				      const void *p_new_M_r_col,
				      long B,
				      const long *d,
				      long r,
				      long s,
				      long d_new_r,
				      long d_new_s,
				      int is_hash,
				      double *prob)
{
  int rc = -1;
  long i;
  long *t, *count, n_t;
  combine_key_value_pairs(b_out, count_out, n_out,
			  b_in, count_in, n_in,
			  &t, &count, &n_t);

  long *M_t_s = NULL, *M_s_t = NULL, *M_r_row_t = NULL, *M_r_col_t = NULL;

  if (!(M_t_s = malloc(n_t * sizeof(M_t_s[0])))) { goto exit; }
  if (!(M_s_t = malloc(n_t * sizeof(M_s_t[0])))) { goto exit; }

  if (is_hash) {
    const struct hash *cur_M_s_row = (const struct hash *) p_cur_M_s_row;
    const struct hash *cur_M_s_col = (const struct hash *) p_cur_M_s_col;
    hash_search_multi(cur_M_s_col, (const unsigned long *) t, M_t_s, n_t);
    hash_search_multi(cur_M_s_row, (const unsigned long *) t, M_s_t, n_t);
  }
  else {
    const long *cur_M_s_row = (const long *) p_cur_M_s_row;
    const long *cur_M_s_col = (const long *) p_cur_M_s_col;    
    for (i=0; i<n_t; i++) {
      M_t_s[i] = cur_M_s_col[t[i]];
      M_s_t[i] = cur_M_s_row[t[i]];
    }
  }

  double prob_fwd = 0.0, prob_back = 0.0;

  for (i=0; i<n_t; i++) {
    prob_fwd += (double) count[i] * (M_t_s[i] + M_s_t[i] + 1) / (d[t[i]] + B);
  }

  if (!(M_r_row_t = malloc(n_t * sizeof(M_r_row_t[0])))) { goto exit; }
  if (!(M_r_col_t = malloc(n_t * sizeof(M_r_col_t[0])))) { goto exit; }

  if (is_hash) {
    const struct hash *new_M_r_row = (const struct hash *) p_new_M_r_row;
    const struct hash *new_M_r_col = (const struct hash *) p_new_M_r_col;
    hash_search_multi(new_M_r_row, (const unsigned long *) t, M_r_row_t, n_t);
    hash_search_multi(new_M_r_col, (const unsigned long *) t, M_r_col_t, n_t);
  }
  else {
    const long *new_M_r_row = (const long *) p_new_M_r_row;
    const long *new_M_r_col = (const long *) p_new_M_r_col;
    for (i=0; i<n_t; i++) {
      M_r_row_t[i] = new_M_r_row[t[i]];
      M_r_col_t[i] = new_M_r_col[t[i]];
    }
  }
  
  for (i=0; i<n_t; i++) {
    long d_new_ti;
    if (t[i] == r) {
      d_new_ti = d_new_r;
    }
    else if (t[i] == s) {
      d_new_ti = d_new_s;
    }
    else {
      d_new_ti = d[t[i]];
    }
    
    double c = (double) count[i] / (d_new_ti + B);
    prob_back += c * M_r_row_t[i];
    prob_back += c * (M_r_col_t[i] + 1);
  }

  *prob = prob_back / prob_fwd;
  rc = 0;

exit:
  free(M_t_s);
  free(M_s_t);
  free(M_r_row_t);
  free(M_r_col_t);
  return rc;
}

static PyObject* hastings_correction_py(PyObject *self, PyObject *args)
{
  PyObject *obj_b_out, *obj_count_out, *obj_b_in, *obj_count_in, *obj_cur_M_s_row, *obj_cur_M_s_col, *obj_M_r_row, *obj_M_r_col, *obj_d, *obj_d_new;
  long B;
  long r = 0, s = 0;

  if (!PyArg_ParseTuple(args, "OOOOOOOOlOOll",
			&obj_b_out, &obj_count_out, &obj_b_in, &obj_count_in,
			&obj_cur_M_s_row, &obj_cur_M_s_col, &obj_M_r_row, &obj_M_r_col,
			&B, &obj_d, &obj_d_new, &r, &s))
    {
      return NULL;
    }

  const PyObject *ar_b_out = PyArray_FROM_OTF(obj_b_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_out = PyArray_FROM_OTF(obj_count_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_b_in = PyArray_FROM_OTF(obj_b_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_in = PyArray_FROM_OTF(obj_count_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d = PyArray_FROM_OTF(obj_d, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_new = PyArray_FROM_OTF(obj_d_new, NPY_LONG, NPY_IN_ARRAY);  

  long n_out = (long) PyArray_DIM(ar_b_out, 0);
  long n_in = (long) PyArray_DIM(ar_b_in, 0);

  if (n_out != (long) PyArray_DIM(ar_count_out, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Key-Value pair b_out dimension mismatch");
    return NULL;
  }

  if (n_in != (long) PyArray_DIM(ar_count_in, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Key-Value pair b_in dimension mismatch");
    return NULL;
  }

  if (B != (long) PyArray_DIM(ar_d, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Array d dimension mismatch");
    return NULL;
  }

  if (B != (long) PyArray_DIM(ar_d_new, 0)) {
    PyErr_SetString(PyExc_RuntimeError, "Array d dimension mismatch");
    return NULL;
  }

  const long *b_out = (const long *) PyArray_DATA(ar_b_out);
  const long *count_out = (const long *) PyArray_DATA(ar_count_out);
  const long *b_in = (const long *) PyArray_DATA(ar_b_in);
  const long *count_in = (const long *) PyArray_DATA(ar_count_in);
  const long *d = (const long *) PyArray_DATA(ar_d);
  const long *d_new = (const long *) PyArray_DATA(ar_d_new);

  int rc;
  double prob;
  
  if (PyCapsule_GetPointer(obj_cur_M_s_row, "compressed_array_dict")) {
    struct hash *cur_M_s_row = *((struct hash **) PyCapsule_GetPointer(obj_cur_M_s_row, "compressed_array_dict"));
    struct hash *cur_M_s_col = *((struct hash **) PyCapsule_GetPointer(obj_cur_M_s_col, "compressed_array_dict"));
    struct hash *M_r_row = *((struct hash **) PyCapsule_GetPointer(obj_M_r_row, "compressed_array_dict"));
    struct hash *M_r_col = *((struct hash **) PyCapsule_GetPointer(obj_M_r_col, "compressed_array_dict"));

    rc = hastings_correction(
      b_out, count_out, n_out,
      b_in, count_in, n_in,
      cur_M_s_row,
      cur_M_s_col,
      M_r_row,
      M_r_col,
      B,
      d,
      r,
      s,
      d_new[r],
      d_new[s],
      1,
      &prob);
  }
  else {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */
    const PyObject *ar_cur_M_s_row = PyArray_FROM_OTF(obj_cur_M_s_row, NPY_LONG, NPY_IN_ARRAY);
    const PyObject *ar_cur_M_s_col = PyArray_FROM_OTF(obj_cur_M_s_col, NPY_LONG, NPY_IN_ARRAY);
    const PyObject *ar_M_r_row = PyArray_FROM_OTF(obj_M_r_row, NPY_LONG, NPY_IN_ARRAY);
    const PyObject *ar_M_r_col = PyArray_FROM_OTF(obj_M_r_col, NPY_LONG, NPY_IN_ARRAY);

    const long *cur_M_s_row = (const long *) PyArray_DATA(ar_cur_M_s_row);
    const long *cur_M_s_col = (const long *) PyArray_DATA(ar_cur_M_s_col);
    const long *M_r_row = (const long *) PyArray_DATA(ar_M_r_row);
    const long *M_r_col = (const long *) PyArray_DATA(ar_M_r_col);

    rc = hastings_correction(
      b_out, count_out, n_out,
      b_in, count_in, n_in,
      cur_M_s_row,
      cur_M_s_col,
      M_r_row,
      M_r_col,
      B,
      d,
      r,
      s,
      d_new[r],
      d_new[s],
      0,
      &prob);

    Py_DECREF(ar_cur_M_s_row);
    Py_DECREF(ar_cur_M_s_col);    
    Py_DECREF(ar_M_r_row);
    Py_DECREF(ar_M_r_col);    
  }

  if (rc < 0) {
    PyErr_SetString(PyExc_RuntimeError, "hastings_correction failed");
    return NULL;    
  }      

  Py_DECREF(ar_b_out);
  Py_DECREF(ar_count_out);
  Py_DECREF(ar_b_in);
  Py_DECREF(ar_count_in);

  PyObject *ret = Py_BuildValue("d", prob);
  return ret;
}

/*
 * Args: M, r, s, b_out, count_out, b_in, count_in
 * Version for an uncompressed array, using atomic operations.
 */
static PyObject* inplace_apply_movement_uncompressed_interblock_matrix(PyObject *self, PyObject *args)
{
  PyObject *obj_M, *obj_b_out, *obj_count_out, *obj_b_in, *obj_count_in, *obj_d_out, *obj_d_in, *obj_d;
  long r, s;

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

  const long *b_out = (const long *) PyArray_DATA(ar_b_out);
  const long *count_out = (const long *) PyArray_DATA(ar_count_out);
  const long *b_in = (const long *) PyArray_DATA(ar_b_in);
  const long *count_in = (const long *) PyArray_DATA(ar_count_in);
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

static PyObject *take_view(PyObject *ar_M, long idx, long axis)
{
  /* Assumes M is square of type long */
  npy_intp N = PyArray_DIM(ar_M, 0);
  npy_intp size = PyArray_ITEMSIZE(ar_M);
  npy_intp dims[] = {N};

  npy_intp strides[] = {};
  void *data;
  int flags = 0; /* Maybe PyArray_FLAGS(ar_M) is better. */

  if (axis == 0) {
    data = PyArray_GETPTR2(ar_M, idx, 0);
    strides[0] = size;
  }
  else {
    data = PyArray_GETPTR2(ar_M, 0, idx);
    strides[0] = N * size;
  }

  PyArray_Descr *desc = PyArray_DescrNewFromType(PyArray_TYPE(ar_M));
  PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, desc, 1, dims, strides, data, flags, ar_M);
  return ret;
}

#include <sys/random.h>

#if 1
/* Ok if seeded from pool wrappers */
static inline double random_uniform()
{ return drand48(); }

static PyObject *seed(PyObject *self, PyObject *args)
{
  long seed;

  if (!PyArg_ParseTuple(args, "l", &seed)) {
    if (getrandom(&seed, sizeof(seed), 0) < 0) {
      return NULL;
    }
  }
  PyErr_Restore(NULL, NULL, NULL); /* clear the exception */    

  srand48(seed);
  Py_RETURN_NONE;
}
#elif 0
/* Good. */
static inline double random_uniform()
{
  uint64_t r;
  getrandom(&r, sizeof(r), 0);
  return (double) (r & (1ull << 52) - 1) / ((1ull << 52) - 1);
}

static PyObject *seed()
{
  Py_RETURN_NONE;
}

#elif 0
/* Good. */
static inline double random_uniform()
{
  uint32_t r;
  getrandom(&r, sizeof(r), 0);
  return (double) r / 0xffffffff;
}

static PyObject *seed()
{
  Py_RETURN_NONE;
}
#endif

static long multinomial_choice(const long *a, long n, const long *p)
{
  long i, u, csum = 0, csum_tot = 0;

  for (i=0; i<n; i++) {
    csum_tot += p[i];
  }

  double r = random_uniform();
  u = r * csum_tot;

  for (i=0; i<n; i++) {
    csum += p[i];
    if (u < csum) {
      break;
    }
  }

  return a[i];
}

static long propose_new_partition(long r, const long *neighbors, const long *neighbor_weights,
				  const long *partition,
				  struct compressed_array *M,
				  PyObject *Mu,
				  const long *d, long n_neighbors, long N, long B, long agg_move)
{
  long s = r, u, rand_neighbor;

  struct hash *Mu_row = NULL;
  struct hash *Mu_col = NULL;

  if (!agg_move) {
    /* For unit weight graphs all probabilities are 1. */
    rand_neighbor = neighbors[(int) (random_uniform() * n_neighbors)];
  }
  else {
    rand_neighbor = multinomial_choice(neighbors, n_neighbors, neighbor_weights);
  }

  u = partition[rand_neighbor];

  if (rand_neighbor > N) {
    PyErr_SetString(PyExc_RuntimeError, "rand_neighbor out of range");
    return -1;
  }

  double rr = random_uniform();
  if (rr <= ((double) B / (d[u] + B))) {
    if (agg_move) {
      s = (r + 1 + (long) ((B - 1) * random_uniform())) % B;      
    }
    else {
      s = B * random_uniform();
    }
    goto done;
  }
  
  /* proposals by random draw from neighbors of
   * partition[rand_neighbor]
   */
  
  if (Mu) {
    size_t default_width = 16;
    Mu_row = hash_create(default_width, 0);

    if (!Mu_row) {
      goto done;
    }
    
    Mu_col = hash_create(default_width, 0);

    if (!Mu_col) {
      goto done;
    }
    
    npy_intp N = PyArray_DIM(Mu, 0);    
    long *Mu_row_p = PyArray_GETPTR2(Mu, u, 0);
    long *Mu_col_p = PyArray_GETPTR2(Mu, 0, u);

    long i;

    for (i=0; i<N; i++) {
      if (Mu_row_p[i] != 0) {
	hash_set_single(Mu_row, i, Mu_row_p[i]);
      }
    }

    for (i=0; i<N; i++) {
      if (Mu_col_p[i] != 0) {
	hash_set_single(Mu_col, i, Mu_col_p[i]);
      }
    }
  }
  else {
    Mu_row = compressed_take(M, u, 0);
    Mu_col = compressed_take(M, u, 1);
  }

  hash_val_t Mu_row_r = 0, Mu_col_r = 0;
  long sum_row = hash_sum(Mu_row);
  long sum_col = hash_sum(Mu_col);  
  long sum_tot = sum_row + sum_col;

  if (agg_move) {
    hash_search(Mu_row, r, &Mu_row_r);
    hash_search(Mu_col, r, &Mu_col_r);
    sum_row -= Mu_row_r;
    sum_col -= Mu_col_r;
    sum_tot = sum_row + sum_col;

    if (0 == sum_tot) {
      /* The current block has no (available) neighbors, 
       * so randomly propose a different block.
       */
      s = (r + 1 + (long) ((B - 1) * random_uniform())) % B;
      goto done;
    }
  }

  hash_key_t *keys;
  hash_val_t *vals;    
  long i, sum = 0;
  double rand = random_uniform();
  long p =  rand * sum_tot;

  keys = hash_get_keys(Mu_row);
  vals = hash_get_vals(Mu_row);

  for (i=0; i<Mu_row->width; i++) {
    if (keys[i] != EMPTY_KEY && !(agg_move && keys[i] == r)) {
      sum += vals[i];
      if (p < sum) {
	s = keys[i];
	goto done;
      }
    }
  }

  p -= sum_row;
  sum = 0;

  keys = hash_get_keys(Mu_col);
  vals = hash_get_vals(Mu_col);

  for (i=0; i<Mu_col->width; i++) {
    if (keys[i] != EMPTY_KEY && !(agg_move && keys[i] == r)) {
      sum += vals[i];
      if (p < sum) {
	s = keys[i];
	goto done;
      }
    }
  }
  s = -1;
  
done:
  if (Mu) {
    hash_destroy(Mu_row);
    hash_destroy(Mu_col);
  }
  
  return s;
}

static PyObject* propose_new_partition_py(PyObject *self, PyObject *args)
{
  PyObject *obj_neighbors, *obj_neighbor_weights, *obj_partition, *obj_M, *obj_d;
  long r, B, agg_move;

  if (!PyArg_ParseTuple(args, "lOOOOOll", &r, &obj_neighbors, &obj_neighbor_weights,
			&obj_partition, &obj_M, &obj_d, &B, &agg_move)) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse tuple.");
    return NULL;
  }

  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbor_weights = PyArray_FROM_OTF(obj_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d = PyArray_FROM_OTF(obj_d, NPY_LONG, NPY_IN_ARRAY);

  struct compressed_array *M = NULL;
  PyObject *Mu = NULL;

  if ((M = PyCapsule_GetPointer(obj_M, "compressed_array")) == NULL) {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */
    Mu = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY);
  }

  const long *neighbors = (const long *) PyArray_DATA(ar_neighbors);
  const long *neighbor_weights = (const long *) PyArray_DATA(ar_neighbor_weights);
  const long *partition = (const long  *) PyArray_DATA(ar_partition);
  const long *d = (const long *) PyArray_DATA(ar_d);

  long n_neighbors= (long) PyArray_DIM(ar_neighbors, 0);
  long N = (long) PyArray_DIM(ar_partition, 0);

  PyObject *ret;

  long s = propose_new_partition(r, neighbors, neighbor_weights, partition,
				 M, Mu, d, n_neighbors, N, B, agg_move);

  if (s < 0) {
    PyErr_SetString(PyExc_RuntimeError, "multinomial choice for both conditions failed");
    return NULL;
  }

  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_neighbor_weights);
  Py_DECREF(ar_partition);
  Py_DECREF(ar_d);

  ret = Py_BuildValue("l", s);
  return ret;
}

static int compute_new_rows_cols_interblock_compressed(struct compressed_array *x,
						       long r, long s,
						       const long *b_out,
						       const long *count_out,
						       long n_out,
						       const long *b_in,
						       const long *count_in,
						       long n_in,
						       long count_self,
						       long agg_move,
						       struct hash **p_cur_M_r_row,
						       struct hash **p_cur_M_r_col,
						       struct hash **p_cur_M_s_row,
						       struct hash **p_cur_M_s_col,
						       struct hash **p_new_M_r_row,
						       struct hash **p_new_M_r_col,
						       struct hash **p_new_M_s_row,
						       struct hash **p_new_M_s_col)
{
  long i;
  int rc = -1;
  hash_val_t r_row_offset = 0;
  hash_val_t r_col_offset = 0;

  if (!agg_move) {
    for (i=0; i<n_in; i++) {
      if (b_in[i] == r) {
	r_row_offset = count_in[i];
	break;
      }
    }

    for (i=0; i<n_out; i++) {
      if (b_out[i] == r) {
	r_col_offset = count_out[i];
	break;
      }
    }
  }

  hash_val_t s_row_offset = count_self;
  for (i=0; i<n_in; i++) {
    if (b_in[i] == s) {
      s_row_offset += count_in[i];
      break;
    }
  }

  hash_val_t s_col_offset = count_self;
  for (i=0; i<n_out; i++) {
    if (b_out[i] == s) {
      s_col_offset += count_out[i];
      break;
    }
  }

  struct hash *cur_M_r_row = NULL,
    *cur_M_r_col = NULL,
    *cur_M_s_row = NULL,
    *cur_M_s_col = NULL,
    *new_M_r_row = NULL,
    *new_M_r_col = NULL,
    *new_M_s_row = NULL,
    *new_M_s_col = NULL;
    
  /* Take references to current rows and cols. */
  cur_M_r_row = compressed_take(x, r, 0);
  cur_M_r_col = compressed_take(x, r, 1);
  cur_M_s_row = compressed_take(x, s, 0);
  cur_M_s_col = compressed_take(x, s, 1);

  if (agg_move) {
    size_t default_width = 16;
    new_M_r_row = hash_create(default_width, 0);
    new_M_r_col = hash_create(default_width, 0);
  }
  else {
    /* Compute new_M_r_row */
    new_M_r_row = hash_copy(cur_M_r_row, 0);
    new_M_r_row = hash_accum_multi(new_M_r_row, (const unsigned long *) b_out, count_out, n_out, -1);

    if (1 == hash_accum_single(new_M_r_row, r, -r_row_offset)) {
      if (!(new_M_r_row = hash_resize(new_M_r_row))) { goto hash_resize_failed; }
    }

    if (1 == hash_accum_single(new_M_r_row, s, +r_row_offset)) {
      if (!(new_M_r_row = hash_resize(new_M_r_row))) { goto hash_resize_failed; }
    }

    /* Compute new_M_r_col */
    new_M_r_col = hash_copy(cur_M_r_col, 0);
    new_M_r_col = hash_accum_multi(new_M_r_col, (const unsigned long *) b_in,
				   count_in, n_in, -1);

    if (1 == hash_accum_single(new_M_r_col, r, -r_col_offset)) {
      if (!(new_M_r_col = hash_resize(new_M_r_col))) { goto hash_resize_failed; }
    }

    if (1 == hash_accum_single(new_M_r_col, s, +r_col_offset)) {
      if (!(new_M_r_col = hash_resize(new_M_r_col))) { goto hash_resize_failed; }
    }
  }

  /* Compute new_M_s_row */
  new_M_s_row = hash_copy(cur_M_s_row, 0);    
  new_M_s_row = hash_accum_multi(new_M_s_row,
				 (const unsigned long *) b_out, count_out, n_out, +1);

  if (1 == hash_accum_single(new_M_s_row, r, -s_row_offset)) {
    if (!(new_M_s_row = hash_resize(new_M_s_row))) { goto hash_resize_failed; }
  }

  if (1 == hash_accum_single(new_M_s_row, s, +s_row_offset)) {
    if (!(new_M_s_row = hash_resize(new_M_s_row))) { goto hash_resize_failed; }
  }

  /* Compute new_M_s_col */
  new_M_s_col = hash_copy(cur_M_s_col, 0);
  new_M_s_col = hash_accum_multi(new_M_s_col, (const unsigned long *) b_in,
				 count_in, n_in, +1);

  if (1 == hash_accum_single(new_M_s_col, r, -s_col_offset)) {
    if (!(new_M_s_col = hash_resize(new_M_s_col))) { goto hash_resize_failed; }
  }

  if (1 == hash_accum_single(new_M_s_col, s, +s_col_offset)) {
    if (!(new_M_s_col = hash_resize(new_M_s_col))) { goto hash_resize_failed; }
  }

  *p_cur_M_r_row = cur_M_r_row;
  *p_cur_M_r_col = cur_M_r_col;
  *p_cur_M_s_row = cur_M_s_row;
  *p_cur_M_s_col = cur_M_s_col;
  *p_new_M_r_row = new_M_r_row;
  *p_new_M_r_col = new_M_r_col;
  *p_new_M_s_row = new_M_s_row;
  *p_new_M_s_col = new_M_s_col;
  return 0;
    
hash_resize_failed:
  hash_destroy(cur_M_r_row);
  hash_destroy(cur_M_r_col);
  hash_destroy(cur_M_s_row);
  hash_destroy(cur_M_s_col);
  hash_destroy(new_M_r_row);
  hash_destroy(new_M_r_col);
  hash_destroy(new_M_s_row);
  hash_destroy(new_M_s_col);
  return rc;
}


static int compute_new_rows_cols_interblock_uncompressed(PyObject *ar_M,
							 long r, long s,
							 const long *b_out,
							 const long *count_out,
							 long n_out,
							 const long *b_in,
							 const long *count_in,
							 long n_in,
							 long count_self,
							 long agg_move,
							 const long *M,
							 npy_intp N,
							 PyObject **p_cur_M_r_row,
							 PyObject **p_cur_M_r_col,
							 PyObject **p_cur_M_s_row,
							 PyObject **p_cur_M_s_col,
							 PyObject **p_new_M_r_row,
							 PyObject **p_new_M_r_col,
							 PyObject **p_new_M_s_row,
							 PyObject **p_new_M_s_col)
{
  long i;
  hash_val_t r_row_offset = 0;
  hash_val_t r_col_offset = 0;

  if (!agg_move) {
    for (i=0; i<n_in; i++) {
      if (b_in[i] == r) {
	r_row_offset = count_in[i];
	break;
      }
    }

    for (i=0; i<n_out; i++) {
      if (b_out[i] == r) {
	r_col_offset = count_out[i];
	break;
      }
    }
  }

  hash_val_t s_row_offset = count_self;
  for (i=0; i<n_in; i++) {
    if (b_in[i] == s) {
      s_row_offset += count_in[i];
      break;
    }
  }

  hash_val_t s_col_offset = count_self;
  for (i=0; i<n_out; i++) {
    if (b_out[i] == s) {
      s_col_offset += count_out[i];
      break;
    }
  }

  PyObject
    *cur_M_r_row = NULL,
    *cur_M_r_col = NULL,
    *cur_M_s_row = NULL,
    *cur_M_s_col = NULL,
    *new_M_r_row = NULL,
    *new_M_r_col = NULL,
    *new_M_s_row = NULL,
    *new_M_s_col = NULL;

  cur_M_r_row = take_view(ar_M, r, 0);
  cur_M_r_col = take_view(ar_M, r, 1);
  cur_M_s_row = take_view(ar_M, s, 0);
  cur_M_s_col = take_view(ar_M, s, 1);

  new_M_s_row = PyArray_NewCopy((PyArrayObject *) cur_M_s_row, NPY_ANYORDER);
  new_M_s_col = PyArray_NewCopy((PyArrayObject *) cur_M_s_col, NPY_ANYORDER);

  if (agg_move) {
    /* Consider refactoring to return Py_None (with Py_INCREF) */
    npy_intp dims[] = { N };
    new_M_r_row = PyArray_Zeros(1, dims,
				PyArray_DescrNewFromType(PyArray_TYPE(ar_M)), 0);
    new_M_r_col = PyArray_Zeros(1, dims,
				PyArray_DescrNewFromType(PyArray_TYPE(ar_M)), 0);
  }
  else {
    new_M_r_row = PyArray_NewCopy((PyArrayObject *) cur_M_r_row, NPY_ANYORDER);
    long *c_new_M_r_row = (long *) PyArray_DATA(new_M_r_row);

    for (i=0; i<n_out; i++) {
      c_new_M_r_row[b_out[i]] -= count_out[i];
    }
    c_new_M_r_row[r] -= r_row_offset;
    c_new_M_r_row[s] += r_row_offset;

    new_M_r_col = PyArray_NewCopy((PyArrayObject *) cur_M_r_col, NPY_ANYORDER);
    long *c_new_M_r_col = (long *) PyArray_DATA(new_M_r_col);

    for (i=0; i<n_in; i++) {
      c_new_M_r_col[b_in[i]] -= count_in[i];
    }
    c_new_M_r_col[r] -= r_col_offset;
    c_new_M_r_col[s] += r_col_offset;
  }

  long *c_new_M_s_row = (long *) PyArray_DATA(new_M_s_row);
  long *c_new_M_s_col = (long *) PyArray_DATA(new_M_s_col);    
    
  for (i=0; i<n_out; i++) {
    c_new_M_s_row[b_out[i]] += count_out[i];
  }
  c_new_M_s_row[r] -= s_row_offset;
  c_new_M_s_row[s] += s_row_offset;

  for (i=0; i<n_in; i++) {
    c_new_M_s_col[b_in[i]] += count_in[i];
  }
  c_new_M_s_col[r] -= s_col_offset;
  c_new_M_s_col[s] += s_col_offset;

  *p_cur_M_r_row = cur_M_r_row;
  *p_cur_M_r_col = cur_M_r_col;
  *p_cur_M_s_row = cur_M_s_row;
  *p_cur_M_s_col = cur_M_s_col;
  *p_new_M_r_row = new_M_r_row;
  *p_new_M_r_col = new_M_r_col;
  *p_new_M_s_row = new_M_s_row;
  *p_new_M_s_col = new_M_s_col;
  return 0;
  
}

/*
 * Args: M, r, s, b_out, count_out, b_in, count_in, count_self, agg_move
 */
static PyObject* compute_new_rows_cols_interblock(PyObject *self, PyObject *args)
{
  PyObject *obj_M, *obj_b_out, *obj_count_out, *obj_b_in, *obj_count_in;
  long r, s, count_self, agg_move;

  if (!PyArg_ParseTuple(args, "OllOOOOll", &obj_M, &r, &s, &obj_b_out, &obj_count_out, &obj_b_in, &obj_count_in, &count_self, &agg_move)) {
    return NULL;
  }

  const PyObject *ar_b_out = PyArray_FROM_OTF(obj_b_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_out = PyArray_FROM_OTF(obj_count_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_b_in = PyArray_FROM_OTF(obj_b_in, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_count_in = PyArray_FROM_OTF(obj_count_in, NPY_LONG, NPY_IN_ARRAY);

  const long *b_out = (const long *) PyArray_DATA(ar_b_out);
  const long *count_out = (const long *) PyArray_DATA(ar_count_out);
  const long *b_in = (const long  *) PyArray_DATA(ar_b_in);
  const long *count_in = (const long *) PyArray_DATA(ar_count_in);

  long n_out= (long) PyArray_DIM(ar_b_out, 0);
  long n_in = (long) PyArray_DIM(ar_b_in, 0);
  
  PyObject *ar_M;

  PyObject *ret_new_M_r_row = NULL, *ret_new_M_r_col = NULL, *ret_new_M_s_row = NULL, *ret_new_M_s_col = NULL,
	  *ret_cur_M_r_row = NULL, *ret_cur_M_r_col = NULL, *ret_cur_M_s_row = NULL, *ret_cur_M_s_col = NULL;

  struct compressed_array *x = PyCapsule_GetPointer(obj_M, "compressed_array");
  PyObject *ret = NULL;
  
  if (x) {
    struct hash *cur_M_r_row = NULL,
      *cur_M_r_col = NULL,
      *cur_M_s_row = NULL,
      *cur_M_s_col = NULL,
      *new_M_r_row = NULL,
      *new_M_r_col = NULL,
      *new_M_s_row = NULL,
      *new_M_s_col = NULL;
    
    if (compute_new_rows_cols_interblock_compressed(x, r, s, b_out, count_out, n_out,
						    b_in, count_in, n_in,
						    count_self, agg_move,
						    &cur_M_r_row, &cur_M_r_col, &cur_M_s_row, &cur_M_s_col,
						    &new_M_r_row, &new_M_r_col, &new_M_s_row, &new_M_s_col) < 0)
    {
      PyErr_SetString(PyExc_RuntimeError, "compute_new_rows_cols_interblock_compressed failed");
      goto done;
    }

    struct hash **p_new_M_r_row = create_dict(new_M_r_row);
    struct hash **p_new_M_r_col = create_dict(new_M_r_col);
    struct hash **p_new_M_s_row = create_dict(new_M_s_row);
    struct hash **p_new_M_s_col = create_dict(new_M_s_col);    

    struct hash **p_cur_M_r_row = create_dict(cur_M_r_row);
    struct hash **p_cur_M_r_col = create_dict(cur_M_r_col);
    struct hash **p_cur_M_s_row = create_dict(cur_M_s_row);
    struct hash **p_cur_M_s_col = create_dict(cur_M_s_col);    

    ret_new_M_r_row = PyCapsule_New(p_new_M_r_row, "compressed_array_dict", destroy_dict_copy);
    ret_new_M_r_col = PyCapsule_New(p_new_M_r_col, "compressed_array_dict", destroy_dict_copy);
    ret_new_M_s_row = PyCapsule_New(p_new_M_s_row, "compressed_array_dict", destroy_dict_copy);
    ret_new_M_s_col = PyCapsule_New(p_new_M_s_col, "compressed_array_dict", destroy_dict_copy);

    ret_cur_M_r_row = PyCapsule_New(p_cur_M_r_row, "compressed_array_dict", destroy_dict_ref);
    ret_cur_M_r_col = PyCapsule_New(p_cur_M_r_col, "compressed_array_dict", destroy_dict_ref);
    ret_cur_M_s_row = PyCapsule_New(p_cur_M_s_row, "compressed_array_dict", destroy_dict_ref);
    ret_cur_M_s_col = PyCapsule_New(p_cur_M_s_col, "compressed_array_dict", destroy_dict_ref);

    ret = Py_BuildValue("NNNNNNNN",
			ret_new_M_r_row, ret_new_M_r_col, ret_new_M_s_row, ret_new_M_s_col,
			ret_cur_M_r_row, ret_cur_M_r_col, ret_cur_M_s_row, ret_cur_M_s_col);

  }
  else if ((ar_M = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY))) {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */

    PyObject
      *cur_M_r_row = NULL,
      *cur_M_r_col = NULL,
      *cur_M_s_row = NULL,
      *cur_M_s_col = NULL,
      *new_M_r_row = NULL,
      *new_M_r_col = NULL,
      *new_M_s_row = NULL,
      *new_M_s_col = NULL;

    const long *M = (const long *) PyArray_DATA(ar_M);
    npy_intp N = PyArray_DIM(ar_M, 0);
    
    if (compute_new_rows_cols_interblock_uncompressed(ar_M, r, s, b_out, count_out, n_out,
						      b_in, count_in, n_in,
						      count_self, agg_move,
						      M,
						      N,
						      &cur_M_r_row, &cur_M_r_col, &cur_M_s_row, &cur_M_s_col,
						      &new_M_r_row, &new_M_r_col, &new_M_s_row, &new_M_s_col) < 0)
    {
      PyErr_SetString(PyExc_RuntimeError, "compute_new_rows_cols_interblock_compressed failed");
      goto done;
    }
    ret = Py_BuildValue("NNNNNNNN",
			new_M_r_row, new_M_r_col, new_M_s_row, new_M_s_col,
			cur_M_r_row, cur_M_r_col, cur_M_s_row, cur_M_s_col);
    Py_DECREF(ar_M);

  }
  else {
    PyErr_SetString(PyExc_RuntimeError, "Invalid obj_M object type.");
  }

done:
  Py_DECREF(ar_b_out);
  Py_DECREF(ar_count_out);
  Py_DECREF(ar_b_in);
  Py_DECREF(ar_count_in);

  return ret;
}

static inline double xlogx(double x)
{
  return x > 0.0 ? x * log(x) : 0.0;
}

static inline double logx(double x)
{
  return x > 0.0 ? log(x) : 0.0;
}

static inline long degree_substitute(const long *d, long i, long r, long s, long d_new_r, long d_new_s)
{
  if (i == r) {
    return d_new_r;
  }
  else if (i == s) {
    return d_new_s;
  }
  return d[i];
}

static void compute_delta_entropy(PyObject *Mu,
				  struct compressed_array *M,
				  const long r,
				  const long s,
				  const long *restrict b_out,
				  const long *restrict count_out,
				  const long n_out,
				  const long *restrict b_in,
				  const long *restrict count_in,
				  const long n_in,
				  const long *restrict d_out,
				  const long *restrict d_in,
				  const long *restrict d,
				  const long d_out_new_r,
				  const long d_out_new_s,
				  const long d_in_new_r,
				  const long d_in_new_s,
				  int agg_move,
				  double *restrict p_delta_entropy,
				  long *restrict p_Nrr,
				  long *restrict p_Nrs,
				  long *restrict p_Nsr,
				  long *restrict p_Nss
			 )
{
  long i;

  double cur_S_r_row = 0.0, new_S_r_row = 0.0;
  double cur_S_s_row = 0.0, new_S_s_row = 0.0;
  double cur_S_r_col = 0.0, new_S_r_col = 0.0;
  double cur_S_s_col = 0.0, new_S_s_col = 0.0;
  double cur_Srr = 0.0, cur_Srs = 0.0, cur_Ssr = 0.0, cur_Sss = 0.0;
  double new_Srr = 0.0, new_Srs = 0.0, new_Ssr = 0.0, new_Sss = 0.0;
  long Mrr, Mrs, Msr, Mss;
  
  if (Mu) {
    Mrr = *(long *)PyArray_GETPTR2(Mu, r, r);
    Mrs = *(long *)PyArray_GETPTR2(Mu, r, s);
    Msr = *(long *)PyArray_GETPTR2(Mu, s, r);
    Mss = *(long *)PyArray_GETPTR2(Mu, s, s);
  }
  else {
    compressed_get_single(M, r, r, &Mrr);
    compressed_get_single(M, r, s, &Mrs);
    compressed_get_single(M, s, r, &Msr);
    compressed_get_single(M, s, s, &Mss);
  }

  long Nrr = Mrr, Nrs = Mrs, Nsr = Msr, Nss = Mss;
  long Mij;

  /* Temporary -- does not support count_self != 0 */
  long count_self = 0;

  hash_val_t r_row_offset = 0;
  if (!agg_move) {
    for (i=0; i<n_in; i++) {
      if (b_in[i] == r) {
	r_row_offset = count_in[i];
	break;
      }
    }

#if 0
    hash_val_t r_col_offset = 0; /* XXX set but not used */
    for (i=0; i<n_out; i++) {
      if (b_out[i] == r) {
	r_col_offset = count_out[i];
	break;
      }
    }
#endif    
  }

  hash_val_t s_row_offset = count_self;
  for (i=0; i<n_in; i++) {
    if (b_in[i] == s) {
      s_row_offset += count_in[i];
      break;
    }
  }

#if 0  
  /* XXX set but not used */  
  hash_val_t s_col_offset = count_self;
  for (i=0; i<n_out; i++) {
    if (b_out[i] == s) {
      s_col_offset += count_out[i];
      break;
    }
  }
#endif  

  /* Entropy over M_r_row current, and proposed */
  cur_S_r_row -= xlogx(d_out[r]);  
  new_S_r_row -= xlogx(d_out_new_r);
  for (i=0; i<n_out; i++) {
    if (Mu) {
      Mij = *(long *)PyArray_GETPTR2(Mu, r, b_out[i]);
    }
    else {
      compressed_get_single(M, r, b_out[i], &Mij);
    }

    if (b_out[i] == r) {
      Nrr -= count_out[i];
      continue;
    }
    if (b_out[i] == s) {
      Nrs -= count_out[i];
      continue;
    }

    cur_S_r_row += xlogx(Mij);
    cur_S_r_row -= Mij * logx(d_in[b_out[i]]);

    Mij -= count_out[i];
    new_S_r_row += xlogx(Mij);
    new_S_r_row -= Mij * logx(degree_substitute(d_in, b_out[i], r, s, d_in_new_r, d_in_new_s));
  }

  /* Entropy over M_s_row current, and proposed */
  cur_S_s_row -= d_out[s] * logx(d_out[s]);
  new_S_s_row -= d_out_new_s * logx(d_out_new_s);

  for (i=0; i<n_out; i++) {
    if (Mu) {
      Mij = *(long *)PyArray_GETPTR2(Mu, s, b_out[i]);
    }
    else {
      compressed_get_single(M, s, b_out[i], &Mij);
    }

    if (b_out[i] == r) {
      Nsr += count_out[i];
      continue;
    }
    if (b_out[i] == s) {
      Nss += count_out[i];
      continue;
    }

    cur_S_s_row += xlogx(Mij);
    cur_S_s_row -= Mij * logx(d_in[b_out[i]]);

    Mij += count_out[i];
    new_S_s_row += xlogx(Mij);
    new_S_s_row -= Mij * logx(degree_substitute(d_in, b_out[i], r, s, d_in_new_r, d_in_new_s));
  }

  /* Entropy over M_r_col current, and proposed */
  cur_S_r_col -= xlogx(d_in[r]);
  new_S_r_col -= xlogx(d_in_new_r);

  for (i=0; i<n_in; i++) {
    if (Mu) {
      Mij = *(long *)PyArray_GETPTR2(Mu, b_in[i], r);
    }
    else {
      compressed_get_single(M, b_in[i], r, &Mij);
    }

    if (b_in[i] == r || b_in[i] == s) {
      continue;
    }

    cur_S_r_col += xlogx(Mij);
    cur_S_r_col -= Mij * logx(d_out[b_in[i]]);

    Mij -= count_in[i];
    new_S_r_col += xlogx(Mij);
    new_S_r_col -= Mij * logx(degree_substitute(d_out, b_in[i], r, s, d_out_new_r, d_out_new_s));    
  }
  
  /* Entropy over M_s_col current, and proposed */  
  cur_S_s_col -= xlogx(d_in[s]);
  new_S_s_col -= xlogx(d_in_new_s);

  for (i=0; i<n_in; i++) {
    if (Mu) { 
      Mij = *(long *)PyArray_GETPTR2(Mu, b_in[i], s);
    }
    else {
      compressed_get_single(M, b_in[i], s, &Mij);
    }
    
    if (b_in[i] == r || b_in[i] == s) {
      continue;
    }
    cur_S_s_col += xlogx(Mij);
    cur_S_s_col -= Mij * logx(d_out[b_in[i]]);

    Mij += count_in[i];
    new_S_s_col += xlogx(Mij);
    new_S_s_col -= Mij * logx(degree_substitute(d_out, b_in[i], r, s, d_out_new_r, d_out_new_s));    
  }

  /* Corner M[r,r] */
  cur_S_r_row += xlogx(Mrr);
  cur_S_r_row -= Mrr * logx(d_in[r]);

  cur_Srr += xlogx(Mrr);
  cur_Srr -= Mrr * logx(d_in[r]);
  cur_Srr -= Mrr * logx(d_out[r]);

  cur_S_r_col += xlogx(Mrr);
  cur_S_r_col -= Mrr * logx(d_out[r]);

  Nrr -= r_row_offset;
  new_S_r_row += xlogx(Nrr);
  new_S_r_row -= Nrr * logx(d_in_new_r);

  new_S_r_col += Nrr * logx(Nrr);
  new_S_r_col -= Nrr * logx(d_out_new_r);

  new_Srr += xlogx(Nrr);
  new_Srr -= Nrr * logx(d_in_new_r);
  new_Srr -= Nrr * logx(d_out_new_r);

  /* Corner M[r,s] */
  cur_S_r_row += xlogx(Mrs);
  cur_S_r_row -= Mrs * logx(d_in[s]);

  cur_Srs += xlogx(Mrs);
  cur_Srs -= Mrs * logx(d_in[s]);
  cur_Srs -= Mrs * logx(d_out[r]);

  cur_S_s_col += xlogx(Mrs);
  cur_S_s_col -= Mrs * logx(d_out[r]);

  Nrs += r_row_offset;
  new_S_r_row += xlogx(Nrs);
  new_S_r_row -= Nrs * logx(d_in_new_s);

  new_S_s_col += xlogx(Nrs);
  new_S_s_col -= Nrs * logx(d_out_new_r);

  new_Srs += xlogx(Nrs);
  new_Srs -= Nrs * logx(d_in_new_s);
  new_Srs -= Nrs * logx(d_out_new_r);

  /* Corner M[s,r] */
  cur_S_s_row += xlogx(Msr);
  cur_S_s_row -= Msr * logx(d_in[r]);

  cur_S_r_col += xlogx(Msr);
  cur_S_r_col -= Msr * logx(d_out[s]);

  cur_Ssr += xlogx(Msr);
  cur_Ssr -= Msr * logx(d_in[r]);
  cur_Ssr -= Msr * logx(d_out[s]);

  Nsr -= s_row_offset;

  /* Ugly hack for agglomerative moves.
   * Nsr and Nrs should both be zero.
   */
  if (agg_move && Nsr > 0) {
    Nss += Nsr;
    Nsr = 0;
  }
  
  new_S_s_row += xlogx(Nsr);
  new_S_s_row -= Nsr * logx(d_in_new_r);

  new_S_r_col += xlogx(Nsr);
  new_S_r_col -= Nsr * logx(d_out_new_s);

  new_Ssr += xlogx(Nsr);
  new_Ssr -= Nsr * logx(d_in_new_r);
  new_Ssr -= Nsr * logx(d_out_new_s);

  /* Corner M[s,s] */
  cur_S_s_row += xlogx(Mss);
  cur_S_s_row -= Mss * logx(d_in[s]);

  cur_S_s_col += xlogx(Mss);
  cur_S_s_col -= Mss * logx(d_out[s]);

  cur_Sss += xlogx(Mss);
  cur_Sss -= Mss * logx(d_in[s]);
  cur_Sss -= Mss * logx(d_out[s]);

  Nss += s_row_offset;
  new_S_s_row += xlogx(Nss);
  new_S_s_row -= Nss * logx(d_in_new_s);

  new_S_s_col += xlogx(Nss);
  new_S_s_col -= Nss * logx(d_out_new_s);

  new_Sss += xlogx(Nss);
  new_Sss -= Nss * logx(d_in_new_s);
  new_Sss -= Nss * logx(d_out_new_s);
  
  double cur_S = -cur_S_r_row - cur_S_s_row - cur_S_r_col - cur_S_s_col
    + cur_Srr + cur_Srs + cur_Ssr + cur_Sss;
  double new_S = -new_S_r_row - new_S_s_row - new_S_r_col - new_S_s_col
    + new_Srr + new_Srs + new_Ssr + new_Sss;
  
  *p_delta_entropy = new_S - cur_S;
  *p_Nrr = Nrr;
  *p_Nrs = Nrs;
  *p_Nsr = Nsr;
  *p_Nss = Nss;
}

static PyObject *propose_block_merge(PyObject *self, PyObject *args)
{
  PyObject *obj_neighbors, *obj_neighbor_weights,
    *obj_out_neighbors, *obj_out_neighbor_weights, *obj_in_neighbors, *obj_in_neighbor_weights,    
    *obj_partition, *obj_M,
    *obj_block_degrees, *obj_block_degrees_out, *obj_block_degrees_in;
  long r, s, num_blocks, num_out_block_edges, num_in_block_edges;

  if (!PyArg_ParseTuple(args, "OllOOOOOOOlOOOll", &obj_M, &r, &s,
			&obj_out_neighbors, &obj_out_neighbor_weights,
			&obj_in_neighbors, &obj_in_neighbor_weights,			
			&obj_neighbors, &obj_neighbor_weights,
			&obj_partition, &num_blocks,
			&obj_block_degrees, &obj_block_degrees_out, &obj_block_degrees_in,
			&num_out_block_edges, &num_in_block_edges)) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse tuple.");
    return NULL;
  }

  const PyObject *ar_out_neighbors = PyArray_FROM_OTF(obj_out_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_out_neighbor_weights = PyArray_FROM_OTF(obj_out_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_in_neighbors = PyArray_FROM_OTF(obj_in_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_in_neighbor_weights = PyArray_FROM_OTF(obj_in_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbor_weights = PyArray_FROM_OTF(obj_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d = PyArray_FROM_OTF(obj_block_degrees, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_out = PyArray_FROM_OTF(obj_block_degrees_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_in = PyArray_FROM_OTF(obj_block_degrees_in, NPY_LONG, NPY_IN_ARRAY);      

  const long *partition = (const long  *) PyArray_DATA(ar_partition);
  const long *out_neighbors = (const long *) PyArray_DATA(ar_out_neighbors);
  const long *out_neighbor_weights = (const long *) PyArray_DATA(ar_out_neighbor_weights);
  const long *in_neighbors = (const long *) PyArray_DATA(ar_in_neighbors);
  const long *in_neighbor_weights = (const long *) PyArray_DATA(ar_in_neighbor_weights);
  const long *neighbors = (const long *) PyArray_DATA(ar_neighbors);
  const long *neighbor_weights = (const long *) PyArray_DATA(ar_neighbor_weights);

  long N = (long) PyArray_DIM(ar_partition, 0);
  long n_neighbors= (long) PyArray_DIM(ar_neighbors, 0);

  struct compressed_array *M = PyCapsule_GetPointer(obj_M, "compressed_array");
  PyObject *Mu = NULL;
  
  if (!M) {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */
    Mu = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY);
  }

  const long *d = (const long *) PyArray_DATA(ar_d);
  const long *d_out = (const long *) PyArray_DATA(ar_d_out);
  const long *d_in = (const long *) PyArray_DATA(ar_d_in);
  
  PyObject *ret;

  if (s == -1) {
    s = propose_new_partition(r, neighbors, neighbor_weights, partition,
			      M, Mu, d, n_neighbors, N, num_blocks, 1);
  }

  if (s < 0) {
    PyErr_SetString(PyExc_RuntimeError, "propose_new_partition failed");
    return NULL;
  }

  long *b_out = NULL, *b_in = NULL;
  long *count_out = NULL, *count_in = NULL, n_out, n_in;
  
  double delta_entropy = 0.0;

  const long d_out_new_r = d_out[r] - num_out_block_edges;
  const long d_out_new_s = d_out[s] + num_out_block_edges;
  const long d_in_new_r = d_in[r] - num_in_block_edges;
  const long d_in_new_s = d_in[s] + num_in_block_edges;

  long Nrr, Nrs, Nsr, Nss;

  if (!Mu) {
    compressed_take_keys_values(M, r, 0, (unsigned long **) &b_out, &count_out, &n_out);
    compressed_take_keys_values(M, r, 1, (unsigned long **) &b_in, &count_in, &n_in);

    compute_delta_entropy(Mu, M, r, s, b_out, count_out, n_out, b_in, count_in, n_in, d_out, d_in, d, d_out_new_r, d_out_new_s, d_in_new_r, d_in_new_s, 1, &delta_entropy, &Nrr, &Nrs, &Nsr, &Nss);
  }
  else {
    long n_out_neighbors= (long) PyArray_DIM(ar_out_neighbors, 0);
    if (blocks_and_counts(partition, out_neighbors, out_neighbor_weights, n_out_neighbors, &b_out, &count_out, &n_out, NULL) < 0) {
      PyErr_SetString(PyExc_RuntimeError, "blocks_and_counts_failed");
      return NULL;
    }
    long n_in_neighbors= (long) PyArray_DIM(ar_in_neighbors, 0);
    if (blocks_and_counts(partition, in_neighbors, in_neighbor_weights, n_in_neighbors, &b_in, &count_in, &n_in, NULL) < 0) {
      PyErr_SetString(PyExc_RuntimeError, "blocks_and_counts_failed");
      return NULL;
    }

    compute_delta_entropy(Mu, M, r, s, b_out, count_out, n_out, b_in, count_in, n_in, d_out, d_in, d, d_out_new_r, d_out_new_s, d_in_new_r, d_in_new_s, 1, &delta_entropy, &Nrr, &Nrs, &Nsr, &Nss);
  }  
  

  free(b_out);
  free(count_out);  
  free(b_in);
  free(count_in);
  
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_neighbor_weights);
  Py_DECREF(ar_partition);
  Py_DECREF(ar_d_out);
  Py_DECREF(ar_d_in);
  Py_DECREF(ar_d);  

  ret = Py_BuildValue("kd", s, delta_entropy);
  return ret;
}


static PyObject *propose_nodal_movement(PyObject *self, PyObject *args)
{
  PyObject *obj_partition,
    *obj_out_neighbors, *obj_out_neighbor_weights, *obj_in_neighbors, *obj_in_neighbor_weights,
    *obj_M,
    *obj_block_degrees, *obj_block_degrees_out, *obj_block_degrees_in,
    *obj_neighbors, *obj_neighbor_weights, *obj_self_edge_weights;

  long ni, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges, num_blocks, s;
  double beta;
  /* self_edge_weights is a defaultdict, unused for now */
  if (!PyArg_ParseTuple(args, "lOOOOOOlOOOlllOOOdl",
			&ni,
			&obj_partition,
			&obj_out_neighbors, &obj_out_neighbor_weights,
			&obj_in_neighbors, &obj_in_neighbor_weights,
			&obj_M,
			&num_blocks,
			&obj_block_degrees, &obj_block_degrees_out, &obj_block_degrees_in,
			&num_out_neighbor_edges, &num_in_neighbor_edges, &num_neighbor_edges,
			&obj_neighbors, &obj_neighbor_weights,
			&obj_self_edge_weights, &beta, &s))
  {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse tuple.");
    return NULL;
  }

  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_out_neighbors = PyArray_FROM_OTF(obj_out_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_out_neighbor_weights = PyArray_FROM_OTF(obj_out_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_in_neighbors = PyArray_FROM_OTF(obj_in_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_in_neighbor_weights = PyArray_FROM_OTF(obj_in_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);

  struct compressed_array *Mc = PyCapsule_GetPointer(obj_M, "compressed_array");
  PyObject *Mu = NULL;
  
  if (!Mc) {
    PyErr_Restore(NULL, NULL, NULL); /* clear the exception */
    Mu = PyArray_FROM_OTF(obj_M, NPY_LONG, NPY_IN_ARRAY);
  }

  /* d, d_out, d_in have size num_blocks */
  const PyObject *ar_d = PyArray_FROM_OTF(obj_block_degrees, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_out = PyArray_FROM_OTF(obj_block_degrees_out, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_d_in = PyArray_FROM_OTF(obj_block_degrees_in, NPY_LONG, NPY_IN_ARRAY);    
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbor_weights = PyArray_FROM_OTF(obj_neighbor_weights, NPY_LONG, NPY_IN_ARRAY);

  const long *partition = (const long  *) PyArray_DATA(ar_partition);
  const long *out_neighbors = (const long *) PyArray_DATA(ar_out_neighbors);
  const long *out_neighbor_weights = (const long *) PyArray_DATA(ar_out_neighbor_weights);
  const long *in_neighbors = (const long *) PyArray_DATA(ar_in_neighbors);
  const long *in_neighbor_weights = (const long *) PyArray_DATA(ar_in_neighbor_weights);

  const long *d = (const long *) PyArray_DATA(ar_d);
  const long *d_out = (const long *) PyArray_DATA(ar_d_out);
  const long *d_in = (const long *) PyArray_DATA(ar_d_in);

  const long *neighbors = (const long *) PyArray_DATA(ar_neighbors);
  const long *neighbor_weights = (const long *) PyArray_DATA(ar_neighbor_weights);

  long n_out_neighbors = (long) PyArray_DIM(ar_out_neighbors, 0);
  long n_in_neighbors = (long) PyArray_DIM(ar_in_neighbors, 0);  
  long n_neighbors = (long) PyArray_DIM(ar_neighbors, 0);
  long N = (long) PyArray_DIM(ar_partition, 0);

  PyObject *ret = NULL;
  
  long *b_out = NULL, *b_in = NULL;
  long *count_out = NULL, *count_in = NULL, n_out, n_in;
  double p_accept = 0.0, delta_entropy = 0.0, prob_back = 0.0, prob_fwd = 0.0, hastings = 0.0;
  struct hash *h_out = NULL, *h_in = NULL;

  long r = partition[ni];

  if (s == -1) {
    s = propose_new_partition(r, neighbors, neighbor_weights, partition,
			      Mc, Mu, d, n_neighbors, N, num_blocks, 0);
  }

  if (s < 0) {
    PyErr_SetString(PyExc_RuntimeError, "propose_new_partition failed");
    return NULL;
  }
  
  if (s == r) {
    ret = Py_BuildValue("kkkdd", ni,r,s,delta_entropy,p_accept);
    goto done;
  }

  if (blocks_and_counts(partition, out_neighbors, out_neighbor_weights, n_out_neighbors, &b_out, &count_out, &n_out, &h_out) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "blocks_and_counts_failed");
    goto done;
  }

  if (blocks_and_counts(partition, in_neighbors, in_neighbor_weights, n_in_neighbors, &b_in, &count_in, &n_in, &h_in) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "blocks_and_counts_failed");    
    goto done;
  }
  
  int64_t dM_r_row_sum = 0, dM_r_col_sum = 0;

  /* Components of Hastings correction. */
  const double B = (double) PyArray_DIM(ar_d_out, 0);
  long i;

  for (i=0; i<n_out; i++) {
    dM_r_row_sum -= count_out[i];
  }

  for (i=0; i<n_in; i++) {
    dM_r_col_sum -= count_in[i];
  }
  
  const long d_out_new_r = d_out[r] + dM_r_row_sum;
  const long d_out_new_s = d_out[s] - dM_r_row_sum;
  const long d_in_new_r = d_in[r] + dM_r_col_sum;
  const long d_in_new_s = d_in[s] - dM_r_col_sum;  

  long Nrr, Nrs, Nsr, Nss;

  compute_delta_entropy(Mu, Mc, r, s, b_out, count_out, n_out, b_in, count_in, n_in, d_out, d_in, d, d_out_new_r, d_out_new_s, d_in_new_r, d_in_new_s, 0, &delta_entropy, &Nrr, &Nrs, &Nsr, &Nss);
  
  /* Hastings correction */
  for (i=0; i<n_out; i++) {
    long Mts, Mst;
    if (Mu) {
      Mts = *(long *)PyArray_GETPTR2(Mu, b_out[i], s);
      Mst = *(long *)PyArray_GETPTR2(Mu, s, b_out[i]);
    }
    else {
      compressed_get_single(Mc, b_out[i], s, &Mts);
      compressed_get_single(Mc, s, b_out[i], &Mst);
    }
    prob_fwd += count_out[i] * (Mts + Mst + 1) / (B + d[b_out[i]]);
  }

  for (i=0; i<n_in; i++) {
    long Mst, Mts;
    if (Mu) {
      Mts = *(long *)PyArray_GETPTR2(Mu, b_in[i], s);
      Mst = *(long *)PyArray_GETPTR2(Mu, s, b_in[i]);
    }
    else {
      compressed_get_single(Mc, b_in[i], s, &Mts);
      compressed_get_single(Mc, s, b_in[i], &Mst);
    }
    prob_fwd += count_in[i] * (Mts + Mst + 1) / (B + d[b_in[i]]);
  }

  const long d_new_r = d_out_new_r + d_in_new_r;
  const long d_new_s = d_out_new_s + d_in_new_s;

  for (i=0; i<n_out; i++) {
    long Mtr, Mrt;
    if (Mu) {
      Mrt = *(long *)PyArray_GETPTR2(Mu, r, b_out[i]) - count_out[i];
    }
    else {
      compressed_get_single(Mc, r, b_out[i], &Mrt);
      Mrt -= count_out[i];
    }
    hash_val_t count = 0;

    if (b_out[i] == r) {
      Mrt = Nrr;
      Mtr = Nrr;
    }
    else if (b_out[i] == s) {
      Mrt = Nrs;
      Mtr = Nsr;
    }
    else if (0 == hash_search(h_in, b_out[i], &count)) {
      if (Mu) {
	Mtr = *(long *)PyArray_GETPTR2(Mu, b_out[i], r) - count;
      }
      else {
	compressed_get_single(Mc, b_out[i], r, &Mtr);
	Mtr -= count;
      }
    }
    else {
      if (Mu) {
	Mtr = *(long *)PyArray_GETPTR2(Mu, b_out[i], r);
      }
      else {
	compressed_get_single(Mc, b_out[i], r, &Mtr);
      }
    }

    prob_back += count_out[i] * (Mrt + Mtr + 1) / (B + degree_substitute(d, b_out[i], r, s, d_new_r, d_new_s));
  }

  for (i=0; i<n_in; i++) {
    long Mrt, Mtr;
    if (Mu) {
      Mtr = *(long *)PyArray_GETPTR2(Mu, b_in[i], r) - count_in[i];
    }
    else {
      compressed_get_single(Mc, b_in[i], r, &Mtr);
      Mtr -= count_in[i];
    }
    hash_val_t count = 0;

    if (b_in[i] == r) {
      Mtr = Nrr;
      Mrt = Nrr;
    }
    else if (b_in[i] == s) {
      Mtr = Nsr;
      Mrt = Nrs;
    }
    else if (0 == hash_search(h_out, b_in[i], &count)) {
      if (Mu) {
	Mrt = *(long *)PyArray_GETPTR2(Mu, r, b_in[i]) - count;
      }
      else {
	compressed_get_single(Mc, r, b_in[i], &Mrt);
	Mrt -= count;
      }
    }
    else {
      if (Mu) {
	Mrt = *(long *)PyArray_GETPTR2(Mu, r, b_in[i]);
      }
      else {
	compressed_get_single(Mc, r, b_in[i], &Mrt);
      }
    }

    prob_back += count_in[i] * (Mrt + Mtr + 1) / (B + degree_substitute(d, b_in[i], r, s, d_new_r, d_new_s));
  }


  hastings = prob_back / prob_fwd;
  
  if (delta_entropy > 10.0)
    delta_entropy = 10.0;
  else if (delta_entropy < -10.0)
    delta_entropy = -10.0;
  
  p_accept = exp(-beta * delta_entropy) * hastings;

  if (p_accept > 1.0)
    p_accept = 1.0;

  ret = Py_BuildValue("kkkdd", ni,r,s,delta_entropy,p_accept);

done:
  hash_destroy(h_in);
  hash_destroy(h_out);
  
  free(b_out);
  free(count_out);  
  free(b_in);
  free(count_in);

  Py_DECREF(ar_partition);
  Py_DECREF(ar_out_neighbors);
  Py_DECREF(ar_out_neighbor_weights);
  Py_DECREF(ar_in_neighbors);
  Py_DECREF(ar_in_neighbor_weights);
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_neighbor_weights);
  Py_DECREF(ar_d);
  Py_DECREF(ar_d_out);
  Py_DECREF(ar_d_in);

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

/*
 * Args: partition, vertex_id_start, vertex_id_end, neighbors[vid], weights[vid]
 * Version for a compressed array.
 */
static PyObject* rebuild_M_compressed(PyObject *self, PyObject *args)
{
  PyObject *obj_partition, *obj_neighbors, *obj_weights, *obj_M, *obj_d_out, *obj_d_in;
  uint64_t vid_start, vid_end;

  if (!PyArg_ParseTuple(args, "OllOOOOO", &obj_partition, &vid_start, &vid_end, &obj_neighbors, &obj_weights, &obj_M, &obj_d_out, &obj_d_in)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj_M, "compressed_array");
  const PyObject *ar_partition = PyArray_FROM_OTF(obj_partition, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_neighbors = PyArray_FROM_OTF(obj_neighbors, NPY_LONG, NPY_IN_ARRAY);
  const PyObject *ar_weights = PyArray_FROM_OTF(obj_weights, NPY_LONG, NPY_IN_ARRAY);
  PyObject *ar_d_in = PyArray_FROM_OTF(obj_d_in, NPY_LONG, NPY_IN_ARRAY);
  PyObject *ar_d_out = PyArray_FROM_OTF(obj_d_out, NPY_LONG, NPY_IN_ARRAY);

  const uint64_t *partition = (const uint64_t *) PyArray_DATA(ar_partition);
  const uint64_t *neighbors = (const uint64_t *) PyArray_DATA(ar_neighbors);
  const uint64_t *weights = (const uint64_t *) PyArray_DATA(ar_weights);

  int64_t *d_in = (int64_t *) PyArray_DATA(ar_d_in);
  int64_t *d_out = (int64_t *) PyArray_DATA(ar_d_out);

  long i;
  long n = (long) PyArray_DIM(ar_neighbors, 0);

  uint64_t k1 = partition[vid_start];
  for (i=0; i<n; i++) {
    uint64_t k2 = partition[neighbors[i]];
    uint64_t w = weights[i];

    compressed_accum_single(x, k1, k2, w);

    d_in[k2] +=w;
    d_out[k1] += w;
  }

  Py_DECREF(ar_partition);
  Py_DECREF(ar_neighbors);
  Py_DECREF(ar_weights);
  Py_DECREF(ar_d_in);
  Py_DECREF(ar_d_out);
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
  size_t *used = calloc(SHARED_MAX_POOLS, sizeof(size_t));
  size_t *avail = calloc(SHARED_MAX_POOLS, sizeof(size_t));

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
  long pool_id, n_items;

  if (!PyArg_ParseTuple(args, "ll", &pool_id, &n_items)) {
    return NULL;
  }

  shared_reserve(pool_id, n_items);
  Py_RETURN_NONE;
}

static PyObject* hash_pointer(PyObject *self, PyObject *args)
{
  PyObject *obj, *obj_i, *obj_j;
  long i;

  if (!PyArg_ParseTuple(args, "OOO", &obj, &obj_i, &obj_j)) {
    return NULL;
  }

  struct compressed_array *x = PyCapsule_GetPointer(obj, "compressed_array");

  i = PyLong_AsLongLong(obj_i);

  struct hash_outer *ho = &x->rows[i];

  long hash_outer_ptr = (long) ho;
  long hash_inner_ptr = (long) ho->h;

  PyObject *ret = Py_BuildValue("ll", hash_outer_ptr, hash_inner_ptr);
  return ret;
}

static PyObject *info(PyObject *self, PyObject *args)
{
  _Atomic(struct hash_outer) h;
  atomic_ulong i;

  const char *msg_atomic_hash = "false";
  if (atomic_is_lock_free(&h)) {
    msg_atomic_hash = "true";
  }

  const char *msg_atomic_ulong = "false";
  if (atomic_is_lock_free(&i)) {
    msg_atomic_ulong = "true";
  }


  char *msg;
  if (asprintf(&msg, "Compiler: %s hash_entry_size: %s atomic_hash_struct: %s atomic_ulong: %s",
	       __VERSION__, HASH_IMPL_DESCRIPTION, msg_atomic_hash, msg_atomic_ulong) < 0) {
    msg = NULL;
  }
  PyObject *ret = Py_BuildValue("s", msg);
  free(msg);
  return ret;
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
   { "take_dict_ref", take_dict_ref, METH_VARARGS, "Take items along an axis in dict form." },
   { "accum_dict", accum_dict, METH_VARARGS, "Add to items in a dict slice." },
   { "keys_values_dict", keys_values_dict, METH_VARARGS, "Get keys and values from a dict slice." },
   { "print_dict", print_dict, METH_VARARGS, "Print items along an axis in dict form." },
   { "empty_dict", empty_dict, METH_VARARGS, "New row dict." },
   { "getitem_dict", getitem_dict, METH_VARARGS, "Look up in a row dict." },
   { "set_dict", set_dict, METH_VARARGS, "Set a row dict." },
   { "eq_dict", eq_dict, METH_VARARGS, "Compare two dicts." },
   { "copy_dict", copy_dict, METH_VARARGS, "Copy a row dict." },
   { "sum_dict", sum_dict, METH_VARARGS, "Sum the values of a dict." },
   { "sanity_check", sanity_check, METH_VARARGS, "Run a sanity check." },
   { "dict_entropy_row", dict_entropy_row, METH_VARARGS, "Compute part of delta entropy for a row entry." },
   { "dict_entropy_row_excl", dict_entropy_row_excl, METH_VARARGS, "Compute part of delta entropy for a row entry." },
   { "inplace_apply_movement_compressed_interblock_matrix", inplace_apply_movement_compressed_interblock_matrix, METH_VARARGS, "Move node from block r to block s and apply changes to interblock edge count matrix, and other algorithm state." },
   { "inplace_apply_movement_uncompressed_interblock_matrix", inplace_apply_movement_uncompressed_interblock_matrix, METH_VARARGS, "Move node from block r to block s and apply changes to interblock edge count matrix, and other algorithm state." },
   { "compute_new_rows_cols_interblock", compute_new_rows_cols_interblock, METH_VARARGS, "" },
   { "blocks_and_counts", blocks_and_counts_py, METH_VARARGS, "" },
   { "combine_key_value_pairs", combine_key_value_pairs_py, METH_VARARGS, "" },
   { "hastings_correction", hastings_correction_py, METH_VARARGS, "" },
   { "propose_new_partition", propose_new_partition_py, METH_VARARGS, "" },
   { "propose_nodal_movement", propose_nodal_movement, METH_VARARGS, "" },
   { "propose_block_merge", propose_block_merge, METH_VARARGS, "" },
   { "rebuild_M", rebuild_M, METH_VARARGS, "" },
   { "rebuild_M_compressed", rebuild_M_compressed, METH_VARARGS, "" },
   { "nonzero_count", nonzero_count, METH_VARARGS, "" },
   { "shared_memory_report", shared_memory_report, METH_VARARGS, "" },
   { "shared_memory_query", shared_memory_query, METH_VARARGS, "" },
   { "shared_memory_reserve", shared_memory_reserve, METH_VARARGS, "" },
   { "hash_pointer", hash_pointer, METH_VARARGS, "" },
   { "info", info, METH_NOARGS, "Get implementation info." },
   { "seed", seed, METH_VARARGS, "Set random number seed from system urandom." },
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
