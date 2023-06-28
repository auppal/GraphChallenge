#include "shared_mem.h"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#if 1
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

static size_t bytes_cnt = 0;
typedef struct circ_buf_t {
	ssize_t head;
	ssize_t tail;
	size_t count;
	size_t len;
	size_t size;
	char *buf;
} circ_buf_t;

static inline int circ_init(circ_buf_t *b, size_t len, size_t size);
static inline int circ_enq(circ_buf_t *b, const void *elm);
static inline int circ_deq(circ_buf_t *b, void *elm);
static inline const void *circ_peek(circ_buf_t *b, size_t index);
static inline size_t circ_cnt(circ_buf_t *b);
static inline void circ_free(circ_buf_t *b);

static inline int circ_init(circ_buf_t *b, size_t len, size_t size)
{
	b->buf = malloc((len + 1) * size);

	if (!b->buf) {
		return -1;
	}

	b->len = (len + 1);
	b->size = size;
	b->tail = 0;
	b->head = 0;
	b->count = 0;

	return 0;
}

static inline int circ_clear(circ_buf_t *b)
{
	b->tail = 0;
	b->head = 0;
	b->count = 0;

	return 0;
}

static inline int circ_enq(circ_buf_t *b, const void *elm)
{
	ssize_t tail = (b->tail + 1) % b->len;

	if (tail == b->head) {
		return -1;
	}

	memcpy(b->buf + b->tail * b->size, elm, b->size);
	b->tail = tail;
	b->count++;
	return 0;
}

static inline int circ_deq(circ_buf_t *b, void *elm)
{
	if (b->tail == b->head) {
		return -1;
	}

	if (elm) {
		memcpy(elm, &b->buf[b->head * b->size], b->size);
	}

	b->head = (b->head + 1) % b->len;
	b->count--;
	return 0;
}

static inline const void *circ_peek(circ_buf_t *b, size_t index)
{
	if (index >= b->count)
		return NULL;

	ssize_t i = (b->head + index) % b->len;
	return &b->buf[i * b->size];
}

static inline size_t circ_cnt(circ_buf_t *b)
{
	return b->count;
}

static inline void circ_free(circ_buf_t *b)
{
	if (b) {
		free(b->buf);
	}
}

static inline int circ_resize(circ_buf_t *b, size_t new_len)
{
	char *buf = realloc(b->buf, new_len * b->size);

	if (buf) {
		b->len = new_len;
		b->buf = buf;
		return 0;
	}

	return -1;
}

static inline int int_log2(size_t x)
{
	if (x <= 64) {
		return 6;
	}

	return 1 + (size_t) log2(x);
}

struct pool_info {
	circ_buf_t q;
	size_t alloc_items; /* Next allocation size in terms of
			     * items. */
};

#define MAX_POOLS (64)
struct pool_info pools[MAX_POOLS];

static int shared_mem_initialized = 0;
static pid_t parent_pid;

static void *shared_memory_get(size_t size_bytes)
{
	if (!shared_mem_initialized) {
		shared_mem_initialized = 1;
		parent_pid = getpid();
	}

	if (getpid() != parent_pid) {
		fprintf(stderr, "Failed to allocate %ld bytes: parent_pid %d mypid %d shared memory allocation only supported from parent!\n", size_bytes, parent_pid, getpid());
		return NULL;
	}

	/* Map the object into the caller's address space. */
	void *rc = mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
			MAP_ANONYMOUS | MAP_SHARED, -1, 0);

	if (rc == MAP_FAILED) {
		perror("mmap");
		return NULL;
	}

	bytes_cnt += size_bytes;
	return rc;
}


void *shared_malloc(size_t nbytes)
{
	size_t page_size = 4096;
	size_t lg2 = int_log2(nbytes);
	size_t np2 = 1 << lg2;
	struct pool_info *p = &pools[lg2];
	void *addr = NULL;

	// fprintf(stderr, "circ bytes %ld np2 %ld lg2 %ld\n", nbytes, np2, lg2);

	if (!shared_mem_initialized) {
		shared_mem_initialized = 1;
		parent_pid = getpid();
	}

	if (getpid() != parent_pid) {
		fprintf(stderr, "Failed to allocate %ld bytes: parent_pid %d mypid %d shared memory allocation only supported from parent!\n", nbytes, parent_pid, getpid());
		return NULL;
	}

	if (circ_deq(&p->q, &addr) < 0) {
		size_t i;
		size_t alloc_items = 1.25 * p->alloc_items;

		if (alloc_items == 0) {
			alloc_items = 4;
			// fprintf(stderr, "circ initialize with %ld items of size %ld\n", alloc_items, np2);
			if (circ_init(&p->q, alloc_items, sizeof(void *)) < 0) {
				fprintf(stderr, "circ init failed!\n");
				return NULL;
			}
		}
		else {
			// fprintf(stderr, "circ resize with %ld items of size %ld\n", alloc_items, np2);
			circ_resize(&p->q, (p->q.len - 1) + alloc_items);
		}

		void *base = shared_memory_get(np2 * alloc_items);

		if (base < 0) {
			perror("shared_memory_get");
			return NULL;
		}

		for (i=0; i<alloc_items; i++) {
			void *x = (void * ) ((uintptr_t) base + (i * np2));
			// fprintf(stderr, "circ enq %p\n", x);
			if (circ_enq(&p->q, &x) < 0) {
				fprintf(stderr, "circ enq failed!\n");
				return NULL;
			}
		}

		p->alloc_items = alloc_items;
		circ_deq(&p->q, &addr);
		// fprintf(stderr, "circ initialized and returning %p\n", addr);
	}

	return addr;
}

void *shared_calloc(size_t nmemb, size_t size)
{
	void *p = shared_malloc(nmemb * size);
	memset(p, 0, nmemb * size);
	return p;
}

void shared_free(void *addr, size_t nbytes)
{
	size_t lg2 = int_log2(nbytes);

	if (lg2 > MAX_POOLS) {
		fprintf(stderr, "Warning: Invalid pointer in shared_free!\n");
		return;
	}

	struct pool_info *pool = &pools[lg2];
	circ_enq(&pool->q, &addr);
}

void shared_print_report()
{
	size_t i;

	fflush(stdout);

	fprintf(stdout, "Bytes allocated: %ld\n", bytes_cnt);
	
	for (i=6; i<32; i++) {
		ssize_t used = pools[i].q.len - 1;
		ssize_t free = circ_cnt(&pools[i].q);

		if (used < 0) {
			used = 0;
		}
		
		fprintf(stdout, "  Pool %2ld blocks used %5ld free %5ld used bytes %10ld free %10ld\n", i, used, free, used * (1 << i), free * (1 << i));
	}

	fflush(stdout);
}

#else
#include "slab_alloc.h"

#define MAX_CACHES 100
static struct kmem_cache *caches[MAX_CACHES];
static int shared_mem_initialized = 0;
static pid_t parent_pid;

void *shared_malloc(size_t nbytes)
{
#if 0
  if (nbytes != 560) {
    return malloc(nbytes);
  }
#endif

#if 1
  if (!shared_mem_initialized) {
    shared_mem_initialized = 1;
    parent_pid = getpid();
  }

  if (getpid() != parent_pid) {
    fprintf(stderr, "Child caches[0] = %p\n", caches[0]);
    fprintf(stderr, "Failed to allocate %ld bytes: parent_pid %d mypid %d shared memory allocation only supported from parent!\n", nbytes, parent_pid, getpid());
    return NULL;
  }
  else {
    //fprintf(stderr, "Parent caches[0] = %p\n", caches[0]);
  }
#endif  
  
  struct kmem_cache *cp = NULL;
  int i;
  for (i=0; i<MAX_CACHES; i++) {
    if (!caches[i])
      break;

    if (caches[i]->size == nbytes) {
      cp = caches[i];
      break;
    }
  }

  if (!cp && i < MAX_CACHES) {
    fprintf(stderr, "create kmem for size %ld\n", nbytes);
    cp = kmem_cache_create("", nbytes, 0);
    caches[i++] = cp;
    fprintf(stderr, "cp[%d] is %ld\n", i, cp->size);
  }

  if (!cp) {
    return NULL;
  }

  void *p = kmem_cache_alloc(cp, 0);
#if 0
  fprintf(stderr, "shared_malloc return %p from pool %p size %d\n", p, cp, nbytes);
#endif  
  return p;
}

void *shared_calloc(size_t nmemb, size_t size)
{
  void *p = shared_malloc(nmemb * size);
  memset(p, 0, nmemb * size);
  return p;
}

void shared_free(void *p, size_t nbytes)
{
  struct kmem_cache *cp = NULL;
  int i;
  for (i=0; i<MAX_CACHES; i++) {
    if (!caches[i])
      break;

    if (caches[i]->size == nbytes) {
      cp = caches[i];
      break;
    }
  }

#if 0
  fprintf(stderr, "shared_free return %p to pool %p\n", p, cp);
#endif
  
  if (!cp) {
    fprintf(stderr, "ERROR: Pool for size %ld not found.\n", nbytes);
  }
  else {
    kmem_cache_free(cp, p);
  }
}


#endif
