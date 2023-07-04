#define _GNU_SOURCE
#include "shared_mem.h"
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <sys/mman.h>

static void *shared_memory_get(size_t size_bytes);
static int shared_memory_use_mmap = 1;
static size_t bytes_cnt = 0;
typedef struct circ_buf_t {
	size_t head;
	size_t tail;
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
	fprintf(stderr, "circ_init: len %ld size %ld\n", len, size);

	if (shared_memory_use_mmap) {
		b->buf = shared_memory_get((len + 1) * size);
	}
	else {
		b->buf = malloc((len + 1) * size);
		memset(b->buf, 0xfe, (len + 1) * size); /* XXX remove */
	}

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
	size_t tail = (b->tail + 1) % b->len;

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
#if 1
	fprintf(stderr, "Enter circ_resize: old_len %ld new_len %ld\n", b->len - 1, new_len);
	circ_buf_t q;
	
	if (circ_init(&q, new_len, sizeof(void *)) < 0) {
		perror("circ_init failed");
		return -1;
	}

	size_t i;
	void *addr;
	for (i=0; i<b->count; i++) {
		if (circ_deq(b, &addr) < 0) {
			fprintf(stderr, "circ_resize: deq failed\n");
			return -1;
		}
		circ_enq(&q, &addr);
	}

	size_t old_size = b->size * b->len;
	void *old_buf = b->buf;

	*b = q;

	if (shared_memory_use_mmap) {
		munmap(old_buf, old_size);
	}
	else {
		free(old_buf);
	}

	return 0;
#else	
	size_t old_size = b->size * b->len;
	size_t new_size = b->size * (new_len + 1);
	char *buf;

	if (new_size <= old_size) {
		return 0;
	}

#if 0
	static int fail_counter = 20;
	if (fail_counter-- == 0) {
		fprintf(stderr, "circ_resize: force failure\n");
		return -1;
	}
#endif	

	fprintf(stderr, "Enter circ_resize: old_size %ld new_size %ld bytes\n", old_size, new_size);
	if (shared_memory_use_mmap) {
		buf = mmap(NULL, new_size, PROT_READ | PROT_WRITE,
				MAP_ANONYMOUS | MAP_SHARED, -1, 0);

		if (buf == MAP_FAILED) {
			return -1;
		}

		if (b->head <= b->tail) {
			memcpy(buf, b->buf + b->head, (b->tail - b->head) * b->size);
			assert(b->tail - b->head == b->count);
		}
		else {
			memcpy(buf, b->buf, b->tail * b->size);
			memcpy(buf, b->buf + b->head, (b->len - b->head) * b->size);
			fprintf(stderr, "head %ld tail %ld len %ld count %ld\n", b->head, b->tail, b->len, b->count);
			assert(b->tail + b->len - b->head == b->count);
		}

		munmap(b->buf, old_size);
	}
	else {
		buf = malloc(new_size);
		memset(buf, 0xfe, new_size);

		if (!buf) {
			perror("circ_resize");		
			return -1;
		}
		
		if (b->head <= b->tail) {
			memcpy(buf, b->buf + b->head, (b->tail - b->head) * b->size);
		}
		else {
			memcpy(buf, b->buf, b->tail * b->size);
			memcpy(buf, b->buf + b->head, (b->len - b->head) * b->size);
		}

		free(b->buf);
	}

	b->head = 0;
	b->tail = b->count;
	b->len = new_len + 1;
	b->buf = buf;
	return 0;
#endif
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

struct pool_info *pools;

static int shared_mem_initialized = 0;
static pid_t parent_pid;

static void *shared_memory_get(size_t size_bytes)
{
	if (getpid() != parent_pid) {
		fprintf(stderr, "shared_malloc: Failed to allocate %ld bytes: child shared memory not supported!\n", size_bytes);
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

void shared_init()
{
	if (!shared_mem_initialized) {
		shared_mem_initialized = 1;
		parent_pid = getpid();
		pools = shared_memory_get(SHARED_MAX_POOLS * sizeof(struct pool_info));
		fprintf(stderr, "Initialized pools to %p\n", pools);
		if (getenv("USE_MMAP") && !strcmp(getenv("USE_MMAP"), "0")) {
			shared_memory_use_mmap = 0;
			fprintf(stderr, "Disable mmap usage\n");
		}
	}
}

void *shared_malloc(size_t nbytes)
{
	size_t lg2 = int_log2(nbytes);
	size_t np2 = 1 << lg2;
	struct pool_info *p;
	void *addr = NULL;
	size_t page_size = 4096;
	size_t initial_items = page_size / sizeof(void *) - 1;

	// fprintf(stderr, "circ bytes %ld np2 %ld lg2 %ld\n", nbytes, np2, lg2);

	shared_init();

	p = &pools[lg2];

	if (circ_deq(&p->q, &addr) < 0) {
		size_t i;
		size_t alloc_items = 2 * p->alloc_items;

		if (getpid() != parent_pid) {
			fprintf(stderr, "shared_malloc: resize pool %ld failed: child shared memory not supported!\n", lg2);
			return NULL;
		}

		if (alloc_items == 0) {
			alloc_items = initial_items;

			fprintf(stderr,
				"circ initialize pool %ld with %ld entries of size %ld\n",
				lg2, initial_items, np2);

			if (circ_init(&p->q, initial_items, sizeof(void *)) < 0) {
				perror("circ_init");
				return NULL;
			}
		}
		else {
			size_t capacity = (p->q.len - 1);

			fprintf(stderr, "circ resize pool %ld from %ld to %ld items of size %ld\n",
				lg2,
				capacity,
				capacity + alloc_items, np2);
				
			if (circ_resize(&p->q, capacity + alloc_items) < 0) {
				perror("circ_resize");
				return NULL;
			}
		}
		
		fprintf(stderr, "     pool %ld mmap %ld bytes\n", lg2, np2 * alloc_items);

		void *base = shared_memory_get(np2 * alloc_items);

		if (base == 0) {
			fprintf(stderr, "shared_malloc: Failed to add %ld items, %ld bytes into pool %ld\n", alloc_items, np2 * alloc_items, lg2);
			return NULL;
		}

		for (i=0; i<alloc_items; i++) {
			void *x = (void * ) ((uintptr_t) base + (i * np2));
			// fprintf(stderr, "circ enq %ld %p\n", i, x);
			if (circ_enq(&p->q, &x) < 0) {
				fprintf(stderr, "circ enq failed!\n");
				return NULL;
			}
		}

		p->alloc_items = alloc_items;
		circ_deq(&p->q, &addr);
	}

	fprintf(stderr, "shared_malloc pid %d return %p\n", getpid(), addr);
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

	if (lg2 > SHARED_MAX_POOLS) {
		fprintf(stderr, "Warning: Invalid pointer in shared_free!\n");
		return;
	}

	struct pool_info *pool = &pools[lg2];
	circ_enq(&pool->q, &addr);
}

void shared_query(size_t id, size_t *p_n_used, size_t *p_n_free)
{
	shared_init();
	*p_n_free = circ_cnt(&pools[id].q);
	*p_n_used = pools[id].q.len - 1 - *p_n_free;
}

int shared_reserve(size_t lg2, size_t n_items)
{
	struct pool_info *p = &pools[lg2];
	size_t capacity = (p->q.len - 1);
	
	if (lg2 > SHARED_MAX_POOLS) {
		return -1;
	}

	if (n_items <= capacity) {
		return 0;
	}
	
	fprintf(stderr, "shared_reserve: resize pool %ld from %ld to %ld items\n", lg2, capacity, n_items);

	if (circ_resize(&p->q, n_items) < 0) {
		return -1;
	}

	/* Now go and actually fill the new items */
	size_t i, np2 = 1 << lg2;
	size_t alloc_items = n_items - capacity;

	fprintf(stderr, "     pool %ld mmap %ld bytes\n", lg2, np2 * alloc_items);

	void *base = shared_memory_get(np2 * alloc_items);

	if (base == 0) {
		fprintf(stderr, "shared_malloc: Failed to add %ld items, %ld bytes into pool %ld\n",
			alloc_items, np2 * alloc_items, lg2);
		return -1;
	}

	for (i=0; i<alloc_items; i++) {
		void *x = (void * ) ((uintptr_t) base + (i * np2));
		if (circ_enq(&p->q, &x) < 0) {
			fprintf(stderr, "circ enq failed!\n");
			return -1;
		}
	}

	return 0;
}

void shared_print_report()
{
	size_t i, max_nonempty = 0;

	shared_init();

	for (i=SHARED_MAX_POOLS; i>0; i--) {
		if (pools[i - 1].q.buf != NULL) {
			max_nonempty = i;
			break;
		}
	}

	fflush(stdout);

	fprintf(stdout, "Bytes allocated: %ld\n", bytes_cnt);
	
	for (i=6; i<max_nonempty; i++) {
		ssize_t free = circ_cnt(&pools[i].q);
		ssize_t used = pools[i].q.len - 1 - free;

		if (used < 0) {
			used = 0;
		}
		
		fprintf(stdout, "  Pool %2ld buf %p blocks used %5ld free %5ld used bytes %10ld free %10ld\n", i, pools[i].q.buf, used, free, used * (1 << i), free * (1 << i));
	}

	fflush(stdout);
}
