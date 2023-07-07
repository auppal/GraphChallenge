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
#include <stdatomic.h>
#include <errno.h>

#define DEBUG_PRINTF (0)

static void *shared_memory_mmap(size_t size_bytes);
static void *shared_memory_get(size_t size_bytes);
static int shared_memory_use_mmap = 1;
static size_t bytes_cnt = 0;
typedef struct circ_buf_t {
	_Atomic(size_t) head;
	_Atomic(size_t) tail;
	_Atomic(size_t) count;
	size_t buf_len;
	size_t size;
	char *buf;
} circ_buf_t;
static const size_t page_size = 4096;

static inline int circ_init(circ_buf_t *b, size_t len, size_t size);
static inline int circ_enq(circ_buf_t *b, const void *elm);
static inline int circ_deq(circ_buf_t *b, void *elm);
static inline const void *circ_peek(circ_buf_t *b, size_t index);
static inline size_t circ_cnt(circ_buf_t *b);
static inline void circ_free(circ_buf_t *b);

static inline int circ_init(circ_buf_t *b, size_t len, size_t size)
{
#if DEBUG_PRINTF
	fprintf(stderr, "circ_init: len %ld size %ld\n", len, size);
#endif
	if (!b) {
		fprintf(stderr, "circ_init: Invalid value b = %p\n", b);
		return -1;
	}
	
	size_t alloc_pages;

	if (shared_memory_use_mmap) {
		alloc_pages = ((len + 1) * size  + page_size - 1) / page_size;
		b->buf = shared_memory_get(alloc_pages * page_size);
	}
	else {
		alloc_pages = 0;
		b->buf = malloc((len + 1) * size);
		memset(b->buf, 0xfe, (len + 1) * size); /* XXX remove */
	}

	if (!b->buf) {
		return -1;
	}
	
	b->buf_len = (len + 1);
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
	size_t tail = b->tail;	

	for (;;) {
		size_t head = b->head;
		size_t next = (tail + 1) % b->buf_len;

		if (next == b->head) {
			errno = ENOBUFS;
			return -1;
		}

		_Bool rc = atomic_compare_exchange_strong(&b->tail, &tail, next);

		if (rc) {
			b->count++;
			memcpy(b->buf + tail * b->size, elm, b->size);			
			break;
		}
	}

	return 0;
}

static inline int circ_deq(circ_buf_t *b, void *elm)
{
	if (!elm) {
		errno = EINVAL;
		return -1;
	}

	size_t head = b->head;

	for (;;) {
		size_t tail = b->tail;
		size_t next = (head + 1) % b->buf_len;
		
		if (tail == head) {
			errno = EAGAIN;
			return -1;
		}

		_Bool rc = atomic_compare_exchange_strong(&b->head, &head, next);

		if (rc) {
			b->count--;
			memcpy(elm, &b->buf[head * b->size], b->size);
			break;
		}
	}

	return 0;
}

static inline const void *circ_peek(circ_buf_t *b, size_t index)
{
	if (index >= b->count)
		return NULL;

	ssize_t i = (b->head + index) % b->buf_len;
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

#if 0
static inline int circ_resize(circ_buf_t *b, size_t new_len)
{
	
	fprintf(stderr, "Enter circ_resize: old_len %ld new_len %ld\n", b->buf_len - 1, new_len);
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

	size_t old_size = b->alloc_size;
	void *old_buf = b->buf;

	*b = q;

	if (shared_memory_use_mmap) {
		// munmap(old_buf, old_size);
		// shared_free(old_buf, old_size); /* NO! */
	}
	else {
		free(old_buf);
	}

	return 0;
}
#endif

static inline int int_log2(size_t x)
{
	if (x <= 4096) {
		return 12;
	}

	return 1 + (size_t) log2(x);
}

struct pool_info {
	circ_buf_t q;
};


struct huge_info {
	size_t huge_size;
	_Atomic(size_t) huge_offset;
	char *huge_base;
};

static struct pool_info **p_pools;
static struct huge_info *huge;
static int shared_mem_initialized = 0;
static pid_t parent_pid;


static void *shared_memory_mmap(size_t size_bytes)
{
	if (getpid() != parent_pid) {
		fprintf(stderr, "shared_memory_mmap: Failed to allocate %ld bytes: child shared memory not supported!\n", size_bytes);
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

static void *shared_memory_get(size_t size_bytes)
{
#if DEBUG_PRINTF
	if (getpid() != parent_pid) {	
		fprintf(stderr, "shared_memory_get: Allocate %ld bytes in child %d!\n", size_bytes, getpid());
	}
	else {
		fprintf(stderr, "shared_memory_get: Allocate %ld bytes in parent %d!\n", size_bytes, getpid());		
	}
#endif
	
	if (size_bytes & 0xfff) {
		fprintf(stderr, "shared_memory_get: allocation %ld is not divisible by page size!\n", size_bytes);
		errno = EINVAL;
		return NULL;
	}

	size_t offset = huge->huge_offset, new_offset;
	do {
		new_offset = offset + size_bytes;

		if (new_offset >= huge->huge_size) {
			fprintf(stderr, "shared_memory_get: Out of storage\n");
			errno = ENOBUFS;
			return NULL;
		}
	}
	while (!atomic_compare_exchange_strong(&huge->huge_offset, &offset, new_offset));

	char *p = huge->huge_base + offset;

#if DEBUG_PRINTF
	fprintf(stderr, "shared_memory_get: huge_offset %ld huge_size %ld return %p\n", huge->huge_offset, huge->huge_size, p);
#endif	
	
	return p;
}

void shared_init()
{
	if (shared_mem_initialized) {
		return;
	}
	
	shared_mem_initialized = 1;
	parent_pid = getpid();
	p_pools = shared_memory_mmap(SHARED_MAX_POOLS * sizeof(struct pool_info *));

#if DEBUG_PRINTF
	fprintf(stderr, "Initialized p_pools to %p\n", p_pools);
#endif	

	if (getenv("USE_MMAP") && !strcmp(getenv("USE_MMAP"), "0")) {
		shared_memory_use_mmap = 0;
		fprintf(stderr, "Disable mmap usage\n");
	}

	huge = shared_memory_mmap(sizeof(struct huge_info));
	
	size_t page_size = 4096;
	huge->huge_size = 1024 * 10000000ul;
	huge->huge_base = shared_memory_mmap(huge->huge_size);
	huge->huge_offset = 0;
}


void *shared_malloc(size_t nbytes)
{
	size_t lg2 = int_log2(nbytes);
	size_t np2 = 1 << lg2;
	struct pool_info *p;
	void *addr = NULL;
	size_t page_size = 4096;
	size_t initial_items = page_size / sizeof(void *) - 1;

#if DEBUG_PRINTF	
	fprintf(stderr, "shared_malloc nbytes %ld np2 %ld lg2 %ld\n", nbytes, np2, lg2);
#endif
	shared_init();

	p = p_pools[lg2];

	if (!p || circ_deq(&p->q, &addr) < 0) {
		size_t i;
		size_t alloc_items = initial_items, current_capacity = 0, fill_items = initial_items;

		if (p) {
			current_capacity = (p->q.buf_len - 1);			
			alloc_items = 2 * current_capacity;
#if DEBUG_PRINTF			
			fprintf(stderr, "XXX current capacity is %ld\n", current_capacity);
#endif			
			fill_items = alloc_items - current_capacity;
		}
			
		struct pool_info *next = shared_memory_get(((sizeof(struct pool_info) + 4095) / 4096) * 4096);

#if DEBUG_PRINTF
		fprintf(stderr, "shared_malloc: allocated pool %p\n", next);
			
		fprintf(stderr,
			"PID %d: circ initialize next %p pool %ld with %ld entries of size %ld\n",
			getpid(), next, lg2, alloc_items, np2);
#endif	
		if (circ_init(&next->q, alloc_items, sizeof(void *)) < 0) {
			perror("circ_init");
			return NULL;
		}

#if DEBUG_PRINTF
		fprintf(stderr, "XXX next capacity is %ld\n", next->q.buf_len - 1);
		fprintf(stderr, "     pool %ld mmap %ld bytes\n", lg2, np2 * fill_items);
#endif		
		void *base = shared_memory_get(np2 * fill_items);

		if (base == 0) {
			fprintf(stderr, "shared_malloc: Failed to add %ld items, %ld bytes into pool %ld\n", fill_items, np2 * fill_items, lg2);
			return NULL;
		}

#if DEBUG_PRINTF
		fprintf(stderr, "PID %d: shared_malloc: fill %ld items into next pool %ld\n", getpid(), fill_items, lg2);
#endif		
		
		for (i=0; i<fill_items; i++) {
			void *x = (void * ) ((uintptr_t) base + (i * np2));
			// fprintf(stderr, "circ enq %ld %p\n", i, x);
			if (circ_enq(&next->q, &x) < 0) {
				fprintf(stderr, "circ enq failed!\n");
				return NULL;
			}
		}

		circ_deq(&next->q, &addr);
		p_pools[lg2] = next;
	}

#if DEBUG_PRINTF	
	fprintf(stderr, "shared_malloc pid %d from pool %ld return %p\n", getpid(), lg2, addr);
#endif	
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
#if DEBUG_PRINTF
	fprintf(stderr, "shared_free enqueue into pool %ld ptr %p\n", lg2, addr);
#endif	
	struct pool_info *pool = p_pools[lg2];
	circ_enq(&pool->q, &addr);
}

void shared_query(size_t id, size_t *p_n_used, size_t *p_n_free)
{
	shared_init();
	*p_n_free = circ_cnt(&p_pools[id]->q);
	*p_n_used = p_pools[id]->q.buf_len - 1 - *p_n_free;
}

int shared_reserve(size_t lg2, size_t n_items)
{
	struct pool_info *p = p_pools[lg2];
	size_t capacity = (p->q.buf_len - 1);
	
	if (lg2 > SHARED_MAX_POOLS) {
		return -1;
	}

	if (n_items <= capacity) {
		return 0;
	}
#if DEBUG_PRINTF	
	fprintf(stderr, "shared_reserve: resize pool queue %ld from %ld to %ld items\n", lg2, capacity, n_items);
#endif	
	return -1;
#if 0
	/* XXX TODO Remove */
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
#endif
	return 0;
}

void shared_print_report()
{
	size_t i, max_nonempty = 0;

	shared_init();

	fprintf(stdout, "Huge base %p offset %ld\n", huge->huge_base, huge->huge_offset);
	
	for (i=SHARED_MAX_POOLS; i>0; i--) {
		if (p_pools[i - 1] != NULL) {
			max_nonempty = i;
			break;
		}
	}

	fflush(stdout);

	fprintf(stdout, "Bytes allocated: %ld\n", bytes_cnt);
	
	for (i=6; i<max_nonempty; i++) {
		ssize_t free, used;
		void *buf;
		if (p_pools[i]) {
			free = circ_cnt(&p_pools[i]->q);
			used = p_pools[i]->q.buf_len - 1 - free;
			buf = p_pools[i]->q.buf;

			if (used < 0) {
				used = 0;
			}
		}
		else {
			free = 0;
			used = 0;
			buf = NULL;
		}

		fprintf(stdout, "  Pool %2ld buf %p blocks used %5ld free %5ld used bytes %10ld free %10ld\n", i, buf, used, free, used * (1 << i), free * (1 << i));
	}

	fflush(stdout);
}
