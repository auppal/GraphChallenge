/* 
 * slab allocator written by Ahsen Uppal
 * for csci6907 homework 1 -- spring 2014.
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h> /* memset */
#include <errno.h>
#include "slab_alloc.h"

/* for shmget */
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>


#define KMEM_DEBUG 0

/* Doubly-linked list for linking slabs -- a slab contains these structs. */
struct dl_node
{
	struct slab *data;
	struct dl_node *prev;
	struct dl_node *next;
};

/* Singly-linked list for linking buffers (internal to those buffers). */
struct sl_node
{
	struct sl_node *next;
};


int dl_isempty(struct dl_node *head)
{
	return head == NULL;
}


/* Insert an element to the front of the list and return the new head. */
struct dl_node * __attribute__((warn_unused_result)) dl_prepend(struct dl_node *head, struct dl_node *n)
{
	n->prev = NULL;
	n->next = head;

	if (n->next) {
		n->next->prev = n;
	}

	return n;
}

/* Insert an element to the front of the list and return the new head. */
struct sl_node * __attribute__((warn_unused_result)) sl_prepend(struct sl_node *head, struct sl_node *n)
{
	n->next = head;
	return n;
}


/* Remove an item from the front of the list, returning the new head. */
struct sl_node * __attribute__((warn_unused_result)) sl_remove_head(struct sl_node *head)
{
	struct sl_node *next = head->next;
	return next;
}


/* Remove an item from the front of the list, returning the new head. */
struct dl_node * __attribute__((warn_unused_result)) dl_remove_head(struct dl_node *head)
{
	struct dl_node *next = head->next;

	head->prev = NULL;
	head->next = NULL;

	return next;
}

/* Unlink an item from within a list, returning the next element. */
struct dl_node * __attribute__((warn_unused_result)) dl_remove(struct dl_node *node)
{
	struct dl_node *prev = node->prev;
	struct dl_node *next = node->next;

	if (prev) {
		prev->next = next;
	}

	if (next) {
		next->prev = prev;
	}

	node->prev = NULL;
	node->data = NULL;
	node->next = NULL;

	return next;
}

static int slab_id;
#define MAGIC_PATTERN (0xdeadbaad)

struct slab
{
	int id;
	void *base;
	size_t chunk_size;
	size_t n_chunks;
	unsigned int refcnt;
	struct sl_node *buf_freelist;
	struct dl_node dl;
	unsigned int magic;
};


int slab_get_count()
{
	return slab_id;
}

void *slab_idx_to_chunk(struct slab *s, int i)
{
	return (unsigned char *) s->base + s->chunk_size * i;
}

#define USE_POSIX_MEMALIGN 0

#if USE_POSIX_MEMALIGN
int allocate_with_alignment(void **p, size_t alignment, size_t size)
{
#if 1
  fprintf(stderr, "alignment = %ld\n", alignment);
  int rc = posix_memalign(p, alignment, size);
  assert((uintptr_t) *p % alignment == 0);
  return rc;
#else
  *p = malloc(size);
  return *p ? 0 : -1;
#endif
}

#else
static int shared_mem_initialized = 0;
static pid_t parent_pid;

void *shared_memory_get(const char *shm_path, size_t buf_size, size_t alignment)
{
#if 0  
  /* Create shared memory object and set its size to the size
     of our structure. */

  shm_unlink(shm_path);
	
  int fd = shm_open(shm_path, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);

  if (fd == -1) {
    perror("shm_open");
    return NULL;
  }

  if (ftruncate(fd, buf_size + alignment) == -1) {
   perror("ftruncate");
   return NULL;
  }

  /* Map the object into the caller's address space. */
  void *rc = mmap(NULL, buf_size + alignment, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);

  if (rc == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }
#else
  if (!shared_mem_initialized) {
    shared_mem_initialized = 1;
    parent_pid = getpid();
  }

  if (getpid() != parent_pid) {
    fprintf(stderr, "Failed to allocate %ld bytes: parent_pid %d mypid %d shared memory allocation only supported from parent!\n", buf_size, parent_pid, getpid());
    return NULL;
  }
  
  /* Map the object into the caller's address space. */
  void *rc = mmap(NULL, buf_size + alignment, PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_SHARED, -1, 0);

  if (rc == MAP_FAILED) {
    perror("mmap");
    return NULL;
  }
#endif  

  /* Adjust the alignment */
  uintptr_t p = (uintptr_t) rc;
  uintptr_t q = alignment * (1 + p / alignment);

#if DEBUG_PRINT
  fprintf(stderr, "align %ld p %lu q %lu\n", alignment, p, q);
#endif  

  return (void *) q;
}


int allocate_with_alignment(void **p, size_t alignment, size_t size)
{
        static int count = 0;
        char name[256];
	snprintf(name, sizeof(name), "hello-%ld-%ld-%d", alignment, size, count++);
	*p = shared_memory_get(name, size, alignment);

	if (*p == NULL) {
	  fprintf(stderr, "shared_memory_get for alignment %ld and size %ld failed at count %d\n", alignment, size, count);
	}

	assert((uintptr_t) *p % alignment == 0);
	return *p ? 0 : -1;
}
#endif

struct slab *slab_alloc(size_t pagesize, size_t chunk_size, size_t chunk_alignment)
{
	size_t slab_base_offset = sizeof(struct slab);

	/* Round the chunk_size up for alignment. */

	if (chunk_alignment != 0) {

		if (chunk_size % chunk_alignment != 0) {
			size_t padding = chunk_alignment - (chunk_size % chunk_alignment);
#if KMEM_DEBUG
			printf("Padding chunk_size from %u to %u to satisfy alignment %u\n",
			       chunk_size, chunk_size + padding, chunk_alignment);
#endif
			chunk_size += padding;
		}
	}

	void *p;

	size_t mem_alignment = pagesize;

	// printf("alloc %ld\n", pagesize);
	
	if (allocate_with_alignment(&p, mem_alignment, pagesize)) {
		return NULL;
	}

	if (chunk_alignment != 0) {
		/* Align the base pointer. */
		uintptr_t d = (uintptr_t) p + slab_base_offset;

		if (d % chunk_alignment !=0) {
			size_t padding = chunk_alignment - (d % chunk_alignment);
#if KMEM_DEBUG
			printf("Padding base ptr %u by %u from %u to %u to satisfy alignment %u\n",
			       d, padding, slab_base_offset, slab_base_offset + padding, chunk_alignment);
#endif
			slab_base_offset += padding;
		}
	}

	size_t max_size = pagesize - slab_base_offset;

	if (chunk_size > max_size) {
		/* Sanity check -- should never be reached. */
		fprintf(stderr, "FAILED: Allocations larger than %lu not supported!\n", max_size);
		//free(p); /* xxx */
		return NULL;
	}

#if KMEM_DEBUG
	printf("slab_alloc %u bytes return %p\n", pagesize, p);
#endif

	struct slab *s = (struct slab *) p;

	s->id = slab_id++;
	s->chunk_size = chunk_size;
	s->refcnt = 0;
	s->base = (unsigned char *) p + slab_base_offset;
	s->n_chunks = max_size / chunk_size;
	s->buf_freelist = NULL;
	s->magic = MAGIC_PATTERN;

#if KMEM_DEBUG
	printf("Allocated id %d slab %p with base %p chunk size %u usable storage %u chunks %d\n",
	       s->id,
	       s,
	       s->base,
	       s->chunk_size,
	       max_size,
	       s->n_chunks
		);
#endif

	size_t i;
	for (i=0; i<s->n_chunks; i++) {
		s->buf_freelist = sl_prepend(s->buf_freelist, slab_idx_to_chunk(s, i));
	}

	return s;
}

void slab_free(struct slab *s)
{
	if (s->magic != MAGIC_PATTERN) {
		fprintf(stderr, "Warning: slab data corruption detected\n");
		return;
	}

#if KMEM_DEBUG
	memset(s, 0, sizeof(*s));
#endif

#if USE_POSIX_MEMALIGN
	free(s);
#endif
}


/* Find the start of a slab from the address of a pointer allocated
 * from that slab.
 */
struct slab *slab_from_ptr(size_t pagesize, const void *p)
{
	uintptr_t pagemask = ~(pagesize - 1);
	uintptr_t d = (uintptr_t) p & pagemask;
	struct slab *s = (struct slab *) d;
	return s;
}

/* Debug print for dl_list
 */
void dl_printf(struct dl_node *n)
{	
	while (n) {
		printf("node %p data %p slab_id %u prev %p next %p\n",
		       n,
		       n->data,
		       n->data ? ((struct slab *) n->data)->id : -1,
		       n->prev,
		       n->next);
		n = n->next;
	}
}

/* Find the next-highest power of two.
 * This is based on a well-known bit-twiddling hack.
 * See: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 
 */

size_t next_power_of_2(size_t x)
{
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
#if 0
	/* Uncomment for 64-bit machines */
	x |= x >> 32;
#endif
	x++;
	return x;
}


/* Create a cache of objects of a given size and alignment. */
int kmem_cache_init(struct kmem_cache *c, size_t size, size_t alignment)
{
        int scale = 4096;
	size_t pagesize = getpagesize();

	c->alignment = alignment;
	c->size = size;
	c->freelist = NULL;


	/* There has to be enough space to store a slab's free buffer
	 * list within those free buffers.
	 */
	if (size < sizeof(struct sl_node)) {
		size = sizeof(struct sl_node);
	}

	size_t size_and_overhead = size + sizeof(struct slab);

	if (alignment != 0) {
		/* We do not know how much we might have to pad to
		 * get the correct base offset.
		 */
		size_and_overhead += 2 * alignment;
	}

	size_t np2 = next_power_of_2(size_and_overhead);

	if (np2 > scale * pagesize) {
		c->pagesize = np2;
	}
	else {
		c->pagesize = scale * pagesize;
	}

	c->initialized = 1;
	return 0;
}


struct kmem_cache *kmem_cache_create(
	const char *name,
	size_t size,
	size_t alignment
	)
{
	struct kmem_cache *c = malloc(sizeof(struct kmem_cache));

	if (c) {
		kmem_cache_init(c, size, alignment);
	}
	return c;
}

void kmem_cache_destroy(
	struct kmem_cache *cp)
{
	if (cp) {
		free(cp);
	}
}

void *kmem_cache_alloc(
	struct kmem_cache *cp,
	__attribute__((unused)) unsigned flags)
{
	struct slab *s;

	if (dl_isempty(cp->freelist)) {

		if (flags & KM_NOSLEEP) {
			errno = EWOULDBLOCK;
			return NULL;
		}

		s = slab_alloc(cp->pagesize, cp->size, cp->alignment);

		if (s == NULL)
			return NULL;

		s->dl.data = s;
		cp->freelist = dl_prepend(cp->freelist, &s->dl);
		assert(s == cp->freelist->data);
	}

	if (!cp->freelist->data) {
	  fprintf(stderr, "Assertion `cp->freelist->data' failed.\n");
	  abort();
	}
	
	s = cp->freelist->data;

#if KMEM_DEBUG
	if (s->magic != MAGIC_PATTERN) {
		printf("bad magic: cp->freelist is %p and s is %p s->id is %d\n",
		       cp->freelist,
		       s,
		       s->id);

		return NULL;
	}
#endif

	void *p = s->buf_freelist;

	s->buf_freelist = sl_remove_head(s->buf_freelist);

	s->refcnt++;

#if KMEM_DEBUG
	printf("kmem_cache_alloc slab %p id %d refcnt %d n_chunks %d\n", s, s->id, s->refcnt, s->n_chunks);
#endif

	if (s->refcnt == s->n_chunks) {
#if KMEM_DEBUG
		printf("This slab no longer has free space.\n");
#endif
		cp->freelist = dl_remove_head(cp->freelist);
	}


#if KMEM_DEBUG
	dl_printf(cp->freelist);
#endif


	return p;
}


void kmem_cache_free(
	struct kmem_cache *cp,
	void *buf)
{
	if (buf == NULL || cp == NULL)
		return;

	struct slab *s = slab_from_ptr(cp->pagesize, buf);
#if 0
	if (s->magic != MAGIC_PATTERN) {
		fprintf(stderr, "Warning: slab data corruption detected.\n");
		return;
	}
#endif

#if DEBUG_PRINT	
	fprintf(stderr, "kmem_cache_free mapped buf %p to slab %p refcnt is %u\n",
	       buf, s, s->refcnt
		);

	if (s->magic != MAGIC_PATTERN) {
		fprintf(stderr, "Invalid magic pattern detected in kmem_cache_free\n");
		return;
	}
#endif

	s->refcnt--;

	s->buf_freelist = sl_prepend(s->buf_freelist, buf);

	if (s->refcnt == 0) {

#if KMEM_DEBUG
		printf("reap slab id %d ptr %p (and remove from freelist) dl.prev is %p dl.next is %p\n", s->id, s, s->dl.prev, s->dl.next);
#endif

		/* If a slab only holds a single object, then the
		 * refcnt goes from 1 to 0. So it's possible to be
		 * not on the freelist.
		 */

		int on_freelist = s->dl.next != NULL || s->dl.prev != NULL || cp->freelist == &s->dl;
		int is_head = cp->freelist == &s->dl;

		if (on_freelist) {

#if KMEM_DEBUG
	                printf("freelist at kmem_cache_free reap:\n");
			dl_printf(cp->freelist);
#endif

			struct dl_node *next = dl_remove(&s->dl);

			if (is_head) {
#if KMEM_DEBUG
				printf("Removing head node and replacing with next\n");
#endif
				cp->freelist = next;
			}
		}

		slab_free(s);
	}
	else if (s->refcnt == s->n_chunks - 1) {
		/* Slab was off the free list but is now ready to go
		 * back on.
		 */

#if KMEM_DEBUG
		printf("put slab %p back onto the free list\n", s);
#endif
		s->dl.data = s;
		cp->freelist = dl_prepend(cp->freelist, &s->dl);
	}

#if KMEM_DEBUG
	printf("freelist at kmem_cache_free exit:\n");
	dl_printf(cp->freelist);
	printf("kmem_cache_free exit\n");
#endif

}

size_t kmem_effective_pagesize(struct kmem_cache *cp)
{
	return cp->pagesize;
}

size_t get_sizeof_slab(int foo)
{
        return sizeof(struct slab);
}
