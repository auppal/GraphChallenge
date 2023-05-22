#ifndef SLAB_ALLOC_H
#define SLAB_ALLOC_H

#include <stddef.h>

/* 
 * slab allocator written by Ahsen Uppal
 * for csci6907 homework 1 -- spring 2014.
 */

/* Create a cache of objects of a given size and alignment.
 */


struct kmem_cache
{
	int initialized;
	size_t size;
	size_t alignment;
	/* The freelist of slabs */
	struct dl_node *freelist;
	/* Effective pagesize i.e. size and slab overhead rounded up
	 * to the next higher number of pages.
	 */	 
	size_t pagesize;
};

struct kmem_cache *kmem_cache_create(
	const char *name,
	size_t size,
	size_t alignment
	);

#define KM_SLEEP (0)
#define KM_NOSLEEP (1)

void *kmem_cache_alloc(
	struct kmem_cache *cp,
	unsigned flags);


void kmem_cache_free(
	struct kmem_cache *cp,
	void *buf);

void kmem_cache_destroy(
	struct kmem_cache *cp
	);

int slab_get_count();
size_t kmem_effective_pagesize();

#endif
