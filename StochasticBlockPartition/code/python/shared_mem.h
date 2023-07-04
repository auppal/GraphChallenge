#ifndef SHARED_MEM_H
#define SHARED_MEM_H
#include <stddef.h>
void *shared_malloc(size_t nbytes);
void *shared_calloc(size_t nmemb, size_t size);
void shared_free(void *p, size_t alloc_size);
void shared_print_report();
void shared_query(size_t pool, size_t *p_n_used, size_t *p_n_free);
int shared_reserve(size_t pool, size_t n_items);

#define SHARED_MAX_POOLS (64)
#endif
