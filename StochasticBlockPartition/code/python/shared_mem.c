#include "shared_mem.h"
#include <stdlib.h>
#include <stdio.h>

#include "slab_alloc.h"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#if 0
#include "shm_alloc.h"
static int initialized = 0;

void shared_init()
{
  char *tmp_shm_file = "temp_shm_file";
  bool retval;

  remove(tmp_shm_file);

  retval = shm_init(NULL, tmp_shm_file);

  if (retval == false) {
    fprintf(stderr, "shm_init() failed!");
    exit(EXIT_FAILURE);
  }
}  

void *shared_malloc(size_t nbytes)
{
  if (!initialized) {
    shared_init();
  }
  fprintf(stderr, "shm_malloc %ld\n", nbytes);  
  return ptr_malloc(nbytes);
}

void *shared_calloc(size_t nmemb, size_t size)
{
  if (!initialized) {
    shared_init();
  }
  fprintf(stderr, "shm_calloc %ld\n", nmemb * size);
  return ptr_calloc(nmemb, size);
}

void shared_free(void *p)
{
  ptr_free(p);
}

#elif 0
void *shared_malloc(size_t nbytes)
{
  // fprintf(stderr, "shm_malloc %ld\n", nbytes);
  return malloc(nbytes);
}

void *shared_calloc(size_t nmemb, size_t size)
{
  // fprintf(stderr, "shm_calloc %ld\n", nmemb * size);  
  return calloc(nmemb, size);
}

void shared_free(void *p, size_t nbytes)
{
  free(p);
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
