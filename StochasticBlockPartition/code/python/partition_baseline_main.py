from partition_baseline_support import *
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool, Value, Semaphore, Manager, Queue, current_process
import pickle
import timeit
import os, sys, argparse
import time, struct
import traceback
import numpy.random
from compute_delta_entropy import compute_delta_entropy
import random
import shutil
from interblock_edge_count import is_compressed
import collections

try:
    from queue import Empty as queue_empty
except:
    from Queue import Empty as queue_empty

try:
    from mpi4py import MPI
except:
    MPI = None


compressed_threshold = 5000
timing_stats = defaultdict(int)

log_timestamp_prev = 0
def log_timestamp(msg):
    global log_timestamp_prev
    t_now = timeit.default_timer() - t_prog_start
    if log_timestamp_prev == 0:
        log_timestamp_prev = t_now
    t_elp = t_now - log_timestamp_prev
    print("%3.4f +%3.4f %s" % (t_now, t_elp, msg))
    log_timestamp_prev = t_now

def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def find_self_edge_weights(N, out_neighbors):
    self_edge_weights = defaultdict(int)
    for i in range(N):
        if i in out_neighbors[i][:,0]:
            self_edge_weights[i] = np.sum(out_neighbors[i][np.where(
                out_neighbors[i_node][:, 0] == i), 1])
    return self_edge_weights

def acquire_locks_nowait(locks):
    for i,e in enumerate(locks):
        if not e.acquire(False):
            for j in range(i):
                locks[j].release()
            return False
    return True

def acquire_locks(locks):
    if locks is not None:
        for i in locks:
            i.acquire()

def release_locks(locks):
    if locks is not None:
        for i in locks[::-1]:
            i.release()

def entropy_max_argsort(x):
    a = np.argsort(x)
    y = x[a]
    m = np.zeros(y.shape, dtype=bool)

    # Positions of transition starts
    m[0]  = 1
    m[1:] = y[1:] != y[0:-1]
    starts = np.where(m)[0]

    # Positions of transition ends
    ends = np.empty(starts.shape, dtype=int)
    ends[0:-1] = starts[1:]
    ends[-1] = y.shape[0]

    # Run the counters
    ii = starts.copy()
    out = np.empty(x.shape, dtype=int)
    k = 0

    while k < x.shape[0]:
        for idx,elm in enumerate(starts):
            if ii[idx] < ends[idx]:
                out[k] = a[ii[idx]]
                ii[idx] += 1
                k += 1
    return out

def blocks_and_counts(partition, neighbors):
    """"
    Compute neighboring blocks and edge counts to those blocks for a vertex.
    neighbors has shape (n_neighbors, 2)
    """
    return compressed_array.blocks_and_counts(partition, neighbors[:, 0], neighbors[:, 1]);

def compute_best_block_merge_wrapper(tup):
    (blocks, num_blocks) = tup

    interblock_edge_count = syms['interblock_edge_count']
    block_partition = syms['block_partition']
    block_degrees = syms['block_degrees']
    block_degrees_out = syms['block_degrees_out']
    block_degrees_in = syms['block_degrees_in']
    args = syms['args']

    return compute_best_block_merge(blocks, num_blocks, interblock_edge_count, block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in, args)


def compute_best_block_merge(blocks, num_blocks, M, block_partition, block_degrees, n_proposal, block_degrees_out, block_degrees_in, args):
    best_overall_merge = [-1 for i in blocks]
    best_overall_delta_entropy = [np.Inf for i in blocks]
    n_proposals_evaluated = 0

    for current_block_idx,r in enumerate(blocks):
        if r is None:
            break

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, r, 1, sort = False)
        out_idx, out_weight = take_nonzero(M, r, 0, sort = False)

        block_neighbors = np.concatenate((in_idx, out_idx))
        block_neighbor_weights = np.concatenate((in_weight, out_weight))
        
        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        n_proposal = 10
        delta_entropy = np.empty(n_proposal)


        proposals = np.empty(n_proposal, dtype=int)

        # propose new blocks to merge with
        for proposal_idx in range(n_proposal):
            s = propose_new_partition(
                r,
                block_neighbors,
                block_neighbor_weights,
                num_block_edges,
                block_partition, M, block_degrees, num_blocks,
                agg_move = 1)

            s = int(s)
            proposals[proposal_idx] = s

            # compute the two new rows and columns of the interblock edge count matrix
            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, cur_M_r_row, cur_M_s_row, cur_M_r_col, cur_M_s_col = \
                compute_new_rows_cols_interblock_edge_count_matrix(M, r, s,
                                                                   out_idx, out_weight,
                                                                   in_idx, in_weight,
                                                                   M[r, r],
                                                                   agg_move = 1)
            # compute change in entropy / posterior
            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(r,
                                            s,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[proposal_idx] = compute_delta_entropy(r, s,
                                                                cur_M_r_row,
                                                                cur_M_s_row,
                                                                cur_M_r_col,
                                                                cur_M_s_col,
                                                                new_M_r_row,
                                                                new_M_s_row,
                                                                new_M_r_col,
                                                                new_M_s_col,
                                                                block_degrees_out,
                                                                block_degrees_in,
                                                                block_degrees_out_new,
                                                                block_degrees_in_new)
        mi = np.argmin(delta_entropy)
        best_entropy = delta_entropy[mi]
        n_proposals_evaluated += n_proposal

        if best_entropy < best_overall_delta_entropy[current_block_idx]:
            best_overall_merge[current_block_idx] = proposals[mi]
            best_overall_delta_entropy[current_block_idx] = best_entropy

    return blocks, best_overall_merge, best_overall_delta_entropy, n_proposals_evaluated


propose_node_movement_profile_stats = []
def propose_node_movement_profile_wrapper(tup):
    mypid = current_process().pid
    rc = []
    cnt = len(propose_node_movement_profile_stats)
    propose_node_movement_profile_stats.append(cProfile.runctx("rc.append(propose_node_movement_wrapper(tup))", globals(), locals(), filename="propose_node_movement-%d-%d.prof" % (mypid,cnt)))
    return rc[0]


def get_locks(finegrain, vertex_locks, ni, vertex_neighbors):
    if finegrain:
        neighbors = sorted([i for i in vertex_neighbors[ni][:, 0]] + [ni])
        locks = [vertex_locks[i] for i in neighbors]
    else:
        locks = [vertex_locks[0]]
    return locks

update_id = -1
def propose_node_movement_wrapper(tup):
    global update_id, mypid

    rank,start,stop,step = tup

    args = syms['args']
    vertex_locks,block_locks = syms['locks']
    
    (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights) = syms['static_state']
    (update_id_shared, M, partition, block_degrees, block_degrees_out, block_degrees_in,) = syms['nodal_move_state']
    barrier = syms['barrier']

    if args.sort:
        reorder = syms['reorder']
    
    if update_id != update_id_shared.value:
        if update_id == -1:
            # Ensure every worker has a different random seed.
            mypid = current_process().pid
            numpy.random.seed((mypid + int(timeit.default_timer() * 1e6)) % 4294967295)

        update_id = update_id_shared.value

    if args.verbose > 3:
        print("Rank %d pid %d start %d stop %d step %d" % (rank,mypid,start,stop,step))


    worker_delta_entropy = 0.0
    worker_n_moves = 0
    pop_cnt = 0

    # r is current_block, s is proposal
    if args.blocking:
        for ni in range(start, stop, step):
            if args.sort:
                ni = reorder[ni]

            locks = get_locks(args.finegrain, vertex_locks, ni, vertex_neighbors)

            args.critical == 0 and acquire_locks(locks)

            movement = propose_node_movement(ni, partition, out_neighbors, in_neighbors,
                                             M, num_blocks, block_degrees, block_degrees_out, block_degrees_in,
                                             vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges,
                                             vertex_neighbors, self_edge_weights, args, vertex_lock=None, block_lock=None)

            (ni2, r, s, delta_entropy, p_accept, new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, block_degrees_out_new, block_degrees_in_new) = movement
            accept = (np.random.uniform() <= p_accept)

            if accept:
                worker_delta_entropy += delta_entropy
                worker_n_moves += 1

                if args.critical == 0:
                    move_node(ni, r, s, partition,
                              out_neighbors, in_neighbors, self_edge_weights, M,
                              block_degrees_out, block_degrees_in, block_degrees, vertex_locks = None, blocking=True)
                elif args.critical == 1:
                    acquire_locks(locks)
                    move_node(ni, r, s, partition,
                              out_neighbors, in_neighbors, self_edge_weights, M,
                              block_degrees_out, block_degrees_in, block_degrees, vertex_locks = None, blocking=True)
                    release_locks(locks)
                else:
                    move_node(ni, r, s, partition,
                              out_neighbors, in_neighbors, self_edge_weights, M,
                              block_degrees_out, block_degrees_in, block_degrees, vertex_locks = locks, blocking=True)

            args.critical == 0 and release_locks(locks)
    else:
        # Non-blocking locking implicitly only makes sense for critcal sections 1 or 2
        queue = collections.deque()
        for ni in range(start, stop, step):        
            movement = propose_node_movement(ni, partition, out_neighbors, in_neighbors,
                                             M, num_blocks, block_degrees, block_degrees_out, block_degrees_in,
                                             vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges,
                                             vertex_neighbors, self_edge_weights, args, vertex_lock=None, block_lock=None)
            (ni2, r, s, delta_entropy, p_accept, new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, block_degrees_out_new, block_degrees_in_new) = movement
            accept = (np.random.uniform() <= p_accept)
            if accept:
                worker_delta_entropy += delta_entropy
                worker_n_moves += 1
                queue.append((ni,r,s))

        barrier.wait()

        while queue:
            ni,r,s = queue.popleft()
            pop_cnt += 1
            locks = get_locks(args.finegrain, vertex_locks, ni, vertex_neighbors)

            if args.critical == 1:
                if not acquire_locks_nowait(locks):
                    queue.append((ni,r,s))
                    continue

                move_node(ni, r, s, partition,
                          out_neighbors, in_neighbors, self_edge_weights, M,
                          block_degrees_out, block_degrees_in, block_degrees, vertex_locks=None)
                release_locks(locks)
            else:
                if not move_node(ni, r, s, partition,
                                 out_neighbors, in_neighbors, self_edge_weights, M,
                                 block_degrees_out, block_degrees_in, block_degrees, vertex_locks=locks, blocking=args.blocking):
                    queue.append((ni,r,s))
                    continue

    return rank,mypid,update_id,start,stop,step,worker_n_moves,worker_delta_entropy,pop_cnt



def propose_node_movement(current_node, partition, out_neighbors, in_neighbors, M, num_blocks,
                          block_degrees, block_degrees_out, block_degrees_in,
                          vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args, vertex_lock=None, block_lock=None):
    # SHR: read partition[ni]
    r = partition[current_node]

    # SHR: read u = partition[rand_neighbor]
    # SHR: read row M[u,:] col M[:,u]
    
    s = propose_new_partition(
        r,
        vertex_neighbors[current_node][:, 0],
        vertex_neighbors[current_node][:, 1],
        vertex_num_neighbor_edges[current_node],
        partition,
        M, block_degrees, num_blocks, agg_move = 0, M_lock=block_lock)

    num_out_neighbor_edges = vertex_num_out_neighbor_edges[current_node]
    num_in_neighbor_edges = vertex_num_in_neighbor_edges[current_node]
    num_neighbor_edges = vertex_num_neighbor_edges[current_node]

    # determine whether to accept or reject the proposal
    if s == r:
        (current_node, r, s, delta_entropy, p_accept, new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, block_degrees_out_new, block_degrees_in_new) = current_node, r, int(s), 0.0, False, None, None, None, None, None, None
    else:
        # SHR: read partition[out_neighbors[current_node]]
        # SHR: read partition[in_neighbors[current_node]]
        blocks_out,count_out = compressed_array.blocks_and_counts(
            partition, out_neighbors[current_node][:, 0], out_neighbors[current_node][:, 1])
        blocks_in,count_in = compressed_array.blocks_and_counts(
            partition, in_neighbors[current_node][:, 0], in_neighbors[current_node][:, 1])

        # compute the two new rows and columns of the interblock edge count matrix
        self_edge_weight = self_edge_weights[current_node]

        # SHR: read M[r,:] M[s,:] M[:,r] M[:s]
        new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, cur_M_r_row, cur_M_s_row, cur_M_r_col, cur_M_s_col = \
            compute_new_rows_cols_interblock_edge_count_matrix(M, r, s,
                                                               blocks_out, count_out, blocks_in, count_in,
                                                               self_edge_weight, agg_move = 0, M_lock=block_lock)

        # SHR: read block_degrees_out[:], block_degrees_in[:], block_degrees[:]
        block_degrees_out_new = block_degrees_out.copy()
        block_degrees_in_new = block_degrees_in.copy()
        block_degrees_new = block_degrees.copy()

        block_degrees_out_new[r] -= num_out_neighbor_edges
        block_degrees_out_new[s] += num_out_neighbor_edges
        block_degrees_in_new[r] -= num_in_neighbor_edges
        block_degrees_in_new[s] += num_in_neighbor_edges
        block_degrees_new[s] = block_degrees_out[s] + block_degrees_in[s]
        block_degrees_new[r] = block_degrees_out[r] + block_degrees_in[r]

        # SHR: read block_degrees[t] where t is union of b_out and b_in
        Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in,
                                                          r,
                                                          s,
                                                          cur_M_s_row,
                                                          cur_M_s_col,
                                                          new_M_r_row,
                                                          new_M_r_col,
                                                          num_blocks, block_degrees,
                                                          block_degrees_new)

        # compute change in entropy / posterior
        delta_entropy = compute_delta_entropy(r, s,
                                              cur_M_r_row,
                                              cur_M_s_row,
                                              cur_M_r_col,
                                              cur_M_s_col,
                                              new_M_r_row,
                                              new_M_s_row,
                                              new_M_r_col,
                                              new_M_s_col,
                                              block_degrees_out,
                                              block_degrees_in,
                                              block_degrees_out_new,
                                              block_degrees_in_new)

        # Clamp to avoid under- and overflow
        if delta_entropy > 10.0:
            delta_entropy = 10.0
        elif delta_entropy < -10.0:
            delta_entropy = -10.0

        p_accept = np.min([np.exp(-args.beta * delta_entropy) * Hastings_correction, 1])

    return current_node, r, s, delta_entropy, p_accept, new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, block_degrees_out_new, block_degrees_in_new    


def move_node_wrapper(tup):
    global update_id, mypid

    rank,start,stop,step = tup

    args = syms['args']
    vertex_locks,block_locks = syms['locks']
    (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights) = syms['static_state']
    (update_id_shared, M, partition, block_degrees, block_degrees_out, block_degrees_in,) = syms['nodal_move_state']
    partition_next = syms['partition_next']

    for ni in range(start, stop, step):
        if partition[ni] == partition_next[ni]:
            continue
        move_node(ni, partition[ni], partition_next[ni], partition,
                  out_neighbors, in_neighbors, self_edge_weights, M,
                  block_degrees_out, block_degrees_in, block_degrees, vertex_locks=vertex_locks, blocking=True)
    return True


def move_node(ni, r, s, partition,out_neighbors, in_neighbors, self_edge_weights, M, block_degrees_out, block_degrees_in, block_degrees, vertex_locks = None, blocking=True):

    if blocking:
        acquire_locks(vertex_locks)
    else:
        if not acquire_locks_nowait(vertex_locks):
            return False

    blocks_out,count_out = compressed_array.blocks_and_counts(partition,
                                                              out_neighbors[ni][:, 0], out_neighbors[ni][:, 1])
    blocks_in,count_in = compressed_array.blocks_and_counts(partition,
                                                            in_neighbors[ni][:, 0], in_neighbors[ni][:, 1])
    partition[ni] = s

    release_locks(vertex_locks)

    if is_compressed(M):
        compressed_array.inplace_compute_new_rows_cols_interblock_edge_count_matrix(M.x, r, s, blocks_out, count_out, blocks_in, count_in, block_degrees_out, block_degrees_in, block_degrees)
    else:
        compressed_array.inplace_atomic_new_rows_cols_M(M, r, s, blocks_out, count_out, blocks_in, count_in, block_degrees_out, block_degrees_in, block_degrees)

    return True

def nodal_moves_sequential(delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args):
    max_num_nodal_itr = args.max_num_nodal_itr
    delta_entropy_moving_avg_window = args.delta_entropy_moving_avg_window    
    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)
    
    if args.verbose > 1:
        compressed_array.shared_memory_report()

    if args.sanity_check:
        sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees)

    if args.mpi == 1:
        comm = MPI.COMM_WORLD
        mpi_chunk_size = N // comm.size
        start_vert = comm.rank * mpi_chunk_size
        stop_vert = min((comm.rank + 1) * mpi_chunk_size, N)
        move_node_iterator = range(start_vert, stop_vert)
    else:
        start_vert = 0
        stop_vert = N        
        
    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0
        itr_delta_entropy[itr] = 0
        proposal_cnt = 0

        if args.sort:
            #L = np.argsort(partition)
            L = entropy_max_argsort(partition)
        else:
            L = range(start_vert, stop_vert)

        update_id_cnt = 0
        
        for i in L:
            movement = propose_node_movement(i, partition, out_neighbors, in_neighbors, M, num_blocks,
                                             block_degrees, block_degrees_out, block_degrees_in,
                                             vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args)

            if args.sanity_check:
                sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees)

            (ni, r, s, delta_entropy, p_accept, new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, block_degrees_out_new, block_degrees_in_new) = movement
            accept = (np.random.uniform() <= p_accept)

            if not accept:
                continue

            total_num_nodal_moves_itr += 1
            num_nodal_moves += 1
            itr_delta_entropy[itr] += delta_entropy

            if args.verbose > 3:
                print("Move %5d from block %5d to block %5d." % (ni, r, s))

            move_node(ni, r, s, partition,
                      out_neighbors, in_neighbors, self_edge_weights, M,
                      block_degrees_out, block_degrees_in, block_degrees)

        if args.mpi == 1:
            partition_next = partition.copy()
            # Synchronize with all other instances at the end of each iteration.
            send_tuple = (comm.rank,
                          move_node_iterator,
                          num_nodal_moves,
                          itr_delta_entropy[itr],
                          proposal_cnt,
                          partition[move_node_iterator])

            mpi_result = comm.allgather(send_tuple)
            num_nodal_moves = 0
            itr_delta_entropy[itr] = 0
            proposal_cnt = 0
            for i in mpi_result:
                ci,citr,num_nodal_moves_piece,itr_delta_entropy_piece,proposal_cnt_piece,partition_piece = i
                num_nodal_moves += num_nodal_moves_piece
                itr_delta_entropy[itr] += itr_delta_entropy_piece
                proposal_cnt += proposal_cnt_piece
                partition_next[citr] = partition_piece

            # Carry out the movements from remote jobs
            w = np.where(partition_next != partition)[0]
            for ni in w:
                move_node(ni, partition[ni], partition_next[ni], partition,
                          out_neighbors, in_neighbors, self_edge_weights, M,
                          block_degrees_out, block_degrees_in, block_degrees)

        if args.sanity_check:
            sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees)            

        if args.verbose:
            print("Itr: {:3d}, number of nodal moves: {:3d}, delta S: {:0.9f}".format(itr, num_nodal_moves,
                                                                                itr_delta_entropy[itr] / float(
                                                                                    overall_entropy_cur)))
        if num_nodal_moves <= (N * args.min_nodal_moves_ratio):
            break

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):  
            if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold * overall_entropy_cur)):
                    break

    return total_num_nodal_moves_itr,partition,M,block_degrees_out,block_degrees_in,block_degrees

def recompute_M(B, partition, out_neighbors):
    M = np.zeros((B,B), dtype=int)
    for v in range(len(out_neighbors)):
        k1 = partition[v]
        if len(out_neighbors[v]) > 0:
            k2,count = compressed_array.blocks_and_counts(
                partition, out_neighbors[v][:, 0], out_neighbors[v][:, 1])
            M[k1, k2] += count
    return M


def sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees):
    M2 = recompute_M(M.shape[0], partition, out_neighbors)
    bd_out = np.sum(M2, axis=1)
    bd_in = np.sum(M2, axis=0)
    bd = np.add(bd_out, bd_in)
    
    if is_compressed(M):
        compressed_array.sanity_check(M.x)
        if not (M == M2):
            raise Exception("Sanity check of interblock edge count matrix failed.")
    else:
        if not np.array_equal(M, M2):
            raise Exception("Sanity check of interblock edge count matrix failed.")
        if not np.array_equal(block_degrees_out, bd_out):
            raise Exception("Sanity check of block_degrees_out failed.")
        if not np.array_equal(block_degrees_in, bd_in):
            raise Exception("Sanity check of block_degrees_in failed.")
        if not np.array_equal(block_degrees, bd):
            raise Exception("Sanity check of block_degrees failed.")
    

def nodal_moves_parallel(n_thread_move, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args):
    global syms

    max_num_nodal_itr = args.max_num_nodal_itr
    delta_entropy_moving_avg_window = args.delta_entropy_moving_avg_window
    
    if args.verbose > 1:
        compressed_array.shared_memory_report()

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    block_lock = None

    if args.finegrain:
        vertex_lock = [mp.Lock() for i in range(N)]
    else:
        vertex_lock = [mp.Lock()]

    update_id_shared = Value('i', 0)
    static_state = (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights)

    shape = partition.shape

    syms = {}
    syms['locks'] = vertex_lock,block_lock
    syms['static_state'] = static_state
    syms['nodal_move_state'] = (update_id_shared, M, partition, block_degrees, block_degrees_out, block_degrees_in)
    syms['args'] = args
    barrier = mp.Barrier(n_thread_move)
    syms['barrier'] = barrier

    if args.sort:
        reorder = np.arange(N, dtype=int)
        syms['reorder'] = reorder

    update_id_cnt = 0
    wrapper_fn = propose_node_movement_wrapper if args.profile == 0 else propose_node_movement_profile_wrapper

    if args.node_propose_batch_size == 0:
        group_size = (N + n_thread_move - 1) // n_thread_move
    else:
        group_size = args.node_propose_batch_size

    if args.mpi == 1:
        comm = MPI.COMM_WORLD
        mpi_chunk_size = N // comm.size
        start_vert = comm.rank * mpi_chunk_size
        stop_vert = min((comm.rank + 1) * mpi_chunk_size, N)
        move_node_iterator = range(start_vert, stop_vert)
        partition_next = shared_memory_copy(partition)
        syms['partition_next'] = partition_next
    else:
        start_vert = 0
        stop_vert = N        

    pool = Pool(n_thread_move)
    n_thread_movers = n_thread_move
    pool_movers = Pool(n_thread_movers)

    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0
        itr_delta_entropy[itr] = 0
        proposal_cnt = 0

        if args.sort:
            # Try to arrange the verticies such that each worker processes vertices that are far from each other.
            s = np.argsort(partition)
            k = (N + n_thread_move - 1) // n_thread_move
            c = 0
            for j in range(0,k):
                for i in range(j,N,n_thread_move):
                    if c == N:
                        break
                    reorder[c] = i
                    c += 1

        if args.blocking:
            # Blocking locks do not need to call barrier and thus do not require n_thread_move numbers of threads for each invocation
            chunks = [((i // group_size) % n_thread_move, i, min(i+group_size, stop_vert), 1) for i in range(start_vert,stop_vert,group_size)]
            movements = pool.imap_unordered(propose_node_movement_wrapper, chunks)
            for movement in movements:
                rank,worker_pid,worker_update_id,start,stop,step,n_moves,delta_entropy,pop_cnt = movement
                proposal_cnt += (stop - start) // step
                total_num_nodal_moves_itr += n_moves
                num_nodal_moves += n_moves
                itr_delta_entropy[itr] += delta_entropy
        else:
            for start in range(start_vert, stop_vert, n_thread_move * group_size):
                chunk = [(i, min(start + i * group_size, stop_vert), min(start + (i + 1) * group_size, stop_vert), 1) for i in range(n_thread_move)]
                for movement in pool.imap_unordered(wrapper_fn, chunk):
                        rank,worker_pid,worker_update_id,start,stop,step,n_moves,delta_entropy,pop_cnt = movement
                        total_num_nodal_moves_itr += n_moves
                        num_nodal_moves += n_moves
                        itr_delta_entropy[itr] += delta_entropy
                        timing_stats['pop_cnt'] += pop_cnt
                        timing_stats['nodal_moves'] += n_moves
                barrier.reset()

        if args.mpi == 1:
            partition_next[:] = partition[:]
            # Synchronize with all other instances at the end of each iteration.
            send_tuple = (comm.rank,
                          move_node_iterator,
                          num_nodal_moves,
                          itr_delta_entropy[itr],
                          proposal_cnt,
                          partition[move_node_iterator])

            mpi_result = comm.allgather(send_tuple)
            num_nodal_moves = 0
            itr_delta_entropy[itr] = 0
            proposal_cnt = 0
            for i in mpi_result:
                ci,citr,num_nodal_moves_piece,itr_delta_entropy_piece,proposal_cnt_piece,partition_piece = i
                num_nodal_moves += num_nodal_moves_piece
                itr_delta_entropy[itr] += itr_delta_entropy_piece
                proposal_cnt += proposal_cnt_piece
                partition_next[citr] = partition_piece

            # Carry out the movements from remote jobs
            if 0:
                w = np.where(partition_next != partition)[0]                
                for ni in w:
                    move_node(ni, partition[ni], partition_next[ni], partition,
                              out_neighbors, in_neighbors, self_edge_weights, M,
                              block_degrees_out, block_degrees_in, block_degrees)
            else:
                gs = (N + n_thread_movers - 1) // n_thread_movers
                chunks = [(i, i*gs, min((i+1)*gs,N), 1) for i in range(n_thread_movers)]
                for i in pool_movers.imap_unordered(move_node_wrapper, chunks):
                    pass

        if args.sanity_check:
            sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees)

        if args.verbose > 0:
            print("Itr: {:3d}, number of nodal moves: {:3d}, delta S: {:0.9f}".format(itr, num_nodal_moves,
                                                                            itr_delta_entropy[itr] / float(overall_entropy_cur)))
        if num_nodal_moves <= (N * args.min_nodal_moves_ratio):
            break

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):
            if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold * overall_entropy_cur)):
                    break

    pool.close()
    pool_movers.close()

    if args.verbose > 1:
        compressed_array.shared_memory_report()

    return total_num_nodal_moves_itr,partition,M,block_degrees_out,block_degrees_in,block_degrees


def entropy_for_block_count(num_blocks, num_target_blocks, delta_entropy_threshold, M, block_degrees, block_degrees_out, block_degrees_in, out_neighbors, in_neighbors, N, E, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, partition, args, verbose = False):
    global syms

    rank_str = ""
    if args.mpi == 1:
        comm = MPI.COMM_WORLD
        rank_str = "in rank {}".format(comm.rank)

    n_thread_merge = args.t_merge
    n_thread_move = args.t_move

    n_merges = 0
    n_proposals_evaluated = 0

    # begin agglomerative partition updates (i.e. block merging)
    if verbose:
        merge_start_time = timeit.default_timer()
        print("\nMerging down blocks from {} to {} at time {:4.4f}{}".format(num_blocks, num_target_blocks, merge_start_time - t_prog_start, rank_str))

#    if verbose > 1:
#        print("Density at block size {} is {:1.8f}".format(num_blocks, np.count_nonzero(M) / (num_blocks ** 2)))

    best_merge_for_each_block = np.ones(num_blocks, dtype=int) * -1  # initialize to no merge
    delta_entropy_for_each_block = np.ones(num_blocks) * np.Inf  # initialize criterion
    block_partition = np.arange(num_blocks)
    n_merges += 1

    merge_block_iterator = range(num_blocks)

    if args.mpi == 1:
        merge_block_iterator = range(comm.rank, num_blocks, comm.size)

    if n_thread_merge > 0:
        syms = {}
        syms['interblock_edge_count'] = M
        syms['block_partition'] = block_partition
        syms['block_degrees'] = block_degrees
        syms['block_degrees_out'] = block_degrees_out
        syms['block_degrees_in'] = block_degrees_in
        syms['args'] = args

        pool_size = min(n_thread_merge, num_blocks)

        pool = Pool(n_thread_merge)
        for current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated in pool.imap_unordered(compute_best_block_merge_wrapper, [((i,),num_blocks) for i in merge_block_iterator]):
            for current_block_idx,current_block in enumerate(current_blocks):
                best_merge_for_each_block[current_block] = best_merge[current_block_idx]
                delta_entropy_for_each_block[current_block] = best_delta_entropy[current_block_idx]
            n_proposals_evaluated += fresh_proposals_evaluated                
        pool.close()
    else:
        current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated \
            = compute_best_block_merge(merge_block_iterator, num_blocks, M,
                                       block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in, args)
        n_proposals_evaluated += fresh_proposals_evaluated
        for current_block_idx,current_block in enumerate(current_blocks):
            if current_block is not None:
                best_merge_for_each_block[current_block] = best_merge[current_block_idx]
                delta_entropy_for_each_block[current_block] = best_delta_entropy[current_block_idx]

    # During MPI operation, not every entry in best_merge_for_each_block
    # and delta_entropy_for_each_block will have been filled in.
    if args.mpi == 1:
        send_tuple = (comm.rank,
                      1234,
                      merge_block_iterator,
                      n_proposals_evaluated,
                      best_merge_for_each_block[merge_block_iterator],
                      delta_entropy_for_each_block[merge_block_iterator])

        mpi_result = comm.allgather(send_tuple)
        n_proposals_evaluated = 0
        for i in mpi_result:
            # print("    mpi_result ",comm.rank,len(mpi_result),len(mpi_result[0]), len(mpi_result[1]))
            # assert(len(i) == 6)
            ci,magic,itr,n_fresh_proposals,best_merge_piece,delta_entropy_piece = i
            assert(magic == 1234)
            n_proposals_evaluated += n_fresh_proposals
            best_merge_for_each_block[itr] = best_merge_piece
            delta_entropy_for_each_block[itr] = delta_entropy_piece

    if (n_proposals_evaluated == 0):
        raise Exception("No proposals evaluated.")

    overall_entropy_per_num_blocks = np.empty(len(num_target_blocks))
    state_per_num_blocks = [None] * len(num_target_blocks)

    best_overall_entropy = np.Inf
    best_merges = delta_entropy_for_each_block.argsort()

    for i,t in enumerate(num_target_blocks):
        num_blocks_to_merge = num_blocks - t
        # carry out the best merges
        (partition_t, num_blocks_t) = carry_out_best_merges(delta_entropy_for_each_block, best_merges,
                                                            best_merge_for_each_block, partition,
                                                            num_blocks, num_blocks_to_merge)
        # xxx brute force
        partition_t = shared_memory_copy(partition_t)

        # re-initialize edge counts and block degrees
        if args.sparse == 2:
            if num_blocks >= compressed_threshold:
                use_compressed = 1
            else:
                use_compressed = 0
        else:
            use_compressed = (args.sparse != 0)

        # XXX This type of full re-initialization seems wasteful memory for large graphs.

        M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t = \
                initialize_edge_counts(out_neighbors,
                                       num_blocks_t,
                                       partition_t,
                                       use_compressed, args)

        # compute the global entropy for MCMC convergence criterion
        overall_entropy = compute_overall_entropy(M_t, block_degrees_out_t, block_degrees_in_t, num_blocks_t, N,
                                                  E)

        overall_entropy_per_num_blocks[i] = overall_entropy
        state_per_num_blocks[i] = (overall_entropy, partition_t, num_blocks_t, M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t)

    derivative_based_stop = False

    if len(num_target_blocks) > 1:
        S = overall_entropy_per_num_blocks[::-1]
        dS_dn = 0.5 * (S[2] - S[1]) + 0.5 * (S[1] - S[0])

        if len(num_target_blocks) == 3 \
           and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[0] \
           and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[2]:
            if verbose:
                print("Optimal stopping criterion found at %d blocks derivative %s at time %4.4f." % (num_target_blocks[1], dS_dn, timeit.default_timer() - t_prog_start))

            derivative_based_stop = True
            best_idx = 1
        else:
            # Can be zero when there are some nodes with no edges and so reducing the block count
            # does not change the entropy.
            if (S[2] - S[1] - (S[1] - S[0])) != 0:
                extrapolated_newton = num_target_blocks[1] - 0.5 * (S[2] - S[0]) / (S[2] - S[1] - (S[1] - S[0]))
            else:
                extrapolated_newton = 0.0

            if verbose:
                print("Stopping criterion not found at %s blocks extrapolate to %s blocks derivative %s at time %4.4f." % (num_target_blocks[1], extrapolated_newton, dS_dn, timeit.default_timer() - t_prog_start))

            best_idx = np.argsort(overall_entropy)
            best_idx = 1 # xxx
    else:
        dS_dn = 0
        best_idx = 0

    (overall_entropy, partition, num_blocks_t, M, block_degrees_out, block_degrees_in, block_degrees) = state_per_num_blocks[best_idx]

    num_blocks_merged = num_blocks - num_blocks_t
    num_blocks = num_blocks_t

    if verbose:
        merge_end_time = move_start_time = timeit.default_timer()
        print("Best num_blocks = %s" % num_blocks)
        print("blocks %s entropy %s" % (num_target_blocks, overall_entropy_per_num_blocks))
        print("Beginning nodal updates at ", timeit.default_timer() - t_prog_start)

    if args.sanity_check:
        sanity_check_state(partition, out_neighbors, M, block_degrees_out, block_degrees_in, block_degrees)
    
    if n_thread_move > 0:
        total_num_nodal_moves_itr,partition,M,block_degrees_out,block_degrees_in,block_degrees = nodal_moves_parallel(n_thread_move, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args)
    else:
        total_num_nodal_moves_itr,partition,M,block_degrees_out,block_degrees_in,block_degrees = nodal_moves_sequential(delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, args)

    # Enable to force a sync of algorithm state to debug
    if 0 and args.mpi == 1:
        total_num_nodal_moves_itr = comm.bcast(total_num_nodal_moves_itr, root=0)
        partition = comm.bcast(partition, root=0)
        M = comm.bcast(M, root=0)
        block_degrees_out = comm.bcast(block_degrees_out, root=0)
        block_degrees_in = comm.bcast(block_degrees_in, root=0)
        block_degrees = comm.bcast(block_degrees, root=0)
    
    # compute the global entropy for determining the optimal number of blocks
    overall_entropy = compute_overall_entropy(M, block_degrees_out, block_degrees_in, num_blocks, N, E)

    if verbose:
        move_end_time = timeit.default_timer()
        merge_time = merge_end_time - merge_start_time
        move_time = move_end_time - move_start_time
        timing_stats['time_in_merge'] += merge_time
        timing_stats['time_in_move'] += move_time
        move_rate = total_num_nodal_moves_itr / move_time
        print("Total number of nodal moves: {:3d}, overall_entropy: {:0.2f}, merge_time: {:0.3f}, move_time: {:0.3f} secs, moves_per_sec: {:4.3f}".format(total_num_nodal_moves_itr, overall_entropy, merge_time, move_time, move_rate))

    if args.visualize_graph:
        graph_object = plot_graph_with_partition(out_neighbors, partition, graph_object)

    return overall_entropy, n_proposals_evaluated, n_merges, total_num_nodal_moves_itr, M, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, derivative_based_stop, dS_dn


def load_graph_parts(input_filename, args):

    if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
        print("File doesn't exist: '{}'!".format(input_filename))
        sys.exit(1)

    if args.parts >= 1:
        true_partition_available = False
        print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
        out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, strm_piece_num=1)
        for part in range(2, args.parts + 1):
                print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
                out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
    else:
        true_partition_available = True
        if true_partition_available:
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available)
        else:
            out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=true_partition_available)
            true_partition = None

    return out_neighbors, in_neighbors, N, E, true_partition


def find_optimal_partition(out_neighbors, in_neighbors, N, E, self_edge_weights, args, stop_at_bracket = False, verbose = 0, alg_state = None, num_block_reduction_rate = 0.50, min_number_blocks = 0):

    if verbose > -1:
        print('Number of nodes %d edges %d ' % (N,E))
        if verbose > 0:
            density = E / float(N*N)
            max_in_deg = max((len(i) for i in in_neighbors))
            max_out_deg = max((len(i) for i in out_neighbors))
            print("Graph density %f max_in_degree %d max_out_degree %d" % (density,max_in_deg,max_out_deg))

    # partition update parameters
    args.beta = 3.0  # exploitation versus exploration (higher value favors exploitation)

    # agglomerative partition update parameters
    num_agg_proposals_per_block = 10  # number of agglomerative merge proposals per block
    # num_block_reduction_rate is fraction of blocks to reduce until the golden ratio bracket is established

    # nodal partition updates parameters

    delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                     # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    args.delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    vertex_num_in_neighbor_edges = np.empty(N, dtype=int)
    vertex_num_out_neighbor_edges = np.empty(N, dtype=int)
    vertex_num_neighbor_edges = np.empty(N, dtype=int)

    # XXX I think it is ok to apply unique to each node's neighbors. But it may be an issue if it messes up the neighbor selection probabilities.
    vertex_neighbors = [np.unique(np.concatenate((out_neighbors[i], in_neighbors[i])), axis=0) for i in range(N)]

    for i in range(N):
        vertex_num_out_neighbor_edges[i] = sum(out_neighbors[i][:,1])
        vertex_num_in_neighbor_edges[i] = sum(in_neighbors[i][:,1])
        vertex_num_neighbor_edges[i] = vertex_num_out_neighbor_edges[i] + vertex_num_in_neighbor_edges[i]


    optimal_num_blocks_found = False

    if not alg_state:
        # initialize by putting each node in its own block (N blocks)
        num_blocks = N
        partition = np.arange(num_blocks, dtype=int)

        if args.sparse == 2:
            if num_blocks >= compressed_threshold:
                use_compressed = 1
            else:
                use_compressed = 0
        else:
            use_compressed = (args.sparse != 0)

        # initialize edge counts and block degrees
        interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors,
                                     num_blocks,
                                     partition,
                                     use_compressed, args)
        if args.sanity_check:
            sanity_check_state(partition, out_neighbors, interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees)

        # initialize items before iterations to find the partition with the optimal number of blocks
        hist, graph_object = initialize_partition_variables()

        initial_num_block_reduction_rate = max(args.initial_block_reduction_rate, num_block_reduction_rate)

        num_blocks_to_merge = int(num_blocks * initial_num_block_reduction_rate)
        golden_ratio_bracket_established = False
        delta_entropy_threshold = delta_entropy_threshold1
        n_proposals_evaluated = 0
        total_num_nodal_moves = 0
    elif len(alg_state) == 1:
        hist, graph_object = initialize_partition_variables()
        partition = alg_state[0]
        num_blocks = 1 + np.max(partition)

        golden_ratio_bracket_established = False
        delta_entropy_threshold = delta_entropy_threshold1
        n_proposals_evaluated = 0
        total_num_nodal_moves = 0
        use_compressed = (args.sparse != 0)

        interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors, num_blocks, partition, use_compressed, args)

        overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, N, E)
        partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, hist, optimal_num_blocks_found = \
               prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                        block_degrees_out, block_degrees_in, num_blocks, hist,
                                                        B_rate = num_block_reduction_rate )
    else:
        # resume search from a previous partition state
        (hist, num_blocks, overall_entropy, partition, interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves) = alg_state

    args.n_proposal = 1

    while not optimal_num_blocks_found:
        # Using multiple target_blocks can be useful if you want to estimate the derivative of entropy to try to more quickly find a convergence.
        # Must be in decreasing order because reducing by carrying out merges modifies state.
        # target_blocks = [num_blocks - num_blocks_to_merge + 1, num_blocks - num_blocks_to_merge, num_blocks - num_blocks_to_merge - 1]
        target_blocks = [num_blocks - num_blocks_to_merge]

        (overall_entropy,n_proposals_itr,n_merges_itr,total_num_nodal_moves_itr, \
         interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, derivative_based_stop, dS_dn) \
         = entropy_for_block_count(num_blocks, target_blocks,
                                   delta_entropy_threshold,
                                   interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in,
                                   out_neighbors, in_neighbors, N, E,
                                   vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights,
                                   partition, args, verbose)

        if verbose:
            print("num_blocks = %s num_blocks_merged = %s M.shape = %s" % (num_blocks, num_blocks_merged, str(interblock_edge_count.shape)))

        num_blocks -= num_blocks_merged
        total_num_nodal_moves += total_num_nodal_moves_itr
        n_proposals_evaluated += n_proposals_itr

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try

        partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, hist, optimal_num_blocks_found = \
            prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                     block_degrees_out, block_degrees_in, num_blocks, hist,
                                                     num_block_reduction_rate)

        (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist


        if verbose:
            print('Overall entropy: {} Number of blocks: {} Proposals evaluated: {} Overall nodal moves: {}'.format(old_overall_entropy, old_num_blocks, n_proposals_evaluated, total_num_nodal_moves))

            if optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(num_blocks))

        if np.all(np.isfinite(old_overall_entropy)):
            delta_entropy_threshold = delta_entropy_threshold2
            if not golden_ratio_bracket_established:
                golden_ratio_bracket_established = True
                print("Golden ratio found at blocks %s at time %4.4f entropy %s" % (old_num_blocks, timeit.default_timer() - t_prog_start, old_overall_entropy))

            if stop_at_bracket:
                break

        if num_blocks <= min_number_blocks:
            break

    alg_state = (hist,num_blocks,overall_entropy,partition,interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves)

    return alg_state, partition

def find_optimal_partition_wrapper(tup):
    args = syms['args']

    args.t_merge = max(1, args.t_merge // args.decimation)
    args.t_move = max(1, args.t_move // args.decimation)

    out_neighbors, in_neighbors, N, E, true_partition = tup
    self_edge_weights = find_self_edge_weights(N, out_neighbors)
    return find_optimal_partition(out_neighbors, in_neighbors, N, E, self_edge_weights, args, stop_at_bracket = True, verbose = min(0, args.verbose - 1))


# See: https://stackoverflow.com/questions/17223301/python-multiprocessing-is-it-possible-to-have-a-pool-inside-of-a-pool
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

from concurrent.futures import ProcessPoolExecutor

def merge_partitions(partitions, stop_pieces, out_neighbors, verbose):
    """
    Create a unified graph block partition from the supplied partition pieces into a partiton of size stop_pieces.
    """

    if args.sparse == 2:
        if num_blocks >= compressed_threshold:
            use_compressed = 1
        else:
            use_compressed = 0
    else:
        use_compressed = (args.sparse != 0)

    pieces = len(partitions)
    N = sum(len(i) for i in partitions)

    # The temporary partition variable is for the purpose of computing M.
    # The axes of M are concatenated block ids from each partition.
    # And each partition[i] will have an offset added to so all the interim partition ranges are globally unique.
    #
    partition = np.zeros(N, dtype=int)

    while pieces > stop_pieces:

        Bs = [max(partitions[i]) + 1 for i in range(pieces)]
        B =  sum(Bs)

        partition_offsets = np.zeros(pieces, dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        if verbose > 1:
            print("")
            print("Reconstitute graph from %d pieces B[piece] = %s" % (pieces,Bs))
            print("partition_offsets = %s" % partition_offsets)

        # It would likely be faster to re-use already computed values of M from pieces:
        #     M[ 0:B0,     0:B0   ] = M_0
        #     M[B0:B0+B1, B0:B0+B1] = M_1
        # Instead of relying on initialize_edge_counts.

        M, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors, B, partition, use_compressed)

        if verbose > 2:
            print("M.shape = %s, M = \n%s" % (str(M.shape),M))

        next_partitions = []
        for i in range(0, pieces, 2):
            print("Merge piece %d and %d into %d" % (i, i + 1, i // 2))
            partitions[i],_ = merge_two_partitions(M, block_degrees_out, block_degrees_out, block_degrees_out,
                                                   partitions[i], partitions[i + 1],
                                                   partition_offsets[i], partition_offsets[i + 1],
                                                   Bs[i], Bs[i + 1],
                                                   verbose,
                                                   use_compressed)
            next_partitions.append(np.concatenate((partitions[i], partitions[i+1])))

        partitions = next_partitions
        pieces //= 2

    return partitions



def merge_two_partitions(M, block_degrees_out, block_degrees_in, block_degrees, partition0, partition1, partition_offset_0, partition_offset_1, B0, B1, verbose):
    """
    Merge two partitions each from a decimated piece of the graph.
    Note
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    block count matrix between all the blocks, of which partition 0 and partition 1 are just subsets
        partition_offset_0 and partition_offset_1 are the starting offsets within M of each partition piece
    """
    # Now reduce by merging down blocks from partition 0 into partition 1.
    # This requires computing delta_entropy over all of M (hence the partition_offsets are needed).

    delta_entropy = np.empty((B0,B1))

    for r in range(B0):
        current_block = r + partition_offset_0

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1, sort = False)
        out_idx, out_weight = take_nonzero(M, current_block, 0, sort = False)

        block_neighbors = np.concatenate((in_idx, out_idx))
        block_neighbor_weights = np.concatenate((in_weight, out_weight))

        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        for s in range(B1):
            proposal = s + partition_offset_1

            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col \
                = compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal,
                                                                     out_idx, out_weight,
                                                                     in_idx, in_weight,
                                                                     M[current_block, current_block], agg_move = 1)

            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(current_block,
                                            proposal,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[r, s] = compute_delta_entropy(current_block, proposal, M,
                                                        new_M_r_row,
                                                        new_M_s_row,
                                                        new_M_r_col,
                                                        new_M_s_col,
                                                        block_degrees_out,
                                                        block_degrees_in,
                                                        block_degrees_out_new,
                                                        block_degrees_in_new)

    best_merge_for_each_block = np.argmin(delta_entropy, axis = 1)

    if verbose > 2:
        print("delta_entropy = \n%s" % delta_entropy)
        print("best_merge_for_each_block = %s" % best_merge_for_each_block)

    delta_entropy_for_each_block = delta_entropy[np.arange(delta_entropy.shape[0]), best_merge_for_each_block]

    # Global number of blocks (when all pieces are considered together).
    num_blocks = M.shape[0]
    num_blocks_to_merge = B0
    best_merges = delta_entropy_for_each_block.argsort()

    # Note: partition0 will be modified in carry_out_best_merges
    (partition, num_blocks) = carry_out_best_merges(delta_entropy_for_each_block,
                                                    best_merges,
                                                    best_merge_for_each_block + partition_offset_1,
                                                    partition0,
                                                    num_blocks,
                                                    num_blocks_to_merge, verbose=(verbose > 2))

    return partition, num_blocks

def naive_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors = None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
        t_part = 0.0

        if part == 1:
            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
        else:
            out_neighbors, in_neighbors, N, E = \
                    load_graph(input_filename,
                               load_true_partition=0,
                               strm_piece_num=part,
                               out_neighbors=out_neighbors,
                               in_neighbors=in_neighbors)

        # Run to ground.
        print('Running partition for part %d N %d E %d' % (part,N,E))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = None)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        t_all_parts += t_part

        if part == args.parts:
            print('Evaluate final partition.')
        else:
            print('Evaluate part %d' % part)

        precision,recall = evaluate_partition(true_partition, partition)
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts


def copy_alg_state(alg_state):
    # Create a deep copy of algorithmic state.
    (hist, num_blocks, overall_entropy, partition, interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves) = alg_state

    (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

    hist_copy = tuple((i.copy() for i in hist))
    try:
        num_blocks_copy = num_blocks.copy()
    except AttributeError:
        num_blocks_copy = num_blocks
    overall_entropy_copy = overall_entropy.copy()
    partition_copy = partition.copy()
    interblock_edge_count_copy = interblock_edge_count.copy()
    block_degrees_out_copy = block_degrees_out.copy()
    block_degrees_in_copy = block_degrees_in.copy()
    block_degrees_copy = block_degrees.copy()
    golden_ratio_bracket_established_copy = golden_ratio_bracket_established # bool
    delta_entropy_threshold_copy = delta_entropy_threshold # float
    num_blocks_to_merge_copy = num_blocks_to_merge # int
    optimal_num_blocks_found_copy = optimal_num_blocks_found # bool
    n_proposals_evaluated_copy = n_proposals_evaluated # int
    total_num_nodal_moves_copy = total_num_nodal_moves # int


    alg_state_copy = (hist_copy, num_blocks_copy, overall_entropy_copy, partition_copy, interblock_edge_count_copy, block_degrees_out_copy, block_degrees_in_copy, block_degrees_copy, golden_ratio_bracket_established_copy, delta_entropy_threshold_copy, num_blocks_to_merge_copy, optimal_num_blocks_found_copy, n_proposals_evaluated_copy, total_num_nodal_moves_copy)

    return alg_state_copy



def incremental_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors,alg_state = None,None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        t_part = 0.0

        if part == 1:
            print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
            min_number_blocks = N / 2
        else:
            # Load true_partition here so the sizes of the arrays all equal N.
            if alg_state:
                print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

                out_neighbors, in_neighbors, N, E, alg_state,t_compute = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = alg_state)
                t_part += t_compute
                print("Intermediate load_graph compute time for part %d is %f" % (part,t_compute))
                t0 = timeit.default_timer()
                hist = alg_state[0]
                (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

                print("Incrementally updated alg_state for part %d" %(part))
                print('New Overall entropy: {}'.format(old_overall_entropy))
                print('New Number of blocks: {}'.format(old_num_blocks))
                print("")

                # Check for self edges
                self_edge_weights = find_self_edge_weights(N, out_neighbors)

                verbose = 1

                n_thread_merge = args.t_merge
                n_thread_move = args.t_move

                vertex_num_in_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_out_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_neighbor_edges = np.empty(N, dtype=int)
                vertex_neighbors = [np.unique(np.concatenate((out_neighbors[i], in_neighbors[i])), axis=0) for i in range(N)]
                for i in range(N):
                    vertex_num_out_neighbor_edges[i] = sum(out_neighbors[i][:,1])
                    vertex_num_in_neighbor_edges[i] = sum(in_neighbors[i][:,1])
                    vertex_num_neighbor_edges[i] = vertex_num_out_neighbor_edges[i] + vertex_num_in_neighbor_edges[i]
                #delta_entropy_threshold = delta_entropy_threshold1 = 5e-4
                delta_entropy_threshold = 1e-4

                for j in [0,2,1]:
                    if old_interblock_edge_count[j] == []:
                        continue

                    print("Updating previous state in bracket history.")

                    M_old = old_interblock_edge_count[j].copy()
                    M = old_interblock_edge_count[j]
                    partition = old_partition[j]
                    block_degrees_out = old_block_degrees_out[j]
                    block_degrees_in = old_block_degrees_in[j]
                    block_degrees = old_block_degrees[j]
                    num_blocks = old_num_blocks[j]
                    overall_entropy = old_overall_entropy[j]

                    total_num_nodal_moves_itr = nodal_moves_parallel(n_thread_move, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, self_edge_weights, verbose, args)

                t1 = timeit.default_timer()
                print("Intermediate nodal move time for part %d is %f" % (part,(t1-t0)))
                t_part += (t1 - t0)
            else:
                # We are not doing partitioning yet. Just wait.
                out_neighbors, in_neighbors, N, E, true_partition = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = None)

            print("Loaded piece %d N %d E %d" % (part,N,E))
            min_number_blocks = int(min_number_blocks / 2)

        print('Running partition for part %d N %d E %d and min_number_blocks %d' % (part,N,E,min_number_blocks))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition,alg_state = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 1, alg_state = alg_state, min_number_blocks = min_number_blocks)
        min_number_blocks /= 2

        alg_state_copy = copy_alg_state(alg_state)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until save point for part %d is %f" % (part,(t1-t0)))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = alg_state_copy, min_number_blocks = 5)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until completion for part %d is %f" % (part,(t1-t0)))

        print('Evaluate part %d' % (part))
        precision,recall = evaluate_partition(true_partition, partition)

        t_all_parts += t_part
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts

def do_main(args):
    global syms, t_prog_start

    if args.threads > 0:
        if args.t_merge == 0:
            args.t_merge = args.threads
        if args.t_move == 0:
            args.t_move = args.threads

    min_number_blocks = args.min_number_blocks
    t_prog_start = timeit.default_timer()

    if args.verbose > 0:
        print("Program start at %s sec." % (t_prog_start))
        print("Started: " + time.strftime("%a %b %d %Y %H:%M:%S %Z"))
        print("Python version: " + sys.version)
        d = vars(args)
        args_sorted = sorted([i for i in d.items()])
        print("Arguments: {" + "".join(("%s : %s, " % (k,v) for k,v in args_sorted)) + "}\n")

    np.seterr(all='raise')

    if args.seed != 0:
        numpy.random.seed(args.seed % 4294967295)
    else:
        numpy.random.seed((os.getpid() + int(timeit.default_timer() * 1e6)) % 4294967295)

    input_filename = args.input_filename
    args.visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    try:
        cols,lines = shutil.get_terminal_size()
        np.set_printoptions(linewidth=cols)
    except AttributeError:
        pass

    print("Module Info: " + compressed_array.info())
    
    if args.parts <= 1:
        out_neighbors, in_neighbors, N, E, true_partition = load_graph_parts(input_filename, args)


        if not args.test_resume:
            t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, min_number_blocks=min_number_blocks)
        else:
            print("")
            print("Test stop functionality.")
            print("")
            t_elapsed_partition,partition,alg_state = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 1, min_number_blocks = min_number_blocks)

            print("")
            print("Resume bracket search.")
            print("")

            t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = alg_state)

        if args.skip_eval:
            precision,recall = -1.0,-1.0
        else:
            precision,recall = evaluate_partition(true_partition, partition)

        return t_elapsed_partition,precision,recall
    else:
        if args.naive_streaming:
            t_compute = naive_streaming(args)
        else:
            t_compute = incremental_streaming(args)
        return t_compute
    return


def partition_static_mpi(out_neighbors, in_neighbors, N, E, true_partition, self_edge_weights, args, stop_at_bracket=0, alg_state=None, min_number_blocks=0):
    comm = MPI.COMM_WORLD
    mpi_procs = 2**int(np.log2(comm.size))

    # MPI-based decimation is only supported for powers of 2
    if comm.rank >= mpi_procs:
        comm.Barrier()
        return

    print("Hello! I am rank %4d from %4d running in total limit is %d" % (comm.rank, comm.size, mpi_procs))

    t_prog_start = timeit.default_timer()
    alg_state, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E,
                                    self_edge_weights,
                                    args, stop_at_bracket = stop_at_bracket,
                                    verbose = args.verbose,
                                    alg_state = alg_state, num_block_reduction_rate = 0.50, min_number_blocks=min_number_blocks)
    return alg_state, M_bracket

def partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket=0, alg_state=None, min_number_blocks=0):
    global syms, t_prog_start

    if args.verbose > 1:
        from collections import Counter
        print("Overall true partition statistics:")
        print("[" + "".join(("%5d : %3d, " % (i,e,) for i,e in sorted([(e,i) for (i,e) in Counter(true_partition).items()]))) + "]\n")


    if args.predecimation > 1:
        out_neighbors, in_neighbors, N, E, true_partition = \
                                decimate_graph(out_neighbors, in_neighbors, true_partition,
                                               decimation = args.predecimation, decimated_piece = 0)

    self_edge_weights = find_self_edge_weights(N, out_neighbors)

    if args.mpi == 1:
        decimation = 1
        alg_state, M_bracket = partition_static_mpi(out_neighbors, in_neighbors,
                                                    N, E, true_partition, self_edge_weights,
                                                    args,
                                                    stop_at_bracket, alg_state,
                                                    min_number_blocks)

        partition = alg_state[0][0][1]
        t_prog_end = timeit.default_timer()
    elif args.mpi == 2:
        # Old MPI version
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_procs = 2**int(np.log2(comm.size))

        # MPI-based decimation is only supported for powers of 2
        if comm.rank >= mpi_procs:
            comm.Barrier()
            return

        print("Hello! I am rank %4d from %4d running in total limit is %d" % (comm.rank, comm.size, mpi_procs))

        decimation = mpi_procs
        out_neighbors_piece, in_neighbors_piece, N_piece, E_piece, true_partition_piece \
            = decimate_graph(out_neighbors, in_neighbors, true_partition,
                             decimation, decimated_piece = comm.rank)


        t_prog_start = timeit.default_timer()
        alg_state, M = find_optimal_partition(out_neighbors_piece, in_neighbors_piece, N_piece, E_piece, self_edge_weights, args, stop_at_bracket = False, verbose = args.verbose)
        t_prog_end = timeit.default_timer()

        partition = alg_state[0][0][1]

        if comm.rank != 0:
            comm.send(true_partition_piece, dest=0, tag=11)
            comm.send(partition, dest=0, tag=11)
            comm.Barrier()
            return
        else:
            true_partitions = [true_partition_piece] + [comm.recv(source=i, tag=11) for i in range(1, mpi_procs)]
            partitions = [partition] + [comm.recv(source=i, tag=11) for i in range(1, mpi_procs)]
            comm.Barrier()

    elif args.decimation > 1:
        decimation = args.decimation

        # Re-start timer after decimation is complete
        t_prog_start = timeit.default_timer()

        pieces = [decimate_graph(out_neighbors, in_neighbors, true_partition, decimation, i) for i in range(decimation)]
        _,_,_,_,true_partitions = zip(*pieces)

        if args.verbose > 1:
            for j,_ in enumerate(true_partitions):
                print("Overall true partition %d statistics:" % (j))
                print("[" + "".join(("%5d : %3d, " % (i,e,) for i,e in sorted([(e,i) for (i,e) in Counter(true_partitions[j]).items()]))) + "]\n")


        syms = {}
        syms['args'] = args

        with ProcessPoolExecutor(decimation) as pool:
            results = pool.map(find_optimal_partition_wrapper, pieces)

        alg_states,partitions = (list(i) for i in zip(*results))
    else:
        decimation = 1
        t_prog_start = timeit.default_timer()

        alg_state, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, \
                                                      self_edge_weights, \
                                    args, stop_at_bracket = stop_at_bracket, verbose = args.verbose, \
                                    alg_state = alg_state, num_block_reduction_rate = 0.50, min_number_blocks=min_number_blocks)

        partition = alg_state[0][0][1]
        t_prog_end = timeit.default_timer()

        if args.test_decimation > 0:
            decimation = args.test_decimation
            true_partitions = [true_partition[i::decimation] for i in range(decimation)]
            partitions = [partition[i::decimation] for i in range(decimation)]



    # Either multiprocess pool or MPI results need final merging.
    if decimation > 1:
        if args.verbose > 1:
            for i in range(decimation):
                print("")
                print("Evaluate decimated subgraph %d:" % i)
                evaluate_partition(true_partitions[i], partitions[i])

        t_decimation_merge_start = timeit.default_timer()


        # Merge all pieces into a smaller number.
        partitions = merge_partitions(partitions,
                                      4, out_neighbors, args.verbose)

        # Merge piece into  big partition and then merge down.
        Bs = [max(i) + 1 for i in partitions]
        partition = np.zeros(N, dtype=int)
        partition_offsets = np.zeros(len(partitions), dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        partition = np.concatenate([partitions[i] + partition_offsets[i] for i in range(len(partitions))])

        t_decimation_merge_end = timeit.default_timer()
        print("Decimation merge time is %3.5f" % (t_decimation_merge_end - t_decimation_merge_start))

        t_final_partition_search_start = timeit.default_timer()

        alg_state,partition = find_optimal_partition(out_neighbors, in_neighbors, N, E,
                                                     self_edge_weights,
                                                     args,
                                                     stop_at_bracket = False,
                                                     verbose = args.verbose,
                                                     alg_state = [partition],
                                                     num_block_reduction_rate = 0.25)

        t_final_partition_search_end = timeit.default_timer()
        t_prog_end = timeit.default_timer()
        print("Final partition search took %3.5f" % (t_final_partition_search_end - t_final_partition_search_start))

    t_elapsed_partition = t_prog_end - t_prog_start
    print('\nGraph partition took %.4f seconds' % (t_elapsed_partition))

    print('Timing stats:', [(k,v) for (k,v) in timing_stats.items()])
    if 'pop_cnt' in timing_stats:
        print("Pop / move ratio: %3.3f" % (timing_stats['pop_cnt'] / timing_stats['nodal_moves']))

    if stop_at_bracket:
        return t_elapsed_partition,partition,alg_state
    else:
        return t_elapsed_partition,partition

# See: https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, required=False, default=0, help="Configure number of threads for both agglomerative merge and nodal movements.")
    parser.add_argument("--t-merge", type=int, required=False, default=0, help="Configure number of threads for both agglomerative merge phase (overrides --threads).")
    parser.add_argument("--t-move", type=int, required=False, default=0, help="Configure number of threads for nodal movement phase (overrides --threads)")

    parser.add_argument("-p", "--parts", type=int, required=False, default=0)
    parser.add_argument("-d", "--decimation", type=int, required=False, default=0)
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, help="Verbosity level.")
    parser.add_argument("-g", "--node-propose-batch-size", type=int, required=False, default=4)
    parser.add_argument("--sparse", type=int, required=False, default=0)
    parser.add_argument("-s", "--sort", type=int, required=False, default=0)
    parser.add_argument("-S", "--seed", type=int, required=False, default=-1)
    parser.add_argument("--mpi", type=int, required=False, default=0)
    parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")

    # Debugging options
    parser.add_argument("--min-number-blocks", type=int, required=False, default=0, help="Force stop at this many blocks instead of searching for optimality.")
    parser.add_argument("--initial-block-reduction-rate", type=float, required=False, default=0.50)
    parser.add_argument("--profile", type=int, required=False, help="Profiling level 0=disabled, 1=main, 2=workers.", default=0)
    parser.add_argument("--test-decimation", type=int, required=False, default=0)
    parser.add_argument("--predecimation", type=int, required=False, default=0)
    parser.add_argument("--debug", type=int, required=False, default=0)
    parser.add_argument("--test-resume", type=int, required=False, default=0)
    parser.add_argument("--naive-streaming", type=int, required=False, default=0)
    parser.add_argument("--min-nodal-moves-ratio", type=float, required=False, default=0.0, help="Break nodal move loop early if the number of accepted moves is below this fraction of the number of nodes.")
    parser.add_argument("--skip-eval", type=int, required=False, default=0, help="Skip partition evaluation.")
    parser.add_argument("--max-num-nodal-itr", type=int, required=False, default=100, help="Maximum number of iterations during nodal moves.")
    parser.add_argument("--sanity-check", type=int, required=False, default=0, help="Full recompute interblock edge counts and block counts, and compare against differentially computed version.")

    # Arguments for thread control
    parser.add_argument("--preallocate", type=int, required=False, default=0, help="Whether to preallocate memory.")
    parser.add_argument("--blocking", type=int, required=False, default=1, help="Whether to use blocking waits during nodal moves.")
    parser.add_argument("--finegrain", type=int, required=False, default=0, help="Try to use finegrain locks instead of a single lock.")
    parser.add_argument("--critical", type=int, required=False, default=2, help="Which critical section to use. 0 is the widest, 2 is the narrowest.")

    args = parser.parse_args()

    if args.debug:
        sys.excepthook = info

    if args.profile:
        import cProfile
        cProfile.run('do_main(args)', filename="partition_baseline_main.prof")
    else:
        do_main(args)
