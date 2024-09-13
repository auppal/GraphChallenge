import multiprocessing as mp
from multiprocessing import Process
import timeit, resource
import sys, itertools, os, time, traceback
from collections import defaultdict
from partition_baseline_main import do_main
import time
import argparse
import re
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import contextlib
@contextlib.contextmanager
def redirect_streams(target):
    os.dup2(target.fileno(), sys.stdout.fileno())
    os.dup2(target.fileno(), sys.stderr.fileno())
    yield


base_args = {'debug' : 0, 'decimation' : 0,
             'input_filename' : '../../data/static/simulated_blockmodel_graph_100_nodes',
             'initial_block_reduction_rate' : 0.50,
             'merge_method' : 0, 'mpi' : 0, 'node_move_update_batch_size' : 1, 'node_propose_batch_size' : 64,
             'parts' : 0, 'predecimation' : 0, 'profile' : 0, 'seed' : 0, 'sort' : 0, 'diet': 1, 'defunct' : 0,
             'sparse' : 1, 'test_decimation' : 0, 'threads' : 0, 'verbose' : 2, 'test_resume' : 0, 'merge_proposals_per_block' : 10, 'min_nodal_moves_ratio' : 0.0, 'min_number_blocks' : 0, 't_merge' : 0, 't_move' : 0, 'skip_eval' : 0, 'max_num_nodal_itr' : 100, 'critical' : 0, 'blocking' : 1, 'finegrain' : 0, 'sanity_check' : 0, 'debug_memory' : 0, 'debug_mpi' : 0, 'output_file' : ""}

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


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
class NonDaemonicPool(mp.pool.Pool):
    Process = NoDaemonProcess

from concurrent.futures import ProcessPoolExecutor as Pool

shortname = {'decimation' : 'd',
             'input_filename' : 'F',
             'iteration' : 'itr',
             'merge_method' : 'm',
             'parts' : 'p',
             'predecimation' : 'predec',
             'parallel_phase' : 'P',
             'node_move_update_batch_size' : 'b',
             'node_propose_batch_size' : 'g',
             'sort' : 's',
             'seed' : 'S',
             'threads' : 't',
             'verbose' : 'v'}

def outputname(args):
    # XXX Temporary until too long filename is fixed.
    return time.strftime("output-%Y-%m-%d-%H%M%SZ", time.gmtime())
    out = 'out'
    for k,v in args:
        if k == 'input_filename':
            v = os.path.basename(v)
        if k in shortname:
            k = shortname[k]
        if k == 't' or k == 'itr':
            out += ('-%s%02d' % (k,v))
        else:
            out += ('-%s%s' % (k,v))
    return out


def child_func(queue, fout, func, args):
    rc = 0
    wall_time = 0
    rusage_self = None
    rusage_children = None

    t0 = timeit.default_timer()

    try:
        with redirect_streams(fout):
            func_result = func(args)
    except:
        traceback.print_exc()
        traceback.print_exc(file=fout)
        func_result = None
        rc = 1

    t1 = timeit.default_timer()
    wall_time = t1 - t0
    fout.flush()

    rusage_self = resource.getrusage(resource.RUSAGE_SELF)
    rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    queue.put((rc, wall_time, rusage_self, rusage_children, func_result))
    sys.exit(rc)

def profile_child(out_dir, func, args):
    outname = out_dir + '/' + outputname(args)
    fout = open(outname, 'w')
    args = Bunch(args)
    queue = mp.Queue()
    p = Process(target=child_func, args=(queue, fout, func, args))
    p.start()
    rc,wall_time,rusage_self,rusage_children,func_result = queue.get()
    p.join()
    return outname,rc,wall_time,rusage_self,rusage_children,func_result

def profile_wrapper(tup):
    (out_dir, args) = tup
    return profile_child(out_dir, do_main, args)

def run_test(out_dir, base_args, input_files, iterations, threads, max_jobs = 1):
    results = {}

    work_list = [i for i in itertools.product(input_files, threads, iterations)]

    arg_list = [base_args.copy() for i in work_list]

    for i,(input_filename,thread,iteration) in enumerate(work_list):
        arg_list[i]['input_filename'] = input_filename
        arg_list[i]['threads'] = thread
        arg_list[i] = tuple(sorted((j for j in arg_list[i].items()))) + (('iteration', iteration),)

    pool = Pool(max_jobs)
    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children,func_result) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss. Function result is %s" % (t_elp, mem_rss, str(func_result)))
            print("")
        else:
            print(args)
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def run_sweep_test(out_dir, base_args, input_files, iterations, threads, reduction_rates, max_jobs = 1):
    results = {}

    work_list = [i for i in itertools.product(input_files, threads, iterations, reduction_rates)]

    arg_list = [base_args.copy() for i in work_list]

    for i,(input_filename,thread,iteration,reduction_rate) in enumerate(work_list):
        arg_list[i]['input_filename'] = input_filename
        arg_list[i]['threads'] = thread
        arg_list[i]['initial_block_reduction_rate'] = reduction_rate
        arg_list[i] = tuple(sorted((j for j in arg_list[i].items()))) + (('iteration', iteration),)

    pool = Pool(max_jobs)

    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss" % (t_elp, mem_rss))
            print("")
        else:
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def update_dict(x, y):
    x.update(y)
    return x

def run_var_test(out_dir, base_args, var_args, max_jobs = 1, override_args={}):
    """
    Run a test suite sweeping across a series of parameter values stored in the var_args dict which override the defaults in base_args.
    """
    results = {}

    # Form the cartesian product of all values for keys arranged in alphabetical order.
    work_list = [i for i in itertools.product(*(i[1] for i in var_args))]

    # Create dicts which will override the default argument dict.
    # The keys are (j[0] for j in var_args), but not used in a generator to avoid exhausting it prematurely.

    var_args_d = [{k : v for k,v in zip((j[0] for j in var_args), i)} for i in work_list]
    arg_list = [update_dict(base_args.copy(), d) for d in var_args_d]
    arg_list = [update_dict(d, override_args) for d in arg_list]
    for i in arg_list:
        print(i)
    sys.exit(0)

    # Convert to tuples (for later indexing).
    arg_list = [tuple(sorted((j for j in i.items()))) for i in arg_list]

    pool = Pool(max_jobs)
    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children,func_result) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss. Function result is %s" % (t_elp, mem_rss, str(func_result)))
            print("")
        else:
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def print_results(results):
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1:]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+", type=str, default=[])
    parser.add_argument("-a", "--arg", type=str, default=[], action="append", help="Test case argument overrides in key:value form.")
    args = parser.parse_args()
    override_args = dict((i.split(":")[0],int(i.split(":")[1])) for i in args.arg)
    
    yyyymmdd = time.strftime("%Y-%m-%d")
    out_dir = 'out-' + yyyymmdd
    try: os.mkdir(out_dir)
    except FileExistsError: pass

    input_files = ('../../data/static/simulated_blockmodel_graph_100_nodes',
                   '../../data/static/simulated_blockmodel_graph_500_nodes',
                   '../../data/static/simulated_blockmodel_graph_1000_nodes',
                   '../../data/static/simulated_blockmodel_graph_5000_nodes',
                   '../../data/static/simulated_blockmodel_graph_20000_nodes',
                   '../../data/static/simulated_blockmodel_graph_50000_nodes',
                   '../../data/static/simulated_blockmodel_graph_100000_nodes'
    )

    N = {100: '../../data/static/simulated_blockmodel_graph_100_nodes',
         500: '../../data/static/simulated_blockmodel_graph_500_nodes',
         1000: '../../data/static/simulated_blockmodel_graph_1000_nodes',
         5000: '../../data/static/simulated_blockmodel_graph_5000_nodes',
         20000: '../../data/static/simulated_blockmodel_graph_20000_nodes',
         50000: '../../data/static/simulated_blockmodel_graph_50000_nodes',
         100000: '../../data/static/simulated_blockmodel_graph_100000_nodes',
         1000000: '../../data/static/simulated_blockmodel_graph_1000000_nodes'
    }

    small_files = input_files[:3]
    big_files = input_files[3:]

    iterations = range(1)

    results = {}

    results_f = open(out_dir + time.strftime("/results-%Y-%m-%d-%H%M%SZ.pickle", time.gmtime()), 'wb')

    if 'compare' in args.command:
        fa = args.command[args.command.index("compare") + 1]
        fb = args.command[args.command.index("compare") + 2]
        a = pickle.load(open(fa, "rb"))
        b = pickle.load(open(fb, "rb"))

        parttimes_a = defaultdict(list)
        parttimes_b = defaultdict(list)

        for i in a.items():
            k = i[0]
            for j in k:
                if j[0] == 'input_filename':
                    nodes = int(re.match('.*_graph_(\d+)_nodes', j[1]).group(1))
            outname_a,runtime_a,maxrss_a,(parttime_a,prec_a,recall_a) = a[k]
            outname_b,runtime_b,maxrss_b,(parttime_b,prec_b,recall_b) = b[k]
            parttimes_a[nodes].append(parttime_a)
            parttimes_b[nodes].append(parttime_b)

        print("Comparison",fa,fb)
        print("    Compute:")
        print("                         Before:                   After:                Speedup:")
        for nodes in sorted(parttimes_a.keys()):
            mean_parttime_a = np.mean(parttimes_a[nodes])
            mean_parttime_b = np.mean(parttimes_b[nodes])
            std_parttime_a = np.std(parttimes_a[nodes])
            std_parttime_b = np.std(parttimes_b[nodes])
            print("At  {:4.0f}k ({:3} runs): {:11.5f} +- {:3.5f}  / {:11.5f} +- {:3.5f} {:10.5f}".format(nodes/1e3, len(parttimes_a[nodes]), mean_parttime_a, std_parttime_a, mean_parttime_b, std_parttime_b, mean_parttime_a / mean_parttime_b))


        mem_a = defaultdict(list)
        mem_b = defaultdict(list)

        print("")
        print("    Memory:")
        print("                         Before:                   After:                Ratio:")
        for i in a.items():
            k = i[0]
            for j in k:
                if j[0] == 'input_filename':
                    nodes = int(re.match('.*_graph_(\d+)_nodes', j[1]).group(1))
            outname_a,runtime_a,maxrss_a,(parttime_a,prec_a,recall_a) = a[k]
            outname_b,runtime_b,maxrss_b,(parttime_b,prec_b,recall_b) = b[k]
            mem_a[nodes].append(maxrss_a)
            mem_b[nodes].append(maxrss_b)

        for nodes in sorted(mem_a.keys()):
            mean_maxrss_a = np.mean(mem_a[nodes])
            mean_maxrss_b = np.mean(mem_b[nodes])
            std_maxrss_a = np.std(mem_a[nodes])
            std_maxrss_b = np.std(mem_b[nodes])
            print("At  {:4.0f}k ({:3} runs): {:11.0f} +- {:5.0f}    / {:11.0f} +- {:5.0f} {:10.5f}".format(nodes/1e3, len(mem_a[nodes]), mean_maxrss_a, std_maxrss_a, mean_maxrss_b, std_maxrss_b, mean_maxrss_a / mean_maxrss_b))            
        sys.exit(0)

    if 'regression' in args.command:
        print("Run sanity checks.")

        for i in [100,500,1000]:
            name = N[i]
            result = run_test(out_dir, base_args, [name], range(1), threads = (0,))
            result = run_test(out_dir, base_args, [name], range(1), threads = (2,))

            var_args = (('input_filename', (name,)),
                        ('sparse',(1,)))
            result = run_var_test(out_dir, base_args, var_args)

            if i > 100:
                var_args = (('input_filename', (name,)),('sparse',(1,)),('threads',(2,)))
                result = run_var_test(out_dir, base_args, var_args)

            var_args = (('input_filename', (name,)),
                        ('sparse',(0,)),
                        ('threads',(4,)),
                        ('decimation',(2,)))

            result = run_var_test(out_dir, base_args, var_args)

        print_results(result)
        results.update(result)

    # On a new system, it is best to first do a thread swep to find the optimal number of threads on a reasonably-sized graph.
    if 'thread-sweep' in args.command:
        base_args['verbose'] = 0
        base_args['initial_block_reduction_rate'] = 0.50
        base_args['input_filename'] = N[5000]
        var_args = (('threads', (0,2,4,6,8,10,11,12,14,16,18,20,22,24)),)
        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)
        results.update(result)

    if 'single-tiny' in args.command:
        print("Tiny Single process tests.")
        result = run_test(out_dir, base_args, [N[100]], range(3), threads = (4,), max_jobs = 4)
        print_results(result)
        results.update(result)

    if 'single-small' in args.command:
        result = run_test(out_dir, base_args, small_files, iterations, threads = (0,), max_jobs = 6)
        print_results(result)
        results.update(result)

        avg_time = sum([i[1] for i in results.values()]) / len(results)
        print("Mean time is %s" % (avg_time))

    if 'multi-small' in args.command:
        result = run_test(out_dir, base_args, small_files, iterations, threads = (2,4,8,16,27,32))
        print("Multi process tests.")
        print_results(result)
        results.update(result)

    if 'single-sparse' in args.command:
        print("Sparse tests.")
        base_args['sparse'] = 1
        result = run_test(out_dir, base_args, input_files, iterations, threads = (0,), max_jobs = 6)
        print_results(result)
        results.update(result)

    if 'sparse-sweep' in args.command:
        med_files = [N[50000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(1)),
                    ('initial_block_reduction_rate',(0.50,0.75)),
                    ('sparse',(0,1,2)), ('threads',(12,)))

        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)


    if 'big' in args.command:
        med_files = [N[100000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(1)),
                    ('initial_block_reduction_rate',(0.50,0.75,0.95)),
                    ('sparse',(2,)), ('threads',(56,)),
                    ('decimation',(4,))
        )
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)



    if 'reduction-sweep-small' in args.command:
        med_files = [N[20000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(3)),
                    ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95,0.99)),
                    ('sparse',(0,1,2)), ('threads',(8,)))

        result = run_var_test(out_dir, base_args, var_args, max_jobs=2)
        print_results(result)
        results.update(result)

    if 'reduction-sweep' in args.command:
        print("Single var tests.")
        # var_args = { 'initial_block_reduction_rate' : (0.50,0.75,0.90,0.95,0.99), 'threads' : (0, 55) }
        # var_args = (('input_filename', small_files), ('iteration', range(3)))
        var_args = (('input_filename', big_files[3:]), ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95)), ('sparse',(0,)), ('threads',(16,)), ('decimation',(8,)))
        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)
        results.update(result)

        var_args = (('input_filename', big_files[2:]), ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95), ('sparse',(0,)), ('threads',(32,)), ('decimation',(4,))))
        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)
        results.update(result)

    if 'threading' in args.command:
        files = [N[500]]
        var_args = (('input_filename', files),
                    ('iteration', range(500)),
                    ('blocking', (0,1,)),
                    ('finegrain', (0,1)),
                    ('critical', (0,1,2)),
                    ('sparse',(0,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'threading-sanity1' in args.command:
        files = [N[500]]
        var_args = (('input_filename', files),
                    ('iteration', range(1000)),
                    ('blocking', (1,)),
                    ('finegrain', (0,1)),
                    ('critical', (1,)),
                    ('sanity_check', (1,)),
                    ('sparse',(0,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'threading-sanity1.1' in args.command:
        files = [N[5000]]
        var_args = (('input_filename', files),
                    ('iteration', range(200)),
                    ('blocking', (1,)),
                    ('finegrain', (1,)),
                    ('critical', (1,)),
                    ('sanity_check', (1,)),
                    ('sparse',(0,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'threading-sanity2' in args.command:
        files = [N[5000]]
        var_args = (('input_filename', files),
                    ('iteration', range(1000)),
                    ('blocking', (1,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sanity_check', (1,)),
                    ('sparse',(1,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'memory-sanity' in args.command:
        files = [N[500]]
        var_args = (('input_filename', files),
                    ('iteration', range(100)),
                    ('blocking', (0,1,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sanity_check', (0,1,)),
                    ('sparse',(1,)),
                    ('preallocate',(0,1,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'child-alloc' in args.command:
        files = [N[5000]]
        var_args = (('input_filename', files),
                    ('iteration', range(50)),
                    ('blocking', (0,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sanity_check', (1,)),
                    ('sparse',(1,)),
                    ('preallocate',(0,)),
                    ('threads',(12,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'threading-performance' in args.command:
        files = [N[20000]]
        var_args = (('input_filename', files),
                    ('iteration', range(1)),
                    ('blocking', (0,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sparse',(0,1)),
                    ('node_propose_batch_size', (4,8,16,32,64,128,256,512)),
                    ('threads',(12,24)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'group-size' in args.command:
        files = [N[20000]]
        var_args = (('input_filename', files),
                    ('iteration', range(1)),
                    ('blocking', (0,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sparse',(1,)),
                    ('node_propose_batch_size', (0,4,8,16,32,64,128,256,512)),
                    ('threads',(12,24,48,96)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)
        print_results(result)
        results.update(result)

    if 'paces' in args.command:
        files = [N[5000], N[20000], N[50000]] #, N[100000], N[1000000]]
        var_args = (('input_filename', files),
                    ('iteration', range(20)),
                    ('blocking', (0,)),
                    ('finegrain', (0,)),
                    ('critical', (2,)),
                    ('sparse',(1,)),
                    ('node_propose_batch_size', (64,)),
                    ('threads',(24,)))
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1, override_args=override_args)
        print_results(result)
        results.update(result)

    pickle.dump(results, results_f)
