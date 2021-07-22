from partition_baseline_support import *
from updated_partition_main import *
import updated_partition_main
from compute_delta_entropy import *
import numpy as np
import csv
# import pandas as pd


def parallel_execution_time(file, t_merge, t_move, sparse, block_reduction_rate):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, required=False, default=0,
                        help="Configure number of threads for both agglomerative merge and nodal movements.")
    parser.add_argument("--t-merge", type=int, required=False, default=t_merge,
                        help="Configure number of threads for both agglomerative merge phase (overrides --threads).")
    parser.add_argument("--t-move", type=int, required=False, default=t_move,
                        help="Configure number of threads for nodal movement phase (overrides --threads)")

    parser.add_argument("-p", "--parts", type=int, required=False, default=0)
    parser.add_argument("-d", "--decimation", type=int, required=False, default=0)
    parser.add_argument("-v", "--verbose", type=int, required=False, default=2, help="Verbosity level.")
    parser.add_argument("-b", "--node-move-update-batch-size", type=int, required=False, default=1)
    parser.add_argument("-g", "--node-propose-batch-size", type=int, required=False, default=4)
    parser.add_argument("--sparse", type=int, required=False, default=sparse)
    parser.add_argument("--sparse-algorithm", type=int, required=False, default=0)
    parser.add_argument("--sparse-data", type=int, required=False, default=0)
    parser.add_argument("-s", "--sort", type=int, required=False, default=0)
    parser.add_argument("-S", "--seed", type=int, required=False, default=-1)
    parser.add_argument("-m", "--merge-method", type=int, required=False, default=0)
    parser.add_argument("--mpi", action="store_true", default=False)
    parser.add_argument("input_filename", nargs="?", type=str, default=file)

    # Debugging options
    parser.add_argument("--min-number-blocks", type=int, required=False, default=0,
                        help="Force stop at this many blocks instead of searching for optimality.")
    parser.add_argument("--initial-block-reduction-rate", type=float, required=False, default=block_reduction_rate)
    parser.add_argument("--profile", type=int, required=False, help="Profiling level 0=disabled, 1=main, 2=workers.",
                        default=0)
    parser.add_argument("--test-decimation", type=int, required=False, default=0)
    parser.add_argument("--predecimation", type=int, required=False, default=0)
    parser.add_argument("--debug", type=int, required=False, default=0)
    parser.add_argument("--test-resume", type=int, required=False, default=0)
    parser.add_argument("--naive-streaming", type=int, required=False, default=0)
    parser.add_argument("--min-nodal-moves-ratio", type=float, required=False, default=0.0,
                        help="Break nodal move loop early if the number of accepted moves is below this fraction of the number of nodes.")
    parser.add_argument("--skip-eval", type=int, required=False, default=0, help="Skip partition evaluation.")
    parser.add_argument("--max-num-nodal-itr", type=int, required=False, default=100,
                        help="Maximum number of iterations during nodal moves.")
    parser.add_argument("--compressed-nodal-moves", type=int, required=False, default=False,
                        help="Whether to use compressed representation during nodal movements -- usually uncompressed is faster.")

    args = parser.parse_args()

    updated_partition_main.merge_time = 0
    updated_partition_main.nodal_move_time = 0
    updated_partition_main.partition_time = 0
    updated_partition_main.node_move_update_batch_size = 0
    updated_partition_main.block_size = [-1] * 100
    updated_partition_main.merging_times = [-1] * 100
    updated_partition_main.nodal_move_times = [-1] * 100
    updated_partition_main.entropy_index = 0
    updated_partition_main.precision_score = 0
    updated_partition_main.recall_score = 0
    # if args.compute_delta_entropy == 0:
    #     print("ORIGINAL")
    # elif args.compute_delta_entropy == 0:
    #     print("SPARSE")
    # else:
    #     print("ALT")

    if args.debug:
        sys.excepthook = info

    if args.profile:
        import cProfile
        cProfile.run('do_main(args)', filename=args.profile)
    else:
        do_main(args)
        return updated_partition_main.merge_time, updated_partition_main.nodal_move_time, \
               updated_partition_main.partition_time, \
               updated_partition_main.block_size[:updated_partition_main.entropy_index], \
               updated_partition_main.merging_times[:updated_partition_main.entropy_index], \
               updated_partition_main.nodal_move_times[:updated_partition_main.entropy_index],\
               updated_partition_main.precision_score,\
               updated_partition_main.recall_score


def thread_variations(file, t_merge, t_move, sparse, reduction_rate):
    merge_time, move_time, partition_time, block_sizes, merging_times, nodal_move_times, precision, recall \
        = parallel_execution_time(file, t_merge, t_move, sparse, reduction_rate)

    move_time -= merge_time

    return merge_time, move_time, partition_time, block_sizes, merging_times, nodal_move_times, precision, recall


input_files = ["../../data/static/simulated_blockmodel_graph_500_nodes",
               "../../data/static/simulated_blockmodel_graph_1000_nodes",
               "../../data/static/simulated_blockmodel_graph_5000_nodes",
               "../../data/static/simulated_blockmodel_graph_20000_nodes",
               "../../data/static/simulated_blockmodel_graph_50000_nodes",
               "../../data/static/simulated_blockmodel_graph_100000_nodes"]


# Threads Testing #
fields = ['Reduction-rate', 't-merge', 't-move', 'Merge time', 'Move time', 'Partition time', 'Precision', 'Recall'] # field names

# Table 4 - Accuracy
# 500, 1000, 5K, 20K, 50K, 100K
# Reduction rates: 0.5, 0.75, 0.9
# 5 tests each
precision_list = []
recall_list = []

# 500 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('500_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)


precision_list = []
recall_list = []

# 1000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('1000_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)


precision_list = []
recall_list = []

# 5000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('5000_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)



precision_list = []
recall_list = []

# 20000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('20000_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)



precision_list = []
recall_list = []

# 50000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('50000_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)


# 100000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.75)

precision_list.append(precision)
recall_list.append(recall)

with open('100000_reduction075_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

precision_list = []
recall_list = []

# 500 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[0], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('500_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

precision_list = []
recall_list = []

# 1000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[1], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('1000_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

precision_list = []
recall_list = []

# 5000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[2], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('5000_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

precision_list = []
recall_list = []

# 20000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[3], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('20000_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

precision_list = []
recall_list = []

# 50000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[4], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('50000_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)

# 100000 nodes
merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
    input_files[5], 12, 12, 2, 0.9)

precision_list.append(precision)
recall_list.append(recall)

with open('100000_reduction09_accuracy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(precision_list)
    write.writerow(recall_list)
# Initial Block Reduction Rates Table
# 1) 500 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 1, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 1, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('500_nodes_sparse_1_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 0, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 0, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('500_nodes_sparse_0_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
#
# # 2) 1000 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 5, 0, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 5, 0, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('1000_nodes_sparse_0_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
# # 3) 5000 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 5, 0, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 5, 0, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('5000_nodes_sparse_0_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
# # 4) 20000 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 5, 0, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 5, 0, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('20000_nodes_sparse_0_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
# # 5) 50000 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 5, 0, 0.75)
# rows.append([0.75, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 5, 0, 0.9)
# rows.append([0.9, 44, 5, merge, move, partition, precision, recall])
#
# with open('50000_nodes_sparse_0_reduction.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)

# Compression Table

# 1) 500 blocks
# rows = [] # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 5, 0, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 5, 1, 0.5)
# rows.append([0.5, 44, 5, merge, move, partition, precision, recall])
#
# with open('compression_comparison.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)


# Base cases

#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[0], 12, 12, 0, 0.5)
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[1], 12, 12, 0, 0.5)
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[2], 12, 12, 0, 0.5)
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[3], 12, 12, 0, 0.5)
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#

# # Data Compression Comparison
# rows = []  # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[0], 12, 12, 1, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('500_compression.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("500_compression_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[1], 12, 12, 1, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('1000_compression.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("1000_compression_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[2], 12, 12, 1, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('5000_compression.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("5000_compression_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[3], 12, 12, 1, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('20000_compression.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("20000_compression_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[4], 12, 12, 1, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('50000_compression.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("50000_compression_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []


# Parallelism Control Comparison
# rows = []  # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[0], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('500_parallelism.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("500_parallelism_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[1], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('1000_parallelism.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("1000_parallelism_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[2], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('5000_parallelism.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("5000_parallelism_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[3], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('20000_parallelism.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("20000_parallelism_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[4], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('50000_parallelism.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("50000_parallelism_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []

# Reduction rate comparison
# rows = []  # data rows
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[5], 12, 12, 0, 0.5)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('100000_base.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("100000_base_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[1], 12, 12, 0, 0.75)
#
# rows.append([0.75, 12, 12, merge, move, partition, precision, recall])
#
# with open('1000_reduction_rate.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# with open("1000_reduction_rate_breakdown.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows = []

# 2) merge threads = 12
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[4], 12, 12, 0, 0.5)
#
# with open("50k_merging_breakdown_12.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('50k_merging_12.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# # 3) merge threads = 24
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[4], 24, 12, 0, 0.5)
#
# with open("50k_merging_breakdown_24.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows.append([0.5, 24, 12, merge, move, partition, precision, recall])
#
# with open('50k_merging_24.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#
# # 4) merge threads = 44
# rows = []
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(
#     input_files[4], 44, 12, 0, 0.5)
#
# with open("50k_merging_breakdown_44.csv", 'w') as f:
#     wr = csv.writer(f)
#     wr.writerow(block_sizes)
#     wr.writerow(merging_times)
#     wr.writerow(nodal_move_times)
#
# rows.append([0.5, 44, 12, merge, move, partition, precision, recall])
#
# with open('50k_merging_44.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(fields)
#     write.writerows(rows)
#


# rows.append([0.5, 12, 12, merge, move, partition, precision, recall])
#
# with open('base_case.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)

# Combined Testing
# rows = [] # data rows

# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[0], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[1], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[2], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[3], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])
#
# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[4], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])

# merge, move, partition, block_sizes, merging_times, nodal_move_times, precision, recall = thread_variations(input_files[5], 44, 2, 2, 0.75)
# rows.append([0.75, 44, 2, merge, move, partition, precision, recall])
#
#
# with open('combined_testing_100K.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)


# # 4) Optimal threads
# fields = ['Graph size', 't-merge', 't-move', 'Merge time', 'Move time', 'Partition time'] # field names
# rows = [] # data rows
#
# merge, move, partition = thread_variations(input_files[0], 36, 7, 0)
# rows.append([5000, 36, 7, merge, move, partition])
#
# merge, move, partition = thread_variations(input_files[1], 44, 3, 0)
# rows.append([20000, 44, 3, merge, move, partition])
#
# merge, move, partition = thread_variations(input_files[2], 44, 3, 0)
# rows.append([50000, 44, 3, merge, move, partition])
#
# with open('optimal_sparse_0.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
#
# rows = []
#
# merge, move, partition = thread_variations(input_files[0], 20, 5, 1)
# rows.append([5000, 20, 5, merge, move, partition])
#
# merge, move, partition = thread_variations(input_files[1], 16, 3, 1)
# rows.append([20000, 16, 3, merge, move, partition])
#
# merge, move, partition = thread_variations(input_files[2], 48, 3, 1)
# rows.append([50000, 48, 3, merge, move, partition])
#
# with open('optimal_sparse_1_50000.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(fields)
#     write.writerows(rows)
