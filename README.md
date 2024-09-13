
# GraphChallenge
Code from our GraphChallenge papers:

Uppal, Ahsen J., Thomas B. Rolinger, and H. Howie Huang. "Decontentioned Stochastic Block Partition." 2023 IEEE High Performance Extreme Computing Conference (HPEC), IEEE, 2023.

Uppal, Ahsen J., Jaeseok Choi, Thomas B. Rolinger, and H. Howie Huang. "Faster stochastic block partition using aggressive initial merging, compressed representation, and parallelism control." 2021 IEEE High Performance Extreme Computing Conference (HPEC), IEEE, 2021.

Uppal, Ahsen J., Guy Swope, and H. Howie Huang. "Scalable stochastic block partition." 2017 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2017.

Uppal, Ahsen J., and H. Howie Huang. "Fast stochastic block partition for streaming graphs." 2018 IEEE High Performance extreme Computing Conference (HPEC). IEEE, 2018.

Graph Challenge 
http://graphchallenge.org

# Bulding:
    cd GraphChallenge/StochasticBlockPartition/code/python
	CC="clang" python setup.py build

# Running:
    # Partition a 5k node input using 8 threads.
	export PYTHONPATH=$PWD/build/lib.linux-x86_64-cpython-310
	python partition_baseline_main.py ../../data/static/simulated_blockmodel_graph_5000_nodes -t 8

# Write parition to output file:
	export PYTHONPATH=$PWD/build/lib.linux-x86_64-cpython-310
	python partition_baseline_main.py ../../data/static/simulated_blockmodel_graph_5000_nodes -t 8 -o partition.txt

# Additional Options:
	python partition_baseline_main.py --help
