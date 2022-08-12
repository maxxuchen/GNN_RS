import os
import gzip
import queue
import pickle
import argparse
import threading
import numpy as np
import networkx as nx

from bc.bc_config import NUM_NODES, RADIUS
from common.dataset_utility import cal_weight


def make_samples(out_dir, out_queue, stop_flag):
    sample_counter = 0
    while not stop_flag.is_set():
        G = nx.random_geometric_graph(np.random.choice(NUM_NODES), RADIUS)
        nx.set_edge_attributes(G, {e: {'weight': cal_weight(G, e)} for e in G.edges})  # assign weight
        MWM = nx.algorithms.matching.max_weight_matching(G)
        filename = f'{out_dir}/sample_{threading.current_thread().name}_{sample_counter}.pkl'

        if not stop_flag.is_set():
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'graph': G,
                    'MWM': MWM,
                }, f)
                sample_counter += 1

            out_queue.put({
                'type': 'sample',
                'filename': filename,
            })
            # print(f"[w {threading.current_thread().name}] generate one graph.\n", end='')


def collect_samples(n_samples, out_dir, n_jobs):
    # create output directory, throws an error if it already exists
    os.makedirs(out_dir)

    # start workers
    answers_queue = queue.SimpleQueue()
    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
            target=make_samples,
            args=(out_dir, answers_queue, workers_stop_flag),
            daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    i = 0
    while i < n_samples:
        sample = answers_queue.get()
        os.rename(sample['filename'], f'{out_dir}/sample_{i + 1}.pkl')
        i += 1

        if i % 100 == 0:
            print(f'generate {i} samples now, {n_samples-i} left.')

        # stop workers
        if answers_queue.qsize() + i >= n_samples:
            workers_stop_flag.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=6,
    )
    args = parser.parse_args()

    # hyper parameters
    num_train = 100000
    num_valid = 20000

    # set path
    DIR = os.path.dirname(__file__)
    train_path = os.path.join(DIR, 'samples/train')
    valid_path = os.path.join(DIR, 'samples/valid')

    collect_samples(num_train, train_path, args.njobs)
    collect_samples(num_valid, valid_path, args.njobs)
