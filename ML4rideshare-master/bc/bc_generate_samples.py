import os
import gzip
import queue
import pickle
import argparse
import threading
import numpy as np

from bc.bc_config import NUM_VEHICLE
from env.simulator import Simulator as Environment
from es.utility.matcher import RR_Matcher, VR_Matcher


def make_samples(data_path, seed, out_dir, out_queue, stop_flag):
    sample_counter = 0
    while not stop_flag.is_set():
        done = False
        rr_matcher = RR_Matcher('heu')
        vr_matcher = VR_Matcher('heu')
        env = Environment(n_vehicles=NUM_VEHICLE, data_path=data_path, seed=seed)
        env.reset()
        while not done:
            dispatch_action = {}
            rr_graph = env.get_rr_match_graph()
            if rr_graph is not None:
                rr_MWM, rr_decision = rr_matcher.get_rr_match_decision(rr_graph)
                env.do_rr_match(rr_decision)
                vr_graph = env.get_vr_match_graph()
                if vr_graph is not None:
                    vr_MWM, vr_decision = vr_matcher.get_vr_match_decision(vr_graph)
                    dispatch_action = env.do_vr_match(vr_decision)

                    if len(rr_graph.edges) > 0 and len(vr_graph.edges) > 0 and not stop_flag.is_set():
                        rr_match_filename = f'{out_dir}/rr_match/sample_{threading.current_thread().name}_{sample_counter}.pkl'
                        vr_match_filename = f'{out_dir}/vr_match/sample_{threading.current_thread().name}_{sample_counter}.pkl'
                        with gzip.open(rr_match_filename, 'wb') as f:
                            pickle.dump({
                                'graph': rr_graph,
                                'MWM': rr_MWM,
                            }, f)
                        with gzip.open(vr_match_filename, 'wb') as f:
                            pickle.dump({
                                'graph': vr_graph,
                                'MWM': vr_MWM,
                            }, f)
                        out_queue.put({
                            'rr_match_filename': rr_match_filename,
                            'vr_match_filename': vr_match_filename,
                        })
                        sample_counter += 1

            if not stop_flag.is_set():
                done, _, _, _, _, _ = env.step(dispatch_action)


def collect_samples(data_path, rng, n_samples, out_dir, n_jobs):
    # create output directory, throws an error if it already exists
    rr_out_dir = os.path.join(out_dir, 'rr_match')
    vr_out_dir = os.path.join(out_dir, 'vr_match')
    os.makedirs(rr_out_dir, exist_ok=True)
    os.makedirs(vr_out_dir, exist_ok=True)

    # start workers
    answers_queue = queue.SimpleQueue()
    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
            target=make_samples,
            args=(data_path, rng.randint(2 ** 32), out_dir, answers_queue, workers_stop_flag),
            daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    i = 0
    while i < n_samples:
        sample = answers_queue.get()
        os.rename(sample['rr_match_filename'], f'{out_dir}/rr_match/sample_{i + 1}.pkl')
        os.rename(sample['vr_match_filename'], f'{out_dir}/vr_match/sample_{i + 1}.pkl')
        i += 1

        if i % 100 == 0:
            print(f'generate {i} samples now, {n_samples - i} left.')

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
        default=1,
    )
    args = parser.parse_args()

    # hyper parameters
    num_train = 100000
    num_valid = 20000

    # set path
    DIR = os.path.dirname(os.path.dirname(__file__))
    train_data_path = os.path.join(DIR, 'data/train.csv')
    valid_data_path = os.path.join(DIR, 'data/test.csv')
    train_out_dir = os.path.join(DIR, 'bc/samples/train')
    valid_out_dir = os.path.join(DIR, 'bc/samples/valid')

    rng = np.random.RandomState(args.seed + 100)
    collect_samples(train_data_path, rng, num_train, train_out_dir, args.njobs)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(valid_data_path, rng, num_valid, valid_out_dir, args.njobs)
