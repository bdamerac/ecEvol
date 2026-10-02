"""Run batches of forward simulations in parallel and write each batch to HDF5."""
import argparse
import multiprocessing as mp
import os
import random
import time

import h5py
import numpy as np

from sim_opt import ecdna_simulation

OUT_DIR = os.environ.get("ECDNA_SIM_OUT", "simulations")

def run_simulation(args):
    intro_generation, ecdna_s, ecdna_mu, mu, next_mutation, final_cell_count = args
    return ecdna_simulation(intro_generation, ecdna_s, ecdna_mu=ecdna_mu, mu=mu, next_mutation=next_mutation, final_cell_count=final_cell_count)

def save_to_h5(filename, results):
    """One group per simulation: parameters as attributes, one spectrum pair per depth."""
    depths = [30, 50, 100, 200]

    with h5py.File(filename, "w") as h5f:
        for i, (result_30, result_50, result_100, result_200) in enumerate(results):
            grp = h5f.create_group(f"sim_{i}")
            metadata_result = result_30   # parameters are the same at every depth
            for key in ['ecdna_region_cn', 'tot_descended', 'tot_ecdna', 'tot_reg', 
                        'intro_generation', 'ecdna_s', 'final_cell_count', 'mu', 'ecdna_mu', 'clonal_mutations']:
                grp.attrs[key] = metadata_result[key]
            for depth, result in zip(depths, [result_30, result_50, result_100, result_200]):
                grp.create_dataset(f"r1_spectrum_{depth}", data=np.array(result['r1_spectrum'], dtype=np.float32))
                grp.create_dataset(f"r2_spectrum_{depth}", data=np.array(result['r2_spectrum'], dtype=np.float32))

def parallel_simulations(num_simulations, batch_id, id, num_workers=None):
    """Sample parameters for one batch, simulate in parallel, save to HDF5."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)

    param_list = [
        (
            random.randint(0, 9),      # emergence time g
            random.uniform(0, 3),      # selection coefficient s
            random.uniform(0.01, 3),   # ecDNA mutation rate
            random.randint(1, 500),    # chromosomal mutation rate
            random.randint(1, 5000),   # clonal mutations
            20000                      # final population size
        ) for _ in range(num_simulations)
    ]

    print(f"[id {id}] Starting batch {batch_id}: {num_simulations} simulations using {num_workers} parallel processes...")

    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_simulation, param_list)

    end_time = time.time()
    print(f"[id {id}] Completed batch {batch_id} in {(end_time - start_time)/60:.2f} minutes.")

    os.makedirs(OUT_DIR, exist_ok=True)
    output_filename = os.path.join(OUT_DIR, f"id_{id}_batch_{batch_id}.h5")
    save_to_h5(output_filename, results)
    print(f"[id {id}] Batch {batch_id} results saved to {output_filename}")

    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel tumor evolution simulations.")
    parser.add_argument("--id", type=int, required=True, help="job id, used in the output filenames")
    args = parser.parse_args()

    num_batches = 1000
    num_simulations_per_batch = 100

    for batch_id in range(1, num_batches + 1):
        parallel_simulations(num_simulations_per_batch, batch_id, args.id)
    print(f"id {args.id}: completed {num_batches} batches")
