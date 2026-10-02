import argparse
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEPTHS = (30, 50, 100, 200)
DEFAULT_CHECKPOINTS = (10000, 20000, 30000, 40000, 50000)
MIN_TOT_DESCENDED = 4000


@dataclass(frozen=True)
class SimulationParams:
    intro_generation: int
    ecdna_s: float
    ecdna_mu: float
    mu: int
    next_mutation: int
    max_population_size: int
    simulation_id: int
    seed: int


def sample_params(simulation_id: int, rng: random.Random, max_population_size: int) -> SimulationParams:
    return SimulationParams(
        intro_generation=rng.randint(0, 9),
        ecdna_s=rng.uniform(0.0, 2.5),
        ecdna_mu=rng.uniform(0.01, 3.0),
        mu=rng.randint(1, 500),
        next_mutation=rng.randint(1, 5000),
        max_population_size=max_population_size,
        simulation_id=simulation_id,
        seed=rng.randint(0, 2**31 - 1),
    )


def simulate_sequencing(allele_fractions: np.ndarray, depth: int, total_dna_segments: int):
    if len(allele_fractions) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    obs_depths = np.random.binomial(
        n=total_dna_segments,
        p=depth / total_dna_segments,
        size=len(allele_fractions),
    )
    mutation_reads = np.random.binomial(n=obs_depths, p=allele_fractions)
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_fracs = np.divide(
            mutation_reads,
            obs_depths,
            out=np.zeros_like(mutation_reads, dtype=float),
            where=obs_depths > 0,
        )

    valid_mask = (mutation_reads > 5) & (obs_fracs > 0.01)
    return obs_fracs[valid_mask], allele_fractions[valid_mask]


def extract_allele_freq_spectra(adj_r1_fracs: np.ndarray, adj_r2_fracs: np.ndarray):
    r1 = np.histogram(adj_r1_fracs, bins=100, range=(0.0, 1.0))[0]
    r2 = np.histogram(adj_r2_fracs, bins=100, range=(0.0, 1.0))[0]
    return r1.astype(np.float32), r2.astype(np.float32)


def mutate_region(mutations: list[int], mutation_rate: float, next_mutation: int) -> int:
    num_mutations = np.random.poisson(mutation_rate)
    mutations.extend(range(next_mutation, next_mutation + num_mutations))
    return next_mutation + num_mutations


def divide_cell(cell: dict, mu: int, ecdna_mu: float, next_mutation: int):
    daughter_cells = [
        {
            "region1": cell["region1"][:],
            "region2": cell["region2"][:],
            "ecDNAs": [],
            "descendent": cell["descendent"],
        }
        for _ in range(2)
    ]

    for daughter in daughter_cells:
        next_mutation = mutate_region(daughter["region1"], 2 * mu, next_mutation)
        next_mutation = mutate_region(daughter["region2"], ecdna_mu, next_mutation)

    ec_dna_pool = []
    for ec_dna in cell["ecDNAs"]:
        new_mutations = ec_dna["mutations"][:]
        next_mutation = mutate_region(new_mutations, ecdna_mu, next_mutation)
        ec_dna_pool.extend([{"mutations": new_mutations} for _ in range(2)])

    np.random.shuffle(ec_dna_pool)
    for ec_dna in ec_dna_pool:
        np.random.choice(daughter_cells)["ecDNAs"].append(ec_dna)

    return daughter_cells, next_mutation


def gather_checkpoint_results(
    cells: list[dict],
    params: SimulationParams,
    current_population_size: int,
):
    tot_ecdna = sum(len(cell["ecDNAs"]) > 0 for cell in cells)
    tot_reg = len(cells) - tot_ecdna
    tot_descended = sum(cell["descendent"] for cell in cells)

    r1_total = 2 * len(cells)
    r2_total = 2 * len(cells) + sum(len(cell["ecDNAs"]) for cell in cells)
    ecdna_region_cn = r2_total / len(cells)

    r1_freqs = defaultdict(int)
    r2_freqs = defaultdict(int)

    for cell in cells:
        for mutation in cell["region1"]:
            r1_freqs[mutation] += 1
        for mutation in cell["region2"]:
            r2_freqs[mutation] += 1
        for ec_dna in cell["ecDNAs"]:
            for mutation in ec_dna["mutations"]:
                r2_freqs[mutation] += 1

    r1_fracs = np.array(list(r1_freqs.values()), dtype=float) / r1_total if r1_freqs else np.array([], dtype=float)
    r2_fracs = np.array(list(r2_freqs.values()), dtype=float) / r2_total if r2_freqs else np.array([], dtype=float)

    depth_results = {}
    for depth in DEPTHS:
        adj_r1_fracs, _ = simulate_sequencing(r1_fracs, depth, r1_total)
        adj_r2_fracs, _ = simulate_sequencing(r2_fracs, depth, r2_total)
        r1_spectrum, r2_spectrum = extract_allele_freq_spectra(adj_r1_fracs, adj_r2_fracs)
        depth_results[depth] = {
            "r1_spectrum": r1_spectrum,
            "r2_spectrum": r2_spectrum,
        }

    metadata = {
        "ecdna_region_cn": float(ecdna_region_cn),
        "tot_descended": int(tot_descended),
        "tot_ecdna": int(tot_ecdna),
        "tot_reg": int(tot_reg),
        "intro_generation": int(params.intro_generation),
        "ecdna_s": float(params.ecdna_s),
        "final_cell_count": int(current_population_size),
        "population_size": int(current_population_size),
        "mu": int(params.mu),
        "ecdna_mu": float(params.ecdna_mu),
        "clonal_mutations": int(params.next_mutation),
        "simulation_id": int(params.simulation_id),
        "seed": int(params.seed),
    }

    return depth_results, metadata


def run_checkpointed_simulation(params: SimulationParams, checkpoints: Iterable[int]):
    checkpoints = sorted(set(int(x) for x in checkpoints))
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    if checkpoints[-1] > params.max_population_size:
        raise ValueError("Largest checkpoint cannot exceed max_population_size.")

    random.seed(params.seed)
    np.random.seed(params.seed)

    random_fraction = np.random.uniform(0.9, 0.999)
    next_mutation = params.next_mutation
    cells = [{
        "region1": list(range(int(next_mutation * random_fraction))),
        "region2": list(range(int(next_mutation * random_fraction), next_mutation)),
        "ecDNAs": [],
        "descendent": False,
    }]

    ec_dna_introduced = False
    introduction_step = 2 ** int(params.intro_generation)
    bmax = 1.0
    checkpoint_results = {}
    checkpoint_set = set(checkpoints)

    while len(cells) < params.max_population_size:
        dividing_cell_index = np.random.randint(len(cells))
        dividing_cell = cells[dividing_cell_index]

        if not ec_dna_introduced and len(cells) == introduction_step:
            intro_cell = cells[np.random.randint(len(cells))]
            r2_muts = intro_cell["region2"][:]
            np.random.shuffle(r2_muts)
            half = len(r2_muts) // 2
            intro_cell["ecDNAs"].append({"mutations": r2_muts[:half][:]})
            intro_cell["region2"] = r2_muts[half:][:]
            intro_cell["descendent"] = True
            ec_dna_introduced = True
            bmax += params.ecdna_s

        r = np.random.uniform(0.0, bmax)
        cur_b = 1.0 if len(dividing_cell["ecDNAs"]) == 0 else bmax
        if r <= cur_b:
            daughter_cells, next_mutation = divide_cell(dividing_cell, params.mu, params.ecdna_mu, next_mutation)
            cells.extend(daughter_cells)
            cells.pop(dividing_cell_index)

            current_population_size = len(cells)
            if current_population_size in checkpoint_set and current_population_size not in checkpoint_results:
                depth_results, metadata = gather_checkpoint_results(
                    cells=cells,
                    params=params,
                    current_population_size=current_population_size,
                )
                checkpoint_results[current_population_size] = (depth_results, metadata)

                if current_population_size == checkpoints[-1]:
                    break

    return checkpoint_results


def simulation_passes_strict_filter(checkpoint_results: dict[int, tuple[dict, dict]], checkpoints: Iterable[int]) -> bool:
    expected = set(int(x) for x in checkpoints)
    if set(checkpoint_results.keys()) != expected:
        return False

    for _, metadata in checkpoint_results.values():
        if int(metadata["tot_descended"]) < MIN_TOT_DESCENDED:
            return False

    return True


def save_results_to_h5(output_path: Path, results: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        for record in results:
            sim_id = int(record["params"]["simulation_id"])
            params_group = h5f.create_group(f"simulation_{sim_id}")
            for key, value in record["params"].items():
                params_group.attrs[key] = value

            for population_size, checkpoint in sorted(record["checkpoints"].items()):
                depth_results = checkpoint["depth_results"]
                metadata = checkpoint["metadata"]

                grp = params_group.create_group(f"pop_{population_size}")
                for key, value in metadata.items():
                    grp.attrs[key] = value

                for depth in DEPTHS:
                    grp.create_dataset(
                        f"r1_spectrum_{depth}",
                        data=np.asarray(depth_results[depth]["r1_spectrum"], dtype=np.float32),
                    )
                    grp.create_dataset(
                        f"r2_spectrum_{depth}",
                        data=np.asarray(depth_results[depth]["r2_spectrum"], dtype=np.float32),
                    )


def parse_checkpoints(raw: str):
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def run_one_simulation(task: tuple[SimulationParams, tuple[int, ...]]):
    params, checkpoints = task
    checkpoint_results = run_checkpointed_simulation(params, checkpoints)
    if not simulation_passes_strict_filter(checkpoint_results, checkpoints):
        return None

    return {
        "params": asdict(params),
        "checkpoints": {
            pop_size: {
                "depth_results": depth_results,
                "metadata": metadata,
            }
            for pop_size, (depth_results, metadata) in checkpoint_results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate checkpointed ecDNA simulations.")
    parser.add_argument("--num-simulations", type=int, default=10)
    parser.add_argument("--checkpoints", type=str, default="10000,20000,30000,40000,50000")
    parser.add_argument("--max-population-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, mp.cpu_count())))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpointed_simulations.h5"),
    )
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    if any(x < 10000 or x > 50000 for x in checkpoints):
        raise ValueError("This experiment expects checkpoints between 10k and 50k.")

    rng = random.Random(args.seed)
    tasks = []
    for simulation_id in range(args.num_simulations):
        params = sample_params(
            simulation_id=simulation_id,
            rng=rng,
            max_population_size=args.max_population_size,
        )
        tasks.append((params, checkpoints))

    if args.num_workers == 1:
        records = [run_one_simulation(task) for task in tasks]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            records = pool.map(run_one_simulation, tasks)

    all_results = [record for record in records if record is not None]

    save_results_to_h5(args.output, all_results)
    print(
        f"Saved {len(all_results)} strict-valid checkpointed simulations "
        f"(requested {args.num_simulations}) to {args.output}"
    )


if __name__ == "__main__":
    main()
