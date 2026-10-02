"""Forward simulation of tumour growth with ecDNA: cell division, mutation, random
ecDNA segregation, and simulated sequencing of the final population."""
from collections import defaultdict

import numpy as np


def simulate_sequencing(allele_fractions, depth, total_DNA_segments):
    """Binomial sampling of read depth and variant reads; returns observed and true fractions."""
    obs_depths = np.random.binomial(n=total_DNA_segments, p=depth / total_DNA_segments, size=len(allele_fractions))
    mutation_reads = np.random.binomial(n=obs_depths, p=allele_fractions)
    obs_fracs = mutation_reads / obs_depths

    valid_mask = (mutation_reads > 5) & (obs_fracs > 0.01)
    return obs_fracs[valid_mask], allele_fractions[valid_mask]

def extract_allele_freq_spectra(adj_r1_fracs, adj_r2_fracs):
    """100-bin allele frequency spectra for the two regions."""
    return np.histogram(adj_r1_fracs, bins=100, range=(0, 1))[0], np.histogram(adj_r2_fracs, bins=100, range=(0, 1))[0]

def gather_results(cells, introduction_generation, ecdna_s, final_cell_count, mu, ecdna_mu, clonal_mutations):
    """Allele fractions in the final population, and the spectra at each sequencing depth."""
    tot_ecdna = sum(len(c['ecDNAs']) > 0 for c in cells)
    tot_reg = len(cells) - tot_ecdna
    tot_descended = sum(c['descendent'] for c in cells)

    r1_total, r2_total = 2 * len(cells), 2 * len(cells) + sum(len(c['ecDNAs']) for c in cells)
    ecdna_region_cn = r2_total / len(cells)

    r1_freqs, r2_freqs = defaultdict(int), defaultdict(int)
    for c in cells:
        for m in c['region1']:
            r1_freqs[m] += 1
        for m in c['region2']:
            r2_freqs[m] += 1
        for e in c['ecDNAs']:
            for m in e['mutations']:
                r2_freqs[m] += 1

    r1_fracs = np.array(list(r1_freqs.values())) / r1_total
    r2_fracs = np.array(list(r2_freqs.values())) / r2_total

    results = []
    for depth in [30, 50, 100, 200]:
        adj_r1_fracs, true_r1_fracs = simulate_sequencing(r1_fracs, depth, r1_total)
        adj_r2_fracs, true_r2_fracs = simulate_sequencing(r2_fracs, depth, r2_total)
        r1_spectrum, r2_spectrum = extract_allele_freq_spectra(adj_r1_fracs, adj_r2_fracs)

        results.append({
            'r1_spectrum': r1_spectrum, 'r2_spectrum': r2_spectrum,
            'ecdna_region_cn': ecdna_region_cn, 'tot_descended': tot_descended,
            'tot_ecdna': tot_ecdna, 'tot_reg': tot_reg,
            'intro_generation': introduction_generation, 'ecdna_s': ecdna_s,
            'final_cell_count': final_cell_count, 'mu': mu, 'ecdna_mu': ecdna_mu,
            'clonal_mutations': clonal_mutations
        })
    return results

def ecdna_simulation(introduction_generation, ecdna_s, mu=100, ecdna_mu=2, next_mutation=0, final_cell_count=10000):
    """Grow one tumour to final_cell_count cells; ecDNA arises when the population reaches 2**g."""
    clonal_mutations = next_mutation
    introduction_step = 2 ** int(introduction_generation)  
    random_fraction = np.random.uniform(0.9, 0.999)

    cells = [{
        'region1': list(range(int(next_mutation * random_fraction))),
        'region2': list(range(int(next_mutation * random_fraction), next_mutation)),
        'ecDNAs': [],
        'descendent': False
    }]
    
    ecDNA_introduced = False
    bmax = 1
    skip_mutations = False
    total_descended = 0
    num_ecDNA_pos = 0
    while len(cells) < final_cell_count:

        dividing_cell_index = np.random.randint(len(cells))
        dividing_cell = cells[dividing_cell_index]
        # Introduce ecDNA at the predetermined step
        if not ecDNA_introduced and len(cells) == introduction_step:
            intro_cell_index = np.random.randint(len(cells))
            intro_cell = cells[intro_cell_index]
            r2_muts = intro_cell['region2'][:]
            np.random.shuffle(r2_muts)
            intro_cell['ecDNAs'].append({'mutations': r2_muts[:len(r2_muts) // 2][:]}) # one homolog (half the mutations) becomes ecDNA
            intro_cell['region2'] = r2_muts[len(r2_muts) // 2 :][:] # only mutations from the undamaged homolog remain. 
            intro_cell['descendent'] = True
            total_descended += 1
            num_ecDNA_pos += 1
            ecDNA_introduced = True
            bmax += ecdna_s


        r = np.random.uniform(0,bmax)
        cur_b = 1 if len(dividing_cell['ecDNAs']) == 0 else bmax
        if r <= cur_b:
            daughter_cells, next_mutation = divide_cell(dividing_cell, mu, ecdna_mu, next_mutation, skip_mutations)
            if dividing_cell['descendent']:
                total_descended += 1
            cells.extend(daughter_cells)
            for d in daughter_cells:
                if len(d['ecDNAs']) > 0:
                    num_ecDNA_pos += 1
                num_ecDNA_pos -= 1
            cells.pop(dividing_cell_index)
    results = gather_results(cells, introduction_generation, ecdna_s, final_cell_count, mu, ecdna_mu, clonal_mutations)
    return results

def divide_cell(cell, mu, ecdna_mu, next_mutation, skip_mutations):
    """Two daughters with new mutations; replicated ecDNA copies are segregated at random."""
    daughter_cells = [
        {'region1': cell['region1'] if skip_mutations else cell['region1'][:], 
         'region2': cell['region2'] if skip_mutations else cell['region2'][:], 
         'ecDNAs': [],
         'descendent': cell['descendent']}
        for _ in range(2)
    ]

    if not skip_mutations:
        for d in daughter_cells:
            next_mutation = mutate_region(d['region1'], 2 * mu, next_mutation)
            next_mutation = mutate_region(d['region2'], ecdna_mu, next_mutation)

    ecDNA_pool = []
    if skip_mutations:
        ecDNA_pool = [{'mutations': e['mutations']} for e in cell['ecDNAs'] for _ in range(2)]
    else:
        for e in cell['ecDNAs']:
            new_mutations = e['mutations'][:]
            next_mutation = mutate_region(new_mutations, ecdna_mu, next_mutation)
            ecDNA_pool.extend([{'mutations': new_mutations} for _ in range(2)])

    np.random.shuffle(ecDNA_pool)
    for e in ecDNA_pool:
        np.random.choice(daughter_cells)['ecDNAs'].append(e)

    return daughter_cells, next_mutation

def mutate_region(mutations, mutation_rate, next_mutation):
    """Append a Poisson number of new mutations; returns the next unused mutation id."""
    num_mutations = np.random.poisson(mutation_rate)
    mutations.extend(range(next_mutation, next_mutation + num_mutations))
    return next_mutation + num_mutations
