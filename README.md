# ecEvol

Code for "Inferring ecDNA Evolution from Allele Frequency Spectra".

ecEvol estimates the evolutionary parameters of an ecDNA from bulk whole-genome
sequencing: its emergence time `g`, selection coefficient `s`, mutation rate `mu`
and descendant fraction `phi`. It takes two allele-frequency spectra per tumour,
one from the ecDNA region and one from the rest of the genome, together with the
ecDNA copy number. Models are trained on forward simulations and applied to the
PCAWG and HMF cohorts.

## Contents

```
simulation/   forward simulator (sim_opt.py), its parallel driver, and the
              population-size checkpoint simulator
scripts/      models, training, inference and the ABC baseline
notebooks/    run_everything.ipynb: ablations, tuning, training, cohort inference
analysis/     figures and statistics in the paper, from the prediction tables
```

| Module | Output |
|---|---|
| `analysis/figure2_recovery.py`, `analysis/popsize.py` | Fig. 2b-g |
| `analysis/figure3_pcawg.py` | Fig. 3 |
| `analysis/figure4_hmf.py` | Fig. 4 |
| `analysis/figure5_survival.py` | Fig. 5 and the survival statistics |
| `analysis/simulation_envelope.py` | Supplementary Fig. 3 |
| `analysis/supp_model_perf.py` | Supplementary Fig. 5 |
| `analysis/supp_by_cancer.py` | Supplementary Fig. 6 |
| `analysis/supp_gene_violins.py` | Supplementary Fig. 7 |
| `analysis/supp_hq_purity.py` | Supplementary Fig. 8 |
| `analysis/supp_ensemble.py` | Supplementary Table 6 |

## Requirements

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Data

The simulated spectra, train/validation/test splits and trained model weights are
available at https://doi.org/10.17632/cvcbsvbrn5.1. Per-feature estimates for the
PCAWG and HMF cohorts are provided as Supplementary Data 1 of the paper.

PCAWG and HMF sequencing data are under controlled access and are not
redistributed here.

## Usage

Training and cohort inference are run from `notebooks/run_everything.ipynb`, which
was run on Google Colab with a GPU. Set the input and output paths in its first
two cells. Models were trained with seeds 42 to 51 at sequencing depths 30x, 50x
and 100x; reported estimates are the mean over the ten seeds.

The modules in `analysis/` produce the figures and statistics from the prediction
tables written by the notebook, for example:

```
python -m analysis.figure3_pcawg
```

They read from `data/` (layout in `analysis/common.py`) and write panels to
`figures/`; set `ECDNA_DATA_DIR` or `ECDNA_OUT_DIR` to use other locations.

## Citation

Manuscript in preparation.
