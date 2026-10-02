"""Supplementary figure: inferred parameters by ecDNA-amplified oncogene, PCAWG and HMF."""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import load_ensemble, fig_dir
from . import figure_style as st

st.apply()
OUT = fig_dir("Supplementary", "ECDNA_SUPP_DIR")
CANON = ['EGFR', 'MDM2', 'CDK4', 'ERBB2', 'MYC', 'CCND1', 'CCNE1', 'FGFR1',
         'KRAS', 'MET', 'MYCL1', 'CDK6']
MIN_N = 5
PARAMS = [('phi_hat', r'$\hat{\phi}$  descendant fraction'),
          ('s_hat', r'$\hat{s}$  selection'),
          ('log10mu', r'$\log_{10}\hat{\mu}$  mutation rate'),
          ('g_exp', r'$\hat{g}$  emergence time')]


def genes(v):
    if pd.isna(v):
        return set()
    return {x for x in re.split(r"[;,\[\]'\" ]+", str(v)) if x and x.lower() != 'nan'}


pc = load_ensemble('pcawg').dropna(subset=['g_exp']).copy()
hm = load_ensemble('hmf').dropna(subset=['g_exp']).copy()
pc['gl'] = pc.Oncogenes.map(genes)
hm['gl'] = hm.agg_ecdna_oncogenes_unique.map(genes)
for d in (pc, hm):
    d['log10mu'] = np.log10(d.mu_hat.clip(lower=1))

cnt = pd.DataFrame([(g, pc.gl.map(lambda s: g in s).sum(), hm.gl.map(lambda s: g in s).sum())
                    for g in CANON], columns=['gene', 'pcawg', 'hmf'])
print(cnt.to_string(index=False))
keep = cnt[(cnt.pcawg >= MIN_N) | (cnt.hmf >= MIN_N)].gene.tolist()


def subset(d, g):
    return d[d.gl.map(len) == 0] if g == 'no oncogene' else d[d.gl.map(lambda s: g in s)]


# order by pooled median emergence time, 'no oncogene' last as the baseline
order = sorted(keep, key=lambda g: pd.concat([subset(pc, g), subset(hm, g)]).g_exp.median())
order.append('no oncogene')

rng = np.random.RandomState(0)
fig, axes = plt.subplots(len(PARAMS), 1, figsize=(7.2, 8.6), sharex=True)
for ax, (col, lab) in zip(axes, PARAMS):
    for i, g in enumerate(order):
        for off, d, c in [(-0.2, pc, st.COHORT_PCAWG), (0.2, hm, st.COHORT_HMF)]:
            v = subset(d, g)[col].dropna().values
            x = i + off
            if len(v) >= 3 and np.ptp(v) > 0:
                parts = ax.violinplot([v], positions=[x], widths=0.36, showextrema=False)
                for b in parts['bodies']:
                    b.set_facecolor(c); b.set_alpha(0.28); b.set_edgecolor(c); b.set_linewidth(0.5)
            if len(v):
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=2.2, color=c,
                           alpha=0.55, lw=0, rasterized=True, zorder=3)
                ax.plot([x - 0.11, x + 0.11], [np.median(v)] * 2, color='0.1', lw=1.1, zorder=4)
    ax.set_ylabel(lab, fontsize=7)
    if col == 'phi_hat':
        ax.set_ylim(-0.03, 1.05)
    ax.axvline(len(order) - 1.5, color='0.8', lw=0.6, ls='--', zorder=0)

axes[-1].set_xticks(range(len(order)))
axes[-1].set_xticklabels(
    [f"{g}\n{len(subset(pc, g))}|{len(subset(hm, g))}" for g in order], fontsize=6.2)
axes[-1].set_xlabel('oncogene on ecDNA  (n PCAWG | n HMF)', fontsize=7)
axes[0].legend(handles=[Line2D([], [], color=st.COHORT_PCAWG, lw=4, alpha=0.5, label='PCAWG'),
                        Line2D([], [], color=st.COHORT_HMF, lw=4, alpha=0.5, label='HMF')],
               frameon=False, fontsize=6.5, loc='lower left', ncol=2)
fig.tight_layout(h_pad=0.4)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'gene_violins.png')
fig.savefig(path, dpi=250)
print('saved', path)

print('\nmedians (PCAWG | HMF):')
for g in order:
    a, b = subset(pc, g), subset(hm, g)
    print('  %-11s ' % g + '  '.join(
        '%s %.2f|%.2f' % (c.split('_')[0], a[c].median() if len(a) else np.nan,
                          b[c].median() if len(b) else np.nan) for c, _ in PARAMS))
