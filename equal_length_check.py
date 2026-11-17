"""
equal_length_check.py -- does the Control > ADHD direction of the TBR depend on the different recording lengths?

In the dataset the recordings of children with ADHD are longer than those of controls (the task was self-paced). This script uses the
same filtered, epoched and artifact-rejected data as full_pipeline.py (primary far-out fence) and the primary TBR (Hann, 256 samples, band
power summed, ratio of epoch-averaged band powers per channel, mean over the 19 channels), but equalizes the amount of data per child:
  (a) first N retained epochs of every child, N = smallest number of retained epochs of any child;
  (b) N retained epochs drawn at random (100 draws, seed 42) with the same N: median and range of Cliff's delta and of the Mann-Whitney p;
  (c) N = 40: children with fewer than 40 retained epochs are excluded (their number is reported), first 40 and 100 random draws;
and reports the Spearman correlation between the global TBR (all retained epochs) and the number of retained epochs, over all children and within each group.
All results are listed; none is selected.

Run:  python equal_length_check.py     (env: REV_ADHD_DIR, REV_OUT)    Writes: table_equal_length.csv, SUMMARY_PASTE_BACK_EQUAL_LENGTH.txt
"""
import os, glob, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, butter, filtfilt, get_window
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------ config (as in full_pipeline.py) --
ENV = os.environ.get
RNG_SEED = 42
ADHD_DIR = ENV("REV_ADHD_DIR", "/kaggle/input/datasets/swarnaramya24phd7149/adhd-dataset19")
OUTPUT_DIR = ENV("REV_OUT", "/kaggle/working/revision_outputs_v2")
FS = 128
EPOCH = 3 * FS
BAND_FILTER = (1, 40)
THETA, BETA = (4, 8), (13, 30)
EXPECTED_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
FENCE_PRIMARY = 3.0
N_BOOT = 1000          # bootstrap replicates for the (secondary) interval estimates inside tbr_group_stats; tests and delta do not depend on it
TEST_MAX_SUBJECTS = int(ENV("REV_TEST_MAX_SUBJECTS", "0"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); _LOG.append(s)
def header(t):
    log("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

def hedges_g(a, b):
    """Hedges' g, a minus b."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))

def prob_superiority(x, y):
    """P(x > y) + 0.5 P(x == y)."""
    ys = np.sort(y)
    less = np.searchsorted(ys, x, "left"); le = np.searchsorted(ys, x, "right")
    return (less.sum() + 0.5 * (le - less).sum()) / (len(x) * len(y))

def describe(a):
    a = np.asarray(a, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=len(a), mean=a.mean(), sd=a.std(ddof=1) if len(a) > 1 else np.nan,
                median=med, q1=q1, q3=q3, min=a.min(), max=a.max())

def tbr_group_stats(x, y, B=N_BOOT, seed=RNG_SEED):
    """x = Control, y = ADHD. Rank-based primary + parametric on log scale."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    ps = prob_superiority(x, y)
    welch_raw = stats.ttest_ind(x, y, equal_var=False)
    welch_log = stats.ttest_ind(np.log(x), np.log(y), equal_var=False)
    rng = np.random.RandomState(seed)
    n1, n2 = len(x), len(y)
    lx, ly = np.log(x), np.log(y)
    bps, bmd, bg = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        xi = rng.randint(0, n1, n1); yi = rng.randint(0, n2, n2)
        bps[b] = prob_superiority(x[xi], y[yi])
        bmd[b] = np.median(x[xi]) - np.median(y[yi])
        bg[b] = hedges_g(lx[xi], ly[yi])
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    return dict(
        n_control=n1, n_adhd=n2, U=u.statistic, p_mwu=u.pvalue,
        prob_superiority=ps, ps_ci=ci(bps),
        cliffs_delta=2 * ps - 1, cliffs_delta_ci=tuple(2 * np.array(ci(bps)) - 1),
        median_diff=np.median(x) - np.median(y), median_diff_ci=ci(bmd),
        welch_raw_t=welch_raw.statistic, welch_raw_p=welch_raw.pvalue,
        welch_log_t=welch_log.statistic, welch_log_p=welch_log.pvalue,
        hedges_g_log=hedges_g(lx, ly), hedges_g_log_ci=ci(bg),
        hedges_g_raw=hedges_g(x, y))

def cliffs_delta(x, y):
    return 2 * prob_superiority(np.asarray(x, float), np.asarray(y, float)) - 1



def find_adhd_source():
    """Finds the ADHD/Control data under ADHD_DIR: the original IEEE .mat files (preferred), or a CSV with columns ID and Class."""
    mats = sorted(glob.glob(os.path.join(ADHD_DIR, "**", "*.mat"), recursive=True))
    if mats: return "mat", mats
    for c in sorted(glob.glob(os.path.join(ADHD_DIR, "**", "*.csv"), recursive=True)):
        try:
            if {"ID", "Class"} <= set(pd.read_csv(c, nrows=1).columns): return "csv", c
        except Exception:
            pass
    raise FileNotFoundError(f"No ADHD data found under {ADHD_DIR}: need the original .mat files or adhdata.csv (columns ID, Class)")

_B, _A = butter(4, [BAND_FILTER[0] / (FS / 2), BAND_FILTER[1] / (FS / 2)], btype="band")

def load_adhd():
    kind, src = find_adhd_source()
    ch = EXPECTED_CHANNELS          # column order of the CSV and of the .mat files = order of the dataset's Channel_Labels file
    subj = {}
    if kind == "csv":
        df = pd.read_csv(src)
        got = [c for c in df.columns if c not in ("ID", "Class")]
        if got != EXPECTED_CHANNELS: log(f"WARNING: channel order differs from Channel_Labels.docx order!\n  found {got}")
        ch = got
        for pid in list(pd.unique(df["ID"])):
            sdf = df[df.ID == pid]; subj[pid] = (sdf["Class"].iloc[0], sdf[ch].values.astype(np.float64))
        log(f"ADHD/Control CSV file: {src}")
    else:
        from scipy.io import loadmat
        for path in src:
            pid = os.path.splitext(os.path.basename(path))[0]
            group = "ADHD" if "ADHD_part" in path else ("Control" if "Control_part" in path else None)
            if group is None: continue
            m = loadmat(path); arr = m[pid] if pid in m else [v for k, v in m.items() if not k.startswith("__")][0]
            subj[pid] = (group, np.asarray(arr, dtype=np.float64))
        log(f"ADHD/Control .mat files: {len(subj)} files, e.g. {src[0]}")
    if TEST_MAX_SUBJECTS:
        keep = []
        for g in ["ADHD", "Control"]:
            keep += [i for i in subj if subj[i][0] == g][:TEST_MAX_SUBJECTS // 2]
        subj = {i: subj[i] for i in keep}
    log(f"  subjects={len(subj)}  (ADHD={sum(v[0]=='ADHD' for v in subj.values())}, "
        f"Control={sum(v[0]=='Control' for v in subj.values())});  channels={len(ch)}")
    return subj, ch

def filtered_epochs(sig):
    """Continuous filtering (all channels) then non-overlapping 3-s epochs (full epochs only)."""
    filt = np.column_stack([filtfilt(_B, _A, sig[:, c]) for c in range(sig.shape[1])]).astype(np.float32)
    n_ep = sig.shape[0] // EPOCH
    ep = filt[:n_ep * EPOCH].reshape(n_ep, EPOCH, sig.shape[1])
    return ep

def band_idx(f, band):
    return (f >= band[0]) & (f <= band[1])

def prepare_epochs(subj):
    """Continuous filtering -> 3-s epochs; returns epochs, epoch max-over-channel peak-to-peak, flat flags."""
    eps, flat_flags, p2p_max = {}, {}, {}
    for pid, (g, sig) in subj.items():
        ep = filtered_epochs(sig)
        eps[pid] = ep
        p2p = ep.max(axis=1) - ep.min(axis=1)
        p2p_max[pid] = p2p.max(axis=1)
        flat_flags[pid] = (ep.std(axis=1) < 1e-8).any(axis=1)
    return eps, p2p_max, flat_flags


def primary_threshold(p2p_max):
    pooled = np.concatenate(list(p2p_max.values()))
    q1, q3 = np.percentile(pooled, [25, 75])
    return q3 + FENCE_PRIMARY * (q3 - q1)       # same rule as compute_tbr in full_pipeline.py


def primary_tbr(ep):
    """Primary global TBR of one child from an array of epochs (n_ep, 384, n_ch)."""
    f, P = welch(ep.astype(np.float64), fs=FS, nperseg=256, noverlap=128, axis=1)
    th = P[:, band_idx(f, THETA), :].sum(axis=1).mean(axis=0); be = P[:, band_idx(f, BETA), :].sum(axis=1).mean(axis=0)
    return float((th / be).mean())

def compare(vals, grp):
    c = np.array([v for p, v in vals.items() if grp[p] == "Control"]); a = np.array([v for p, v in vals.items() if grp[p] == "ADHD"])
    u = stats.mannwhitneyu(c, a, alternative="two-sided")
    return dict(n_control=len(c), n_adhd=len(a), median_control=np.median(c), median_adhd=np.median(a), p_mwu=u.pvalue,
                cliffs_delta=2 * prob_superiority(c, a) - 1)

def run(subj, ch):
    header("Equal amount of data per child")
    eps, p2p_max, flat = prepare_epochs(subj)
    thr = primary_threshold(p2p_max)
    kept = {pid: eps[pid][(p2p_max[pid] <= thr) & (~flat[pid])] for pid in subj}
    kept = {p: e for p, e in kept.items() if len(e) > 0}
    grp = {p: subj[p][0] for p in kept}
    n_ret = {p: len(e) for p, e in kept.items()}
    full = {p: primary_tbr(e) for p, e in kept.items()}
    r0 = compare(full, grp)
    log(f"All retained epochs (primary analysis): Control median {r0['median_control']:.3f}, ADHD median {r0['median_adhd']:.3f}, p={r0['p_mwu']:.4f}, delta={r0['cliffs_delta']:.3f}")
    rows = [dict(analysis="all retained epochs (primary)", N=np.nan, draws=1, **r0)]
    for lab, N in (("N = smallest number of retained epochs", min(n_ret.values())), ("N = 40", 40)):
        use = {p: e for p, e in kept.items() if len(e) >= N}
        excl = [p for p in kept if p not in use]
        g2 = {p: grp[p] for p in use}
        log(f"\n{lab}: N={N} epochs ({N * 3} s) per child; excluded children (fewer than N retained epochs): {len(excl)} "
            f"(ADHD {sum(grp[p]=='ADHD' for p in excl)}, Control {sum(grp[p]=='Control' for p in excl)})")
        first = compare({p: primary_tbr(e[:N]) for p, e in use.items()}, g2)
        log(f"  first {N} epochs: Control median {first['median_control']:.3f}, ADHD median {first['median_adhd']:.3f}, p={first['p_mwu']:.4f}, delta={first['cliffs_delta']:.3f}")
        rows.append(dict(analysis=f"{lab}: first N epochs", N=N, draws=1, **first))
        rng = np.random.RandomState(RNG_SEED); res = []
        for _ in range(100):
            res.append(compare({p: primary_tbr(e[np.sort(rng.choice(len(e), N, replace=False))]) for p, e in use.items()}, g2))
        rd = pd.DataFrame(res)
        log(f"  100 random draws of {N} epochs: Control higher median in {int((rd.median_control > rd.median_adhd).sum())}/100; "
            f"delta median {rd.cliffs_delta.median():.3f} (range {rd.cliffs_delta.min():.3f} to {rd.cliffs_delta.max():.3f}); "
            f"p median {rd.p_mwu.median():.4f} (range {rd.p_mwu.min():.4f} to {rd.p_mwu.max():.4f}); p<0.05 in {int((rd.p_mwu < .05).sum())}/100")
        rows.append(dict(analysis=f"{lab}: random N epochs (median over 100 draws)", N=N, draws=100, n_control=int(rd.n_control.iloc[0]), n_adhd=int(rd.n_adhd.iloc[0]),
                         median_control=rd.median_control.median(), median_adhd=rd.median_adhd.median(), p_mwu=rd.p_mwu.median(), cliffs_delta=rd.cliffs_delta.median(),
                         delta_min=rd.cliffs_delta.min(), delta_max=rd.cliffs_delta.max(), p_min=rd.p_mwu.min(), p_max=rd.p_mwu.max(),
                         n_draws_control_higher=int((rd.median_control > rd.median_adhd).sum()), n_draws_p_below_005=int((rd.p_mwu < .05).sum())))
    log("")
    for lab, sel in (("all children", list(kept)), ("Control", [p for p in kept if grp[p] == "Control"]), ("ADHD", [p for p in kept if grp[p] == "ADHD"])):
        rho, pr = stats.spearmanr([n_ret[p] for p in sel], [full[p] for p in sel])
        log(f"Spearman correlation, global TBR vs number of retained epochs, {lab}: rho={rho:.3f}, p={pr:.4f}, n={len(sel)}")
        rows.append(dict(analysis=f"Spearman TBR vs retained epochs, {lab}", n_control=np.nan, spearman_rho=rho, spearman_p=pr, n=len(sel)))
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "table_equal_length.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "SUMMARY_PASTE_BACK_EQUAL_LENGTH.txt"), "w") as f: f.write("\n".join(_LOG))

if __name__ == "__main__":
    subj, ch = load_adhd()
    run(subj, ch)
