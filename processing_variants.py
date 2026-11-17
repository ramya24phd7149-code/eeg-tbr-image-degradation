"""
processing_variants.py  --  how much does the TBR group difference depend on spectral-processing choices?

Uses the same filtered, epoched and artifact-rejected data as full_pipeline.py (primary far-out fence) and recomputes the
global TBR with alternative, commonly used spectral-processing choices. All variants are listed below and ALL are reported
(none is selected after looking at the result).

Factors (fully crossed unless stated):
  window        : hann (primary), hamming
  nperseg       : 64, 128, 256 (primary), 384 (= whole 3-s epoch, no segment averaging); overlap = 50 % (none for 384)
  band summary  : "sum" of PSD bins in the band (primary) or "mean" PSD in the band
  aggregation   :
     perchan_epochavg  (primary) per channel: mean-over-epochs theta / mean-over-epochs beta, then mean over channels
     perchan_epochratio per channel: theta/beta of each epoch, mean over epochs, then mean over channels
     pooled            mean over channels of theta / mean over channels of beta (ratio of averages)
     chanavg_signal    channels averaged in the time domain first, then one spectrum per epoch, then epoch-averaged ratio
Self-contained: it repeats the data loading, filtering, epoching and rejection code of full_pipeline.py (copied unchanged), so it
can be run on its own. Check: if tbr_per_subject.csv from full_pipeline.py is in the output folder, the primary variant must
reproduce its TBR_global exactly (the script stops if it does not); the primary group medians are always printed.

Run:  python processing_variants.py        (environment variables REV_ADHD_DIR and REV_OUT as in full_pipeline.py; writes
      table_processing_variants.csv and SUMMARY_PASTE_BACK_VARIANTS.txt to REV_OUT)
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

WINDOWS = ["hann", "hamming"]
NPERSEGS = [64, 128, 256, 384]
BANDSUM = ["sum", "mean"]
AGGS = ["perchan_epochavg", "perchan_epochratio", "pooled", "chanavg_signal"]
PRIMARY = ("hann", 256, "sum", "perchan_epochavg")

def _band(P, f, band, how):
    m = band_idx(f, band)
    return P[:, m, ...].sum(axis=1) if how == "sum" else P[:, m, ...].mean(axis=1)

def variant_tbr(ep_keep, window, nperseg, how):
    """ep_keep: (n_ep, 384, n_ch). Returns dict agg -> TBR for one child."""
    nov = 0 if nperseg >= EPOCH else nperseg // 2
    win = get_window(window, nperseg)
    kw = dict(fs=FS, window=win, nperseg=nperseg, noverlap=nov, axis=1)
    f, P = welch(ep_keep.astype(np.float64), **kw)                      # (n_ep, n_f, n_ch)
    th, be = _band(P, f, THETA, how), _band(P, f, BETA, how)      # (n_ep, n_ch)
    out = {"perchan_epochavg": (th.mean(0) / be.mean(0)).mean(),
           "perchan_epochratio": (th / be).mean(0).mean(),
           "pooled": th.mean() / be.mean()}
    _, Pa = welch(ep_keep.astype(np.float64).mean(axis=2), **{**kw, "axis": 1})   # (n_ep, n_f)
    tha, bea = _band(Pa, f, THETA, how), _band(Pa, f, BETA, how)
    out["chanavg_signal"] = (tha / bea).mean()
    return out

def run(subj, ch):
    header("Processing variants: global TBR recomputed under alternative spectral choices")
    eps, p2p_max, flat = prepare_epochs(subj)
    thr = primary_threshold(p2p_max)
    rows = {}
    for pid, (g, sig) in subj.items():
        keep = (p2p_max[pid] <= thr) & (~flat[pid])
        if keep.sum() == 0: continue
        for w, n, how in itertools.product(WINDOWS, NPERSEGS, BANDSUM):
            for agg, v in variant_tbr(eps[pid][keep], w, n, how).items():
                rows.setdefault((w, n, how, agg), {})[pid] = v
    prim = pd.Series(rows[PRIMARY])
    ref_path = os.path.join(OUTPUT_DIR, "tbr_per_subject.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path).set_index("subject_id")
        err = float(np.max(np.abs(prim - ref.loc[prim.index, "TBR_global"])))
        log(f"Check: primary variant vs TBR_global in tbr_per_subject.csv: max abs difference = {err:.2e}")
        assert err < 1e-9, "primary variant does not reproduce TBR_global"
    else:
        log("tbr_per_subject.csv not found in the output folder: no exact check; compare the medians below with the main analysis.")
    g_ = {pid: v[0] for pid, v in subj.items()}
    log(f"Primary variant: global TBR median Control = {np.median([v for p, v in prim.items() if g_[p]=='Control']):.3f}, "
        f"ADHD = {np.median([v for p, v in prim.items() if g_[p]=='ADHD']):.3f}")
    grp = {pid: g for pid, (g, _) in subj.items()}
    out = []
    for key, d in rows.items():
        s = pd.Series(d); c = s[[p for p in s.index if grp[p] == "Control"]].values
        a = s[[p for p in s.index if grp[p] == "ADHD"]].values
        st = tbr_group_stats(c, a, B=N_BOOT)
        out.append(dict(window=key[0], nperseg=key[1], band_summary=key[2], aggregation=key[3], primary=(key == PRIMARY),
                        n_control=len(c), n_adhd=len(a), median_control=np.median(c), median_adhd=np.median(a),
                        mean_control=c.mean(), mean_adhd=a.mean(), higher_group=("Control" if np.median(c) > np.median(a) else "ADHD"),
                        p_mwu=st["p_mwu"], prob_superiority=st["prob_superiority"], cliffs_delta=st["cliffs_delta"]))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUTPUT_DIR, "table_processing_variants.csv"), index=False)
    n = len(df); nc = int((df.higher_group == "Control").sum())
    log(f"{n} variants; median TBR higher in Control in {nc}, in ADHD in {n - nc}.")
    log(f"Cliff's delta (Control minus ADHD) range over variants: {df.cliffs_delta.min():.3f} to {df.cliffs_delta.max():.3f}; "
           f"Mann-Whitney p range: {df.p_mwu.min():.4f} to {df.p_mwu.max():.4f}; p<0.05 in {int((df.p_mwu < .05).sum())}.")
    for agg, s in df.groupby("aggregation"):
        log(f"  {agg:20s} delta {s.cliffs_delta.min():+.3f}..{s.cliffs_delta.max():+.3f}; Control higher in {int((s.higher_group=='Control').sum())}/{len(s)}")
    log(df.drop(columns=["primary"]).round(4).to_string(index=False))
    with open(os.path.join(OUTPUT_DIR, "SUMMARY_PASTE_BACK_VARIANTS.txt"), "w") as f: f.write("\n".join(_LOG))
    return df

if __name__ == "__main__":
    subj, ch = load_adhd()
    run(subj, ch)
