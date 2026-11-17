"""
extra_checks_standalone.py  --  ONE self-contained file (no other .py files needed). Run on Kaggle.

1. Which file(s) in the MindBigData folder are not recording files (14,013 files vs 14,012 recording files).
2. Theta power, beta power and TBR of the two groups separately, for the four electrode selections
   (primary processing: Hann window, 256-sample segments, summed PSD bins, per-channel epoch-averaged power, primary rejection).
3. Effect of the brightness reduction on SSIM, MSE and PSNR with and without noise over the real image pool
   (re-created from stimulus_manifest.csv of the main run, i.e. the same 254 images).

Environment variables (all optional):
  REV_IN   folder (searched recursively) with stimulus_manifest.csv and tbr_per_subject.csv of the main run
  REV_OUT  folder for the results (default /kaggle/working/extra_out)
  REV_ADHD_DIR, REV_MBD_DIR, REV_IMNET_DIR  input data folders (same defaults as full_pipeline.py)
The functions below are copied unchanged from full_pipeline.py / processing_variants.py.
"""
import os, re, glob, warnings
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from scipy.signal import welch, butter, filtfilt, get_window
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
warnings.filterwarnings("ignore", category=UserWarning)

ENV = os.environ.get
RNG_SEED = 42
ADHD_DIR = ENV("REV_ADHD_DIR", "/kaggle/input/datasets/swarnaramya24phd7149/adhd-dataset19")
MINDBIGDATA_DIR = ENV("REV_MBD_DIR", "/kaggle/input/datasets/swarnaramya24phd7149/mindbigdata1")
MINIIMAGENET_DIR = ENV("REV_IMNET_DIR", "/kaggle/input/datasets/deeptrial/miniimagenet/ImageNet-Mini")
IMAGES_DIR = os.path.join(MINIIMAGENET_DIR, "images")
OUTPUT_DIR = ENV("REV_OUT", "/kaggle/working/extra_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)
STIMULUS_SIZE = 300
FS = 128
EPOCH = 3 * FS
BAND_FILTER = (1, 40)
THETA, BETA = (4, 8), (13, 30)
EXPECTED_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
FENCE_PRIMARY = 3.0
N_BOOT = 1000
PSNR_CAP = 100.0
BASE_GAINS = (8.0, 20.0, 0.15)
SSIM_KW = dict(channel_axis=-1, data_range=255, win_size=7, K1=0.01, K2=0.03,
               gaussian_weights=False, use_sample_covariance=True)
TEST_MAX_SUBJECTS = 0
_LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _LOG.append(s)

_B, _A = butter(4, [BAND_FILTER[0] / (FS / 2), BAND_FILTER[1] / (FS / 2)], btype="band")

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

def degrade(img, NTBR, blur=True, noise=True, bright=True, gains=BASE_GAINS, seed=None):
    """Main pipeline operator. Order: blur -> noise(+clip) -> brightness. Degradation ~ (1-NTBR)."""
    bg, ng, brg = gains
    rng = np.random.RandomState(seed) if seed is not None else np.random
    inv = 1.0 - NTBR
    out = img
    if blur:
        k = int(round(1 + inv * bg)); k = k | 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    x = out.astype(np.float32)
    if noise:
        x = np.clip(x + rng.normal(0, inv * ng, img.shape).astype(np.float32), 0, 255)
    if bright:
        x = x * max(0.3, 1 - inv * brg)
    return np.clip(x, 0, 255).astype(np.uint8)

def metrics(orig, deg):
    s = ssim_fn(orig, deg, **SSIM_KW)
    mse = float(np.mean((orig.astype(np.float32) - deg.astype(np.float32)) ** 2))
    p = PSNR_CAP if mse == 0 else min(psnr_fn(orig, deg, data_range=255), PSNR_CAP)
    return s, mse, p

OUT = OUTPUT_DIR
IN = ENV("REV_IN", "/kaggle/input/datasets/swarnaramya24phd7149/final-plos")
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)
def _find_in(name, required=True):
    for d in [IN, OUT, "/kaggle/input/datasets/swarnaramya24phd7149/final-plos", "/kaggle/working"]:
        hits = glob.glob(os.path.join(d, "**", name), recursive=True)
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"{name} not found under {IN} (set REV_IN to the folder that holds it)")
    return None

SELECTIONS = {"Global (19 channels)": None,
              "Frontal (F3, F4, Fz)": ["F3", "F4", "Fz"],
              "Fronto-central (Fz, Cz, F3, F4, C3, C4)": ["Fz", "Cz", "F3", "F4", "C3", "C4"],
              "Cz": ["Cz"]}


# ---------------------------------------------------------------- check 1: non-recording files
def check_files():
    say("=" * 70 + "\nCheck 1: files in the MindBigData folder that are not recording files")
    pat = re.compile(r"MindBigData_Imagenet_Insight_n\d{8}_\d+_\d+_\d+\.csv$")
    other = []
    n = 0
    for root, _, files in os.walk(MINDBIGDATA_DIR):
        for f in files:
            n += 1
            if not pat.search(f):
                other.append(os.path.relpath(os.path.join(root, f), MINDBIGDATA_DIR))
    say(f"Total files: {n}; files not matching the recording-name pattern: {len(other)}")
    for o in other[:20]:
        say("  not a recording:", o)


# ---------------------------------------------------------------- check 2: theta and beta separately
def check_band_power():
    say("=" * 70 + "\nCheck 2: theta power, beta power and TBR separately (primary processing)")
    subj, ch = load_adhd()
    eps, p2p_max, flat = prepare_epochs(subj)
    thr = primary_threshold(p2p_max)
    win = get_window("hann", 256)
    rows = {}
    for pid, (g, sig) in subj.items():
        keep = (p2p_max[pid] <= thr) & (~flat[pid])
        if keep.sum() == 0:
            continue
        f, P = welch(eps[pid][keep].astype(np.float64), fs=FS, window=win, nperseg=256, noverlap=128, axis=1)
        th = P[:, band_idx(f, THETA), :].sum(axis=1).mean(0)      # per channel, mean over epochs
        be = P[:, band_idx(f, BETA), :].sum(axis=1).mean(0)
        rows[pid] = (g, th, be)
    # reproduce the primary TBR as a check
    ref_path = _find_in("tbr_per_subject.csv", required=False)
    if ref_path is None:
        say("tbr_per_subject.csv not found: reproduction check skipped (compare the TBR rows below with Table 2 of the paper).")
    else:
        ref = pd.read_csv(ref_path).set_index("subject_id")
        mine = pd.Series({p: (v[1] / v[2]).mean() for p, v in rows.items()})
        err = float(np.max(np.abs(mine - ref.loc[mine.index, "TBR_global"])))
        say(f"Check: TBR_global reproduced, max abs difference = {err:.2e}")
        assert err < 1e-9
    out = []
    for sel, names in SELECTIONS.items():
        idx = list(range(len(ch))) if names is None else [ch.index(n) for n in names]
        vals = {"theta": {}, "beta": {}, "TBR": {}}
        for p, (g, th, be) in rows.items():
            vals["theta"][p] = th[idx].mean(); vals["beta"][p] = be[idx].mean(); vals["TBR"][p] = (th[idx] / be[idx]).mean()
        for k, d in vals.items():
            s = pd.Series(d)
            c = s[[p for p in s.index if rows[p][0] == "Control"]].values
            a = s[[p for p in s.index if rows[p][0] == "ADHD"]].values
            st = tbr_group_stats(c, a, B=N_BOOT)
            out.append(dict(selection=sel, quantity=k, median_control=np.median(c), median_adhd=np.median(a),
                            p_mwu=st["p_mwu"], prob_superiority_C_gt_A=st["prob_superiority"], cliffs_delta=st["cliffs_delta"]))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "table_extra_band_power.csv"), index=False)
    say("(band power is the sum of PSD bins in the band, in squared recorded units per Hz; Control minus ADHD effect sizes)")
    say(df.round(4).to_string(index=False))


# ---------------------------------------------------------------- check 3: brightness reduction on the real pool
def load_pool():
    man = pd.read_csv(_find_in("stimulus_manifest.csv"))
    imgs = []
    for _, r in man.iterrows():
        bgr = cv2.imread(os.path.join(IMAGES_DIR, r.wnid, r.filename))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]; m = min(h, w)
        crop = rgb[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
        imgs.append(cv2.resize(crop, (STIMULUS_SIZE, STIMULUS_SIZE), interpolation=cv2.INTER_AREA))
    return imgs

def brightness_table(imgs, levels=(0.1, 0.5, 0.9)):
    rows = []
    for c, lv in enumerate(levels):
        for noise in (False, True):
            res = {True: [], False: []}
            for j, im in enumerate(imgs):
                for bright in (False, True):
                    d = degrade(im, lv, blur=True, noise=noise, bright=bright, seed=1000 + 10 * j + c)
                    res[bright].append(metrics(im, d))
            a = np.array(res[False]); b = np.array(res[True])      # columns: SSIM, MSE, PSNR
            rows.append(dict(NTBR=lv, noise=noise,
                             SSIM_without_dimming=a[:, 0].mean(), SSIM_with_dimming=b[:, 0].mean(),
                             share_of_images_where_dimming_raised_SSIM=float((b[:, 0] > a[:, 0]).mean()),
                             PSNR_without_dimming=a[:, 2].mean(), PSNR_with_dimming=b[:, 2].mean()))
    return pd.DataFrame(rows)

def check_brightness():
    say("=" * 70 + "\nCheck 3: effect of the brightness reduction on SSIM and PSNR, with and without noise (blur always on)")
    imgs = load_pool()
    say(f"Pool: {len(imgs)} images")
    df = brightness_table(imgs)
    df.to_csv(os.path.join(OUT, "table_extra_brightness.csv"), index=False)
    say(df.round(4).to_string(index=False))


if __name__ == "__main__":
    check_files()
    check_band_power()
    check_brightness()
    with open(os.path.join(OUT, "extra_checks_summary.txt"), "w") as fh:
        fh.write("\n".join(LOG))
