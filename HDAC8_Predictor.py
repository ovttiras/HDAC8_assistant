######################
# Import libraries
######################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from molvs import standardize_smiles
from math import pi
from pathlib import Path
import time
import json
import urllib.request
import subprocess
import sys
import importlib
import shutil
import os
import textwrap
from functools import lru_cache

from hdac8_consensus import HDAC8ConsensusModel


######################
# Page Title
######################

st.set_page_config(page_title="HDAC8 Assistant", layout="wide")

st.write("<h1 style='text-align: center; color: #FF7F50;'> HDAC8 Assistant</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: #483D8B;'> The app enables the generation and evaluation of chemicals as HDAC8 inhibitors.</h3>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
col1, col2, col3, col4, col5 = st.columns(5)


with col1:
   st.header("Molecule generation")
   st.image("figures/molecule.png", width=125)
   st.text_area('Text to analyze', '''This application generates candidate HDAC8 inhibitors using a SMILES-RNN model pretrained on ChEMBL drug-like compounds. The model is optimized via reinforcement learning (REINVENT) to maximize predicted pIC50 values and ensure applicability domain compliance. A weighted CatBoost consensus QSAR model (ECFP4, MACCS, PubChem and Klekota–Roth fingerprints) serves as the core of the RL reward function. Generated structures undergo deduplication, validity checks, and optional ring-system filtering; they are then ranked by predicted activity and annotated with synthetic accessibility scores (SAScore) and Muegge drug-likeness criteria.''', height=350, label_visibility="hidden" )


with col2:
   st.header("Machine learning")
   st.image("figures/machine-learning.png", width=125)
   st.text_area('Text to analyze', '''This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models built on curated HDAC8 pChEMBL data. Five CatBoost regressors are trained on ECFP4, MACCS, RDKit 2D, PubChem and Klekota–Roth descriptor families; their outputs are combined by non-negative simplex weights (late fusion). The RDKit member currently has zero weight and is not used at inference. Each prediction is reported as pIC50 ± σ_consensus (weighted disagreement of active family models) together with applicability-domain status.''', height=350, label_visibility="hidden" )


with col3:
   st.header("OECD rules")
   st.image("figures/target.png", width=125)
   st.text_area('Text to analyze', '''We follow the best practices for model development and validation recommended by guidelines of the Organization for Economic Cooperation and Development (OECD). Prediction uncertainty is dual-component: (1) epistemic σ_consensus from weighted CatBoost consensus variance; (2) structural distance to the training set via nearest-neighbour Tanimoto distance D = 1 − Tc on the concatenated SHAP–mRMR pooled binary fingerprints (ECFP4, MACCS, PubChem, Klekota–Roth). The AD cutoff is Dcutoff = ⟨D⟩ + Zs, where Z = 0.5. Molecules with D > Dcutoff are flagged Outside AD (low confidence). RDKit 2D descriptors are excluded from the AD metric.''', height=350, label_visibility="hidden" )
# st.write('Sentiment:', run_sentiment_analysis(txt))


with col4:
   st.header("Muegge's rules")
   st.image("figures/puzzle-piece.png", width=125)
   st.text_area('Text to analyze', '''Estimating the drug-likeness of a compound is an important factor in drug development. Muegge's drug-likeness rules were introduced to estimate the potential of a compound to be a drug. Our drug-likeness radar is displayed for a quick assessment of the compliance of the tested compound with the Muegge rules. The application also provides structural analysis for identifying preferred or undesirable molecular fragments.''', height=350, label_visibility="hidden" )
with col5:
   st.header("Structural Alerts")
   st.image("figures/alert.png", width=125)
   st.text_area('Text to analyze', '''Brenk filters which consists in a list of 105 fragments to be putatively toxic, chemically reactive, metabolically unstable or to bear properties responsible for poor pharmacokinetics. PAINS  are molecules containing substructures showing potent response in assays irrespective of the protein target. Such fragments, yielding false positive biological output.''', height=350, label_visibility="hidden" )

with open("manual.pdf", "rb") as file:
    btn=st.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of HDAC8 Assistant web application.pdf",
    mime="application/octet-stream"
)

def rdkit_numpy_convert(f_vs):
    output = []
    for f in f_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)

def try_parse_smiles_from_ketcher(s):
    """
    Try several ways to parse SMILES from Ketcher (export can break kekulization).
    Returns an RDKit Mol or None.
    """
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    # 1) Default parse
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return mol
    except Exception:
        pass
    # 2) No sanitize first, then full sanitize
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(mol)
            return mol
    except Exception:
        pass
    # 3) Sanitize without kekulize, then set aromaticity
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            Chem.SetAromaticity(mol)
            return mol
    except Exception:
        pass
    return None


def _download_if_missing(url: str, destination: Path):
    """Download file only if it does not exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, str(destination))


def _prepare_smiles_rnn_assets(base_dir: Path):
    """Ensure SMILES-RNN scripts, checkpoint and default config exist in DataGeneration."""
    assets = {
        "reinforcement_learning.py": "https://raw.githubusercontent.com/MorganCThomas/SMILES-RNN/main/scripts/reinforcement_learning.py",
        "ChEMBL28pur.ckpt": "https://raw.githubusercontent.com/MorganCThomas/SMILES-RNN/main/priors/ChEMBL28pur.ckpt",
    }
    for filename, url in assets.items():
        _download_if_missing(url, base_dir / filename)


def _ensure_hdac8_hook(rl_script_path: Path):
    """Register HDAC8 reward class in copied reinforcement_learning.py if absent."""
    rl_text = rl_script_path.read_text(encoding="utf-8")
    hook = textwrap.dedent(
        """
        import molscore.scoring_functions as molscore_sfs
        from hdac8_qsar_reward import HDAC8QSARReward

        if all(sf.__name__ != "HDAC8QSARReward" for sf in molscore_sfs.all_scoring_functions):
            molscore_sfs.all_scoring_functions.append(HDAC8QSARReward)
        """
    ).strip()
    marker = "from molscore.manager import MolScore"
    if hook not in rl_text:
        rl_text = rl_text.replace(marker, marker + "\n" + hook, 1)
        rl_script_path.write_text(rl_text, encoding="utf-8")


def _build_smiles_rnn_config(work_dir: Path, models_dir: Path) -> Path:
    """Create per-run config with consensus pIC50 + Tanimoto AD reward."""
    dst = work_dir / "smiles_rnn_config.json"
    models_dir = Path(models_dir).resolve()

    input_dict = {
        "task": "HDAC8_pIC50_AD_reward",
        "output_dir": "./",
        "load_from_previous": False,
        "logging": False,
        "monitor_app": False,
        "diversity_filter": {
            "run": True,
            "name": "Unique",
            "parameters": {},
        },
        "scoring_functions": [
            {
                "name": "HDAC8QSARReward",
                "run": True,
                "parameters": {
                    "prefix": "hdac8",
                    "models_dir": str(models_dir),
                    "padel_threads": 2,
                },
            }
        ],
        "scoring": {
            "method": "single",
            "metrics": [
                {
                    "name": "hdac8_pIC50",
                    "weight": 1.0,
                    "modifier": "lin_thresh",
                    "parameters": {
                        "objective": "maximize",
                        "upper": 8.5,
                        "lower": 0.0,
                        "buffer": 8.5,
                    },
                },
                {
                    "name": "hdac8_in_AD",
                    "weight": 1.0,
                    "modifier": "step",
                    "filter": True,
                    "parameters": {
                        "objective": "maximize",
                        "upper": 1.0,
                        "lower": 0.0,
                    },
                },
            ],
        },
    }

    with dst.open("w", encoding="utf-8") as ofs:
        json.dump(input_dict, ofs, indent=2)
    return dst


def _find_smiles_rnn_result_dir(search_dir: Path):
    """Find latest output directory that contains scores.csv."""
    score_files = sorted(search_dir.rglob("scores.csv"), key=lambda p: p.stat().st_mtime)
    for score_file in reversed(score_files):
        if score_file.is_file():
            return score_file.parent
    return None


def _stage_rl_run_files(base_dir: Path, run_dir: Path) -> Path:
    """Copy static assets from DataGeneration to run folder and build per-run config."""
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    for src in (
        base_dir / "reinforcement_learning.py",
        base_dir / "ChEMBL28pur.ckpt",
        base_dir / "hdac8_qsar_reward.py",
        base_dir / "sitecustomize.py",
        repo_root / "hdac8_consensus.py",
    ):
        if not src.exists():
            raise RuntimeError(f"Required asset is missing: {src}")
        shutil.copy2(src, run_dir / src.name)

    _ensure_hdac8_hook(run_dir / "reinforcement_learning.py")

    models_dir = repo_root / "Models"
    required = [
        models_dir / "pool_feature_names.pkl",
        models_dir / "consensus_simplex_weights.json",
        models_dir / "x_ad_tr.npz",
        models_dir / "ad_meta.json",
        models_dir / "fingerprints_xml" / "PubchemFingerprinter.xml",
        models_dir / "fingerprints_xml" / "KlekotaRothFingerprinter.xml",
    ]
    for fam in ("ECFP4", "MACCS", "PubChem", "KlekotaRoth"):
        required.append(models_dir / f"CatBoost_{fam}.pkl")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("QSAR consensus assets are missing:\n" + "\n".join(missing))

    return _build_smiles_rnn_config(run_dir, models_dir)


def _run_smiles_rnn_rl(
    work_dir: Path,
    config_path: Path,
    n_steps: int = 250,
    batch_size: int = 128,
    seed: int = 42,
    progress_callback=None,
):
    """Execute reinforcement learning generation using SMILES-RNN."""
    if importlib.util.find_spec("smilesrnn") is None or importlib.util.find_spec("molscore") is None:
        raise RuntimeError(
            "SMILES-RNN dependencies are missing. Install `smiles-rnn` and `molscore` first "
            "(same requirements as in notebook)."
        )

    device = "cpu"
    if importlib.util.find_spec("torch") is not None:
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                device = "gpu"
        except Exception:
            device = "cpu"

    command = [
        sys.executable,
        "reinforcement_learning.py",
        "-p",
        "ChEMBL28pur.ckpt",
        "-m",
        str(config_path.name),
        "--model",
        "RNN",
        "-d",
        device,
        "--seed",
        str(int(seed)),
        "RV2",
        "--n_steps",
        str(int(n_steps)),
        "--batch_size",
        str(int(batch_size)),
    ]
    stdout_log = work_dir / "rl_stdout.log"
    stderr_log = work_dir / "rl_stderr.log"
    stdout_log.write_text("", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")
    with open(stdout_log, "a", encoding="utf-8", errors="replace") as out_f, open(
        stderr_log, "a", encoding="utf-8", errors="replace"
    ) as err_f:
        process = subprocess.Popen(
            command,
            cwd=str(work_dir),
            stdout=out_f,
            stderr=err_f,
            text=True,
            env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        last_reported_step = -1
        if progress_callback is not None:
            progress_callback(0, int(n_steps))

        # No wall-clock timeout: full RL (especially on CPU) can exceed 1 hour.
        while process.poll() is None:
            step_files = list(work_dir.glob("**/iterations/*_scores.csv"))
            current_step = min(len(step_files), int(n_steps))
            if progress_callback is not None and current_step != last_reported_step:
                progress_callback(current_step, int(n_steps))
                last_reported_step = current_step
            time.sleep(0.5)

    if process.returncode != 0:
        stderr_tail = stderr_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        stdout_tail = stdout_log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"SMILES-RNN RL run failed (device={device}).\n"
            f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}"
        )
    if progress_callback is not None:
        progress_callback(int(n_steps), int(n_steps))
    return {
        "device": device,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }


@lru_cache(maxsize=1)
def _load_sascorer_module():
    """
    Ertl synthetic accessibility (sascorer). Layout differs by RDKit build:
    Chem.SA_Score, rdkit.Contrib.SA_Score, or Contrib/SA_Score next to RDContribDir.
    """
    try:
        from rdkit.Chem.SA_Score import sascorer as _s

        return _s
    except ImportError:
        pass
    try:
        from rdkit.Contrib.SA_Score import sascorer as _s

        return _s
    except ImportError:
        pass
    from rdkit.Chem import RDConfig

    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if not os.path.isdir(sa_dir):
        raise ImportError(
            "RDKit SA_Score (sascorer) is missing: no rdkit.Chem.SA_Score, no rdkit.Contrib.SA_Score, "
            f"and no folder {sa_dir}. Install a full RDKit build (e.g. conda-forge) or add Contrib SA_Score."
        )
    if sa_dir not in sys.path:
        sys.path.insert(0, sa_dir)
    import sascorer as _s

    return _s


def _prune_datagen_runs(runs_root: Path, keep: int = 2) -> None:
    """Удаляет старые каталоги DataGeneration/runs/<дата_время>/, оставляя только `keep` новейших."""
    if keep < 1 or not runs_root.is_dir():
        return
    subdirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if len(subdirs) <= keep:
        return
    subdirs.sort(key=lambda p: p.name, reverse=True)
    for old in subdirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def _ratio(numerator, denominator):
    if denominator is None or int(denominator) == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _fmt_pct(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _fmt_float(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _smiles_column(df):
    for name in ("smiles", "SMILES", "generated_smiles"):
        if name in df.columns:
            return name
    return None


def _reward_columns(df):
    ad_cols = [c for c in df.columns if str(c).endswith("_in_AD")]
    pic50_cols = [c for c in df.columns if str(c).endswith("_pIC50")]
    return (ad_cols[0] if ad_cols else None, pic50_cols[0] if pic50_cols else None)


def _step_index_from_name(path: Path):
    stem = path.name.split("_")[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _mol_from_smiles_silent(smiles):
    if smiles is None:
        return None
    try:
        if pd.isna(smiles):
            return None
    except (TypeError, ValueError):
        pass
    text = str(smiles).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def _inchi_from_mol(mol):
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
    except Exception:
        return None
    if not inchi:
        return None
    return str(inchi)


def _activity_values(df, pic50_col):
    if df is None or df.empty or not pic50_col or pic50_col not in df.columns:
        return np.array([], dtype=float)
    smi_col = _smiles_column(df)
    if smi_col is None:
        return np.array([], dtype=float)
    values = []
    for smi, pred in zip(df[smi_col].tolist(), df[pic50_col].tolist()):
        if _mol_from_smiles_silent(smi) is None:
            continue
        try:
            val = float(pred)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            values.append(val)
    return np.asarray(values, dtype=float)


def _activity_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else np.nan,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _read_rl_generation_tables(result_dir: Path):
    """Load all sampled structures, step 0 (prior), pooled RL steps, and last iteration."""
    iter_dir = result_dir / "iterations"
    iter_files = sorted(iter_dir.glob("*_scores.csv")) if iter_dir.is_dir() else []
    if iter_files:
        frames = [pd.read_csv(p) for p in iter_files]
        all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        before_step = _step_index_from_name(iter_files[0])
        after_step = _step_index_from_name(iter_files[-1])
        before_df = frames[0]
        post_rl_df = pd.concat(frames[1:], ignore_index=True) if len(frames) > 1 else pd.DataFrame()
        return all_df, before_df, frames[-1], post_rl_df, before_step, after_step

    scores_path = result_dir / "scores.csv"
    df = pd.read_csv(scores_path) if scores_path.is_file() else pd.DataFrame()
    if df.empty or "step" not in df.columns:
        return df, None, None, pd.DataFrame(), None, None
    steps = pd.to_numeric(df["step"], errors="coerce")
    if not steps.notna().any():
        return df, None, None, pd.DataFrame(), None, None
    smin = int(steps.min())
    smax = int(steps.max())
    before_df = df.loc[steps == smin].copy()
    post_rl_df = df.loc[steps > smin].copy()
    return df, before_df, df.loc[steps == smax].copy(), post_rl_df, smin, smax


def _compute_generation_phase_metrics(df, ad_col, pic50_col, exp_keys):
    """MOSES-style metrics for one generator phase (step 0 or pooled RL steps)."""
    smi_col = _smiles_column(df) if df is not None and not df.empty else None
    n_total = 0 if df is None or df.empty else int(len(df))
    n_valid = 0
    unique_rows = []
    seen = set()
    if smi_col is not None and n_total > 0:
        ad_series = (
            pd.to_numeric(df[ad_col], errors="coerce")
            if ad_col and ad_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        for smi, ad_val in zip(df[smi_col].tolist(), ad_series.tolist()):
            mol = _mol_from_smiles_silent(smi)
            if mol is None:
                continue
            n_valid += 1
            try:
                cansmi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if cansmi in seen:
                continue
            seen.add(cansmi)
            unique_rows.append((mol, cansmi, ad_val))

    n_unique = len(unique_rows)
    n_novel = 0
    n_in_ad = 0
    n_inchi_ok = 0
    for mol, _cansmi, ad_val in unique_rows:
        try:
            in_ad = float(ad_val) >= 0.5 if ad_val is not None and np.isfinite(float(ad_val)) else False
        except (TypeError, ValueError):
            in_ad = False
        if in_ad:
            n_in_ad += 1
        inchi = _inchi_from_mol(mol)
        if inchi is None:
            continue
        n_inchi_ok += 1
        if inchi not in exp_keys:
            n_novel += 1

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_novel": n_novel,
        "n_in_ad": n_in_ad,
        "n_inchi_ok": n_inchi_ok,
        "validity": _ratio(n_valid, n_total),
        "uniqueness": _ratio(n_unique, n_valid),
        "novelty": _ratio(n_novel, n_inchi_ok),
        "ad_proportion": _ratio(n_in_ad, n_unique),
        "pic50": _activity_stats(_activity_values(df, pic50_col)),
    }


def _generation_comparison_column_labels(before_step, after_step, peak_step=None):
    if before_step is not None:
        pre_col = f"Pre-RL — step {before_step} (prior model, before RL)"
    else:
        pre_col = "Pre-RL (prior model, before RL)"
    if (
        before_step is not None
        and after_step is not None
        and after_step > before_step
    ):
        post_col = f"Post-RL — steps {before_step + 1}–{after_step} (agent model, RL applied)"
    elif before_step is not None:
        post_col = f"Post-RL — steps ≥ {before_step + 1} (agent model, RL applied)"
    else:
        post_col = "Post-RL (agent model, RL applied)"
    if peak_step is not None:
        peak_col = (
            f"Peak activity — step {peak_step} "
            "(highest mean predicted pIC50)"
        )
    else:
        peak_col = "Peak activity (highest mean predicted pIC50)"
    return pre_col, post_col, peak_col


def _find_peak_pic50_step(all_df, pic50_col, before_step):
    """Return (peak_step, peak_step_df) for the RL step with highest mean predicted pIC50."""
    if all_df is None or all_df.empty or not pic50_col or "step" not in all_df.columns:
        return None, None
    steps = pd.to_numeric(all_df["step"], errors="coerce")
    if not steps.notna().any():
        return None, None
    if before_step is not None:
        rl_mask = steps > before_step
    else:
        smin = int(steps.min())
        rl_mask = steps > smin
    if not rl_mask.any():
        return None, None

    best_step = None
    best_mean = -np.inf
    for step_val in sorted(steps.loc[rl_mask].dropna().astype(int).unique()):
        group = all_df.loc[steps == step_val]
        stats = _activity_stats(_activity_values(group, pic50_col))
        mean = stats.get("mean")
        if mean is not None and np.isfinite(mean) and mean > best_mean:
            best_mean = mean
            best_step = int(step_val)

    if best_step is None:
        return None, None
    return best_step, all_df.loc[steps == best_step].copy()


def _fmt_mean_std(stats):
    mean = (stats or {}).get("mean")
    std = (stats or {}).get("std")
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "n/a"
    if std is None or (isinstance(std, float) and not np.isfinite(std)):
        return f"{float(mean):.2f}"
    return f"{float(mean):.2f} ± {float(std):.2f}"


def _build_generation_comparison_table(
    before_metrics,
    post_rl_metrics,
    before_step,
    after_step,
    peak_metrics=None,
    peak_step=None,
):
    pre_col, post_col, peak_col = _generation_comparison_column_labels(
        before_step, after_step, peak_step
    )
    rows = [
        (
            "Total generated structures",
            lambda m: f"{int(m.get('n_total') or 0):,}",
        ),
        ("Validity (%)", lambda m: _fmt_pct(m.get("validity"))),
        ("Uniqueness (%)", lambda m: _fmt_pct(m.get("uniqueness"))),
        (
            "Novelty (% vs experimental HDAC8)",
            lambda m: _fmt_pct(m.get("novelty")),
        ),
        ("Proportion within AD (S_AD = 1) (%)", lambda m: _fmt_pct(m.get("ad_proportion"))),
        ("Mean predicted pIC50", lambda m: _fmt_mean_std(m.get("pic50"))),
    ]
    peak_metrics = peak_metrics or {}
    return pd.DataFrame(
        [
            {
                "Metric": label,
                pre_col: fmt(before_metrics),
                post_col: fmt(post_rl_metrics),
                peak_col: fmt(peak_metrics),
            }
            for label, fmt in rows
        ]
    )


def compute_generation_quality_report(result_dir: Path):
    """
    Publication metrics for de novo generation (MOSES-style):
    total generated, validity, uniqueness, novelty, AD coverage,
    and predicted pIC50 distributions before vs after RL.
    Novelty is the fraction of unique valid structures whose InChI is absent
    from datasets/HDAC8_exp_data_inchi.csv.
    """
    all_df, before_df, _after_df, post_rl_df, before_step, after_step = _read_rl_generation_tables(
        result_dir
    )
    ad_col, pic50_col = _reward_columns(all_df) if all_df is not None and not all_df.empty else (None, None)

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    try:
        exp_keys = experimental_hdac8_inchi_keys()
        before_metrics = _compute_generation_phase_metrics(before_df, ad_col, pic50_col, exp_keys)
        post_rl_metrics = _compute_generation_phase_metrics(post_rl_df, ad_col, pic50_col, exp_keys)
        peak_step, peak_df = _find_peak_pic50_step(all_df, pic50_col, before_step)
        peak_metrics = _compute_generation_phase_metrics(peak_df, ad_col, pic50_col, exp_keys)
        all_metrics = _compute_generation_phase_metrics(all_df, ad_col, pic50_col, exp_keys)
        comparison_df = _build_generation_comparison_table(
            before_metrics,
            post_rl_metrics,
            before_step,
            after_step,
            peak_metrics,
            peak_step,
        )

        n_total = all_metrics["n_total"]
        n_valid = all_metrics["n_valid"]
        n_unique = all_metrics["n_unique"]
        n_novel = all_metrics["n_novel"]
        n_in_ad = all_metrics["n_in_ad"]
        n_inchi_ok = all_metrics["n_inchi_ok"]
        validity = all_metrics["validity"]
        uniqueness = all_metrics["uniqueness"]
        novelty = all_metrics["novelty"]
        ad_proportion = all_metrics["ad_proportion"]
        before_stats = before_metrics["pic50"]
        after_stats = post_rl_metrics["pic50"]
        peak_stats = peak_metrics["pic50"]
    finally:
        lg.setLevel(RDLogger.WARNING)

    metrics_rows = [
        {
            "metric": "total_generated",
            "value": n_total,
            "numerator": n_total,
            "denominator": "",
            "unit": "count",
            "definition": "Total SMILES sampled during the RL run (all iterations).",
        },
        {
            "metric": "n_valid",
            "value": n_valid,
            "numerator": n_valid,
            "denominator": n_total,
            "unit": "count",
            "definition": "Generated SMILES parsed by RDKit (MolFromSmiles).",
        },
        {
            "metric": "validity",
            "value": validity,
            "numerator": n_valid,
            "denominator": n_total,
            "unit": "fraction",
            "definition": "n_valid / total_generated.",
        },
        {
            "metric": "n_unique_valid",
            "value": n_unique,
            "numerator": n_unique,
            "denominator": n_valid,
            "unit": "count",
            "definition": "Unique valid molecules by canonical SMILES.",
        },
        {
            "metric": "uniqueness",
            "value": uniqueness,
            "numerator": n_unique,
            "denominator": n_valid,
            "unit": "fraction",
            "definition": "n_unique_valid / n_valid.",
        },
        {
            "metric": "n_novel",
            "value": n_novel,
            "numerator": n_novel,
            "denominator": n_inchi_ok,
            "unit": "count",
            "definition": "Unique valid molecules whose InChI is absent from HDAC8_exp_data_inchi.csv.",
        },
        {
            "metric": "novelty",
            "value": novelty,
            "numerator": n_novel,
            "denominator": n_inchi_ok,
            "unit": "fraction",
            "definition": "n_novel / n_unique_valid with a valid InChI (vs experimental HDAC8 set).",
        },
        {
            "metric": "n_in_AD",
            "value": n_in_ad,
            "numerator": n_in_ad,
            "denominator": n_unique,
            "unit": "count",
            "definition": "Unique valid molecules with QSAR applicability-domain flag in_AD >= 0.5.",
        },
        {
            "metric": "proportion_in_AD",
            "value": ad_proportion,
            "numerator": n_in_ad,
            "denominator": n_unique,
            "unit": "fraction",
            "definition": "n_in_AD / n_unique_valid.",
        },
        {
            "metric": "before_rl_step",
            "value": "" if before_step is None else before_step,
            "numerator": "",
            "denominator": "",
            "unit": "index",
            "definition": "First RL iteration (pretrained prior sampling, before optimization).",
        },
        {
            "metric": "pIC50_n_before_RL",
            "value": before_stats["n"],
            "numerator": before_stats["n"],
            "denominator": "",
            "unit": "count",
            "definition": "Valid molecules in the first RL iteration used for the activity distribution.",
        },
        {
            "metric": "pIC50_mean_before_RL",
            "value": before_stats["mean"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Mean predicted HDAC8 pIC50 before RL (first iteration, valid molecules).",
        },
        {
            "metric": "pIC50_median_before_RL",
            "value": before_stats["median"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Median predicted HDAC8 pIC50 before RL.",
        },
        {
            "metric": "pIC50_std_before_RL",
            "value": before_stats["std"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Sample standard deviation of predicted pIC50 before RL.",
        },
        {
            "metric": "after_rl_step",
            "value": "" if after_step is None else after_step,
            "numerator": "",
            "denominator": "",
            "unit": "index",
            "definition": "Last RL iteration index (final agent checkpoint).",
        },
        {
            "metric": "pIC50_n_after_RL",
            "value": after_stats["n"],
            "numerator": after_stats["n"],
            "denominator": "",
            "unit": "count",
            "definition": "Valid molecules across RL steps 1–N used for the post-RL activity distribution.",
        },
        {
            "metric": "pIC50_mean_after_RL",
            "value": after_stats["mean"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Mean predicted HDAC8 pIC50 after RL (RL steps 1–N, valid molecules).",
        },
        {
            "metric": "pIC50_median_after_RL",
            "value": after_stats["median"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Median predicted HDAC8 pIC50 after RL (RL steps 1–N).",
        },
        {
            "metric": "pIC50_std_after_RL",
            "value": after_stats["std"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Sample standard deviation of predicted pIC50 after RL (RL steps 1–N).",
        },
        {
            "metric": "peak_rl_step",
            "value": "" if peak_step is None else peak_step,
            "numerator": "",
            "denominator": "",
            "unit": "index",
            "definition": "RL iteration with the highest mean predicted pIC50 among steps 1–N.",
        },
        {
            "metric": "pIC50_n_peak_RL",
            "value": peak_stats["n"],
            "numerator": peak_stats["n"],
            "denominator": "",
            "unit": "count",
            "definition": "Valid molecules at the peak-activity RL step used for the activity distribution.",
        },
        {
            "metric": "pIC50_mean_peak_RL",
            "value": peak_stats["mean"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Mean predicted HDAC8 pIC50 at the peak-activity RL step.",
        },
        {
            "metric": "pIC50_median_peak_RL",
            "value": peak_stats["median"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Median predicted HDAC8 pIC50 at the peak-activity RL step.",
        },
        {
            "metric": "pIC50_std_peak_RL",
            "value": peak_stats["std"],
            "numerator": "",
            "denominator": "",
            "unit": "pIC50",
            "definition": "Sample standard deviation of predicted pIC50 at the peak-activity RL step.",
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    try:
        metrics_df.to_csv(result_dir / "generation_quality_metrics.csv", index=False)
        comparison_df.to_csv(result_dir / "generation_quality_comparison.csv", index=False)
    except Exception:
        pass

    return {
        "metrics_df": metrics_df,
        "comparison_df": comparison_df,
        "stats": {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_unique": n_unique,
            "n_novel": n_novel,
            "n_in_ad": n_in_ad,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "ad_proportion": ad_proportion,
            "before_step": before_step,
            "after_step": after_step,
            "peak_step": peak_step,
            "before": before_stats,
            "after": after_stats,
            "peak": peak_stats,
            "before_phase": before_metrics,
            "after_phase": post_rl_metrics,
            "peak_phase": peak_metrics,
        },
    }


def _render_generation_quality_report(summary):
    """Show publication metrics requested for the generation/RL section."""
    quality = summary.get("quality") or {}
    comparison_df = summary.get("quality_comparison_df")
    metrics_df = summary.get("quality_metrics_df")
    if (
        not quality
        and (comparison_df is None or getattr(comparison_df, "empty", True))
        and (metrics_df is None or getattr(metrics_df, "empty", True))
    ):
        return

    st.header("**Generation quality report**")
    st.caption(
        "Publication summary comparing generator step 0 (prior model, before RL) with all "
        "subsequent RL steps and the single RL step with the highest mean predicted pIC50. "
        "Novelty is measured against the experimental HDAC8 set "
        "(`datasets/HDAC8_exp_data_inchi.csv`)."
    )

    before_step = quality.get("before_step")
    after_step = quality.get("after_step")
    peak_step = quality.get("peak_step")
    if comparison_df is None or getattr(comparison_df, "empty", True):
        before_phase = quality.get("before_phase") or {}
        post_rl_phase = quality.get("after_phase") or {}
        peak_phase = quality.get("peak_phase") or {}
        comparison_df = _build_generation_comparison_table(
            before_phase,
            post_rl_phase,
            before_step,
            after_step,
            peak_phase,
            peak_step,
        )

    st.subheader("Summary comparison")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download summary comparison (CSV)",
        data=comparison_df.to_csv(index=False),
        file_name="generation_quality_comparison.csv",
        mime="text/csv",
        key="download_generation_quality_comparison",
    )

    with st.expander("Detailed metrics and definitions"):
        display_rows = []
        if metrics_df is not None and not getattr(metrics_df, "empty", True):
            for rec in metrics_df.to_dict("records"):
                unit = rec.get("unit")
                raw = rec.get("value")
                if unit == "fraction":
                    shown = _fmt_pct(raw)
                elif unit == "pIC50":
                    shown = _fmt_float(raw, 3)
                else:
                    shown = "" if raw is None or (isinstance(raw, float) and not np.isfinite(raw)) else raw
                num = rec.get("numerator")
                den = rec.get("denominator")
                count = ""
                if num not in ("", None) and den not in ("", None):
                    count = f"{num} / {den}"
                elif num not in ("", None):
                    count = str(num)
                display_rows.append(
                    {
                        "Metric": rec.get("metric"),
                        "Value": shown,
                        "Count": count,
                        "Definition": rec.get("definition"),
                    }
                )
            st.dataframe(pd.DataFrame(display_rows), hide_index=True)
            st.download_button(
                label="Download generation quality metrics (CSV)",
                data=metrics_df.to_csv(index=False),
                file_name="generation_quality_metrics.csv",
                mime="text/csv",
                key="download_generation_quality_metrics",
            )
        st.markdown(
            """
- **Pre-RL column**: generator step 0 only — sampling from the pretrained prior, before RL optimization.
- **Post-RL column**: all generator steps 1–N pooled — structures sampled while the RL agent was active.
- **Peak activity column**: the single RL step with the highest mean predicted pIC50 (step number shown in the column header).
- **Total generated**: SMILES sampled in the corresponding phase.
- **Validity**: fraction of generated SMILES parsed by RDKit.
- **Uniqueness**: fraction of valid molecules that are unique by canonical SMILES.
- **Novelty (% vs experimental HDAC8)**: fraction of unique valid molecules whose InChI is absent from
  `datasets/HDAC8_exp_data_inchi.csv`.
- **Proportion within AD (S_AD = 1)**: fraction of unique valid molecules with `in_AD >= 0.5`.
- **Mean predicted pIC50**: mean ± standard deviation over valid molecules in the phase.
            """
        )


def check_muegge_rule(mol):
    """Проверка соответствия правилам Muegge (как в анализе PAINS / structural alerts)."""
    violations = 0

    # 1. Молекулярная масса: 200-600
    mol_weight = Descriptors.MolWt(mol)
    if mol_weight < 200 or mol_weight > 600:
        violations += 1

    # 2. LogP: ≤ 5
    logp = Descriptors.MolLogP(mol)
    if logp > 5:
        violations += 1

    # 3. Количество доноров водородных связей ≤ 5
    h_bond_donors = Descriptors.NumHDonors(mol)
    if h_bond_donors > 5:
        violations += 1

    # 4. Количество акцепторов водородных связей ≤ 10
    h_bond_acceptors = Descriptors.NumHAcceptors(mol)
    if h_bond_acceptors > 10:
        violations += 1

    # 5. Количество вращаемых связей ≤ 15
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    if rotatable_bonds > 15:
        violations += 1

    # 6. TPSA ≤ 150
    tpsa = Descriptors.TPSA(mol)
    if tpsa > 150:
        violations += 1

    # 7. Количество колец ≤ 7
    num_rings = Descriptors.RingCount(mol)
    if num_rings > 7:
        violations += 1

    return "yes" if violations == 0 else f"no, {violations} violations"


def run_smiles_rnn_pipeline(
    max_molecules: int = 500,
    n_steps: int = 250,
    batch_size: int = 128,
    seed: int = 42,
    progress_callback=None,
):
    """
    Run notebook-inspired RL pipeline and return filtered generated molecules.
    """
    base_dir = Path("DataGeneration")
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "runs" / timestamp

    _prepare_smiles_rnn_assets(base_dir)
    config_path = _stage_rl_run_files(base_dir, run_dir)
    run_meta = _run_smiles_rnn_rl(
        run_dir,
        config_path,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        progress_callback=progress_callback,
    )
    result_dir = _find_smiles_rnn_result_dir(run_dir)

    if result_dir is None:
        raise RuntimeError("Could not locate SMILES-RNN result directory with scores.csv.")

    run_meta = {
        **run_meta,
        "n_steps": run_meta.get("n_steps", int(n_steps)),
        "batch_size": run_meta.get("batch_size", int(batch_size)),
        "seed": run_meta.get("seed", int(seed)),
    }

    warning_note = None
    try:
        quality = compute_generation_quality_report(result_dir)
    except Exception as exc:
        quality = {
            "metrics_df": pd.DataFrame(),
            "comparison_df": pd.DataFrame(),
            "stats": {},
        }
        warning_note = (
            f"Generation quality report could not be computed ({exc}). "
            "Candidate molecules are still returned."
        )

    df = pd.read_csv(result_dir / "scores.csv")
    initial_count = len(df)

    df_unique = df.drop_duplicates(subset=["smiles"]).dropna(subset=["smiles"]).copy()
    unique_count = len(df_unique)
    df_unique["mol"] = df_unique["smiles"].apply(Chem.MolFromSmiles)
    invalid_count = int(df_unique["mol"].isna().sum())
    df_ok = df_unique.dropna(subset=["mol"]).copy()
    df_ok["cansmi"] = df_ok["mol"].apply(Chem.MolToSmiles)
    df_ok = df_ok.drop_duplicates(subset=["cansmi"]).copy()
    valid_unique_count = len(df_ok)

    df_filtered = df_ok.copy()
    try:
        uru = importlib.import_module("useful_rdkit_utils")
        ring_system_lookup = uru.RingSystemLookup()
        df_filtered["ring_systems"] = df_filtered["mol"].apply(ring_system_lookup.process_mol)
        ring_freq = [uru.get_min_ring_frequency(x) for x in df_filtered["ring_systems"]]
        df_filtered[["min_ring", "min_freq"]] = ring_freq
        df_filtered = df_filtered.query("min_freq > 100 or min_freq < 0").copy()
    except Exception:
        skip_msg = (
            "Quality filters were skipped because `useful_rdkit_utils` is unavailable."
        )
        warning_note = f"{warning_note} {skip_msg}" if warning_note else skip_msg

    if len(df_filtered) == 0:
        _prune_datagen_runs(Path("DataGeneration") / "runs", keep=2)
        return pd.DataFrame(), {
            "initial_count": initial_count,
            "unique_count": unique_count,
            "invalid_count": invalid_count,
            "valid_unique_count": valid_unique_count,
            "post_quality_filter_count": 0,
            "inside_ad_count": 0,
            "experimental_excluded_count": 0,
            "result_dir": str(result_dir),
            "run_dir": str(run_dir),
            "device_used": run_meta.get("device"),
            "n_steps_used": quality.get("stats", {}).get("after_step", run_meta.get("n_steps", "")),
            "batch_size_used": run_meta.get("batch_size", ""),
            "seed_used": run_meta.get("seed", ""),
            "warning": warning_note,
            "quality_metrics_df": quality.get("metrics_df", pd.DataFrame()),
            "quality_comparison_df": quality.get("comparison_df", pd.DataFrame()),
            "quality": quality.get("stats", {}),
        }

    ad_candidates = [c for c in df_filtered.columns if c.endswith("_in_AD")]
    pic50_candidates = [c for c in df_filtered.columns if c.endswith("_pIC50")]
    sigma_candidates = [c for c in df_filtered.columns if c.endswith("_sigma_consensus")]
    dnn_candidates = [c for c in df_filtered.columns if c.endswith("_d_nearest")]
    if not ad_candidates or not pic50_candidates:
        raise RuntimeError(
            "RL output does not contain expected QSAR reward columns (_pIC50 and _in_AD)."
        )

    ad_col = ad_candidates[0]
    pic50_col = pic50_candidates[0]
    sigma_col = sigma_candidates[0] if sigma_candidates else None
    dnn_col = dnn_candidates[0] if dnn_candidates else None

    df_filtered[ad_col] = pd.to_numeric(df_filtered[ad_col], errors="coerce").fillna(0.0)
    df_filtered[pic50_col] = pd.to_numeric(df_filtered[pic50_col], errors="coerce").fillna(0.0)
    if sigma_col:
        df_filtered[sigma_col] = pd.to_numeric(df_filtered[sigma_col], errors="coerce")
    if dnn_col:
        df_filtered[dnn_col] = pd.to_numeric(df_filtered[dnn_col], errors="coerce")

    selected = df_filtered[df_filtered[ad_col] >= 0.5].copy()
    selected = selected.sort_values(pic50_col, ascending=False)

    experimental_excluded_count = 0
    if len(selected) > 0 and "mol" in selected.columns:
        _exp_mask = selected["mol"].apply(mol_has_experimental_hdac8_record)
        experimental_excluded_count = int(_exp_mask.sum())
        selected = selected[~_exp_mask].copy()

    if max_molecules and max_molecules > 0:
        selected = selected.head(int(max_molecules))

    try:
        sascorer_mod = _load_sascorer_module()
    except ImportError as e:
        sas_msg = f"SAScore unavailable ({e})"
        warning_note = f"{warning_note} {sas_msg}" if warning_note else sas_msg
        sascorer_mod = None

    def _sascore_for_mol(mol):
        if mol is None or sascorer_mod is None:
            return np.nan
        try:
            return float(sascorer_mod.calculateScore(mol))
        except Exception:
            return np.nan

    sas_series = selected["mol"].apply(_sascore_for_mol) if "mol" in selected.columns else pd.Series(
        np.nan, index=selected.index
    )

    def _muegge_for_mol(mol):
        if mol is None:
            return ""
        try:
            return check_muegge_rule(mol)
        except Exception:
            return "error"

    muegge_series = (
        selected["mol"].apply(_muegge_for_mol)
        if "mol" in selected.columns
        else pd.Series("", index=selected.index)
    )

    output_cols = [
        c
        for c in ["step", "smiles", "cansmi", pic50_col, sigma_col, dnn_col, ad_col]
        if c and c in selected.columns
    ]
    output_df = selected[output_cols].copy()
    rename_map = {
        "smiles": "generated_smiles",
        "cansmi": "canonical_smiles",
        pic50_col: "predicted_pIC50",
        ad_col: "in_AD",
    }
    if sigma_col:
        rename_map[sigma_col] = "sigma_consensus"
    if dnn_col:
        rename_map[dnn_col] = "d_nearest"
    output_df.rename(columns=rename_map, inplace=True)
    if "in_AD" in output_df.columns:
        ins_at = int(output_df.columns.get_loc("in_AD"))
        output_df.insert(ins_at, "SAScore", sas_series.to_numpy())
    else:
        output_df["SAScore"] = sas_series.to_numpy()

    if "in_AD" in output_df.columns:
        loc_in_ad = int(output_df.columns.get_loc("in_AD"))
        output_df.insert(loc_in_ad + 1, "Muegge rules", muegge_series.to_numpy())
    else:
        output_df["Muegge rules"] = muegge_series.to_numpy()

    output_df.insert(0, "No.", range(1, len(output_df) + 1))
    if "predicted_pIC50" in output_df.columns:
        output_df["predicted_pIC50"] = output_df["predicted_pIC50"].round(4)
    if "sigma_consensus" in output_df.columns:
        output_df["sigma_consensus"] = output_df["sigma_consensus"].round(4)
        # Manuscript-style report: pIC50 ± σ_consensus
        output_df.insert(
            int(output_df.columns.get_loc("sigma_consensus")) + 1,
            "pIC50_with_uncertainty",
            [
                f"{p:.3f} ± {s:.3f}" if pd.notna(p) and pd.notna(s) else ""
                for p, s in zip(output_df["predicted_pIC50"], output_df["sigma_consensus"])
            ],
        )
    if "d_nearest" in output_df.columns:
        output_df["d_nearest"] = output_df["d_nearest"].round(4)
    if "SAScore" in output_df.columns:
        output_df["SAScore"] = pd.to_numeric(output_df["SAScore"], errors="coerce").round(3)
    if "in_AD" in output_df.columns:
        output_df["in_AD"] = output_df["in_AD"].astype(bool)
        output_df["AD_status"] = output_df["in_AD"].map(
            lambda x: "Inside AD" if x else "Outside AD (low confidence)"
        )

    summary = {
        "initial_count": initial_count,
        "unique_count": unique_count,
        "invalid_count": invalid_count,
        "valid_unique_count": valid_unique_count,
        "post_quality_filter_count": len(df_filtered),
        "inside_ad_count": len(output_df),
        "experimental_excluded_count": experimental_excluded_count,
        "result_dir": str(result_dir),
        "run_dir": str(run_dir),
        "device_used": run_meta.get("device"),
        "n_steps_used": quality.get("stats", {}).get("after_step", run_meta.get("n_steps", "")),
        "batch_size_used": run_meta.get("batch_size", ""),
        "seed_used": run_meta.get("seed", ""),
        "warning": warning_note,
        "quality_metrics_df": quality.get("metrics_df", pd.DataFrame()),
        "quality_comparison_df": quality.get("comparison_df", pd.DataFrame()),
        "quality": quality.get("stats", {}),
    }
    _prune_datagen_runs(Path("DataGeneration") / "runs", keep=2)
    return output_df, summary

# HDAC8 consensus QSAR
@st.cache_resource
def load_consensus_model():
    """Load weighted CatBoost consensus + dual-component UQ artifacts once per session."""
    return HDAC8ConsensusModel(Path("Models"))


@st.cache_data
def load_hdac_data():
    """Кэшированная загрузка данных HDAC8"""
    df_exp = pd.read_csv('datasets/HDAC8_exp_data_inchi.csv')
    return (df_exp.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())


@st.cache_data
def experimental_hdac8_inchi_keys():
    """InChI keys из datasets/HDAC8_exp_data_inchi.csv (те же, что для проверки экспериментального pIC50)."""
    return frozenset(load_hdac_data().keys())


def mol_has_experimental_hdac8_record(mol) -> bool:
    """Совпадает ли структура с соединением из экспериментальной таблицы (есть измеренный pIC50)."""
    if mol is None:
        return False
    try:
        inchi = str(Chem.MolToInchi(mol))
    except Exception:
        return False
    return inchi in experimental_hdac8_inchi_keys()


@st.cache_data
def load_structural_alerts():
    """Кэшированная загрузка структурных алертов"""
    pains_df = pd.read_csv('datasets/PAINS.csv', sep=r'\s+')
    brenk_df = pd.read_csv('datasets/unwanted_substructures.csv', sep=r'\s+')
    tox_df = pd.read_csv('datasets/tox_alerts_list.csv', sep=r'\s+')
    vip_df = pd.read_csv('datasets/vip_substructures.csv', sep=r'\s+')
    
    return {
        'pains': [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in pains_df.iterrows()],
        'brenk': [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in brenk_df.iterrows()],
        'tox': [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in tox_df.iterrows()],
        'vip': [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in vip_df.iterrows()]
    }


@lru_cache(maxsize=1000)
def getMolDescriptors_cached(smiles, missingVal=None):
    """Кэшированная версия расчета молекулярных дескрипторов"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

# Функции для расчета различных методов LogP (как в SwissADME)
def calculate_ilogp(mol):
    """Расчет iLOGP - внутренний LogP метод"""
    try:
        # iLOGP использует внутренние дескрипторы RDKit
        return Descriptors.MolLogP(mol)
    except:
        return None

def calculate_wlogp(mol):
    """Расчет WLOGP - взвешенный LogP метод"""
    try:
        # WLOGP основан на взвешенных атомных вкладах
        # Используем более сложный расчет на основе атомных вкладов
        from rdkit.Chem import Crippen
        return Crippen.MolLogP(mol)
    except:
        return None

def calculate_mlogp(mol):
    """Расчет MLOGP - модифицированный LogP метод"""
    try:
        # MLOGP учитывает топологические индексы
        # Используем комбинацию различных дескрипторов
        logp = Descriptors.MolLogP(mol)
        # Добавляем корректировки на основе топологических индексов
        n_aromatic_rings = Descriptors.NumAromaticRings(mol)
        n_saturated_rings = Descriptors.NumSaturatedRings(mol)
        n_aliphatic_carbocycles = Descriptors.NumAliphaticCarbocycles(mol)
        
        # Простая корректировка на основе колец
        ring_correction = (n_aromatic_rings * 0.1) + (n_saturated_rings * 0.05) + (n_aliphatic_carbocycles * 0.03)
        return logp + ring_correction
    except:
        return None

def calculate_consensus_logp(mol):
    """Расчет консенсусного LogP на основе нескольких методов"""
    try:
        methods = []
        
        # Базовый LogP
        basic_logp = Descriptors.MolLogP(mol)
        if basic_logp is not None:
            methods.append(basic_logp)
        
        # Crippen LogP
        try:
            from rdkit.Chem import Crippen
            crippen_logp = Crippen.MolLogP(mol)
            if crippen_logp is not None:
                methods.append(crippen_logp)
        except:
            pass
        
        # MLOGP
        mlogp = calculate_mlogp(mol)
        if mlogp is not None:
            methods.append(mlogp)
        
        if methods:
            return sum(methods) / len(methods)
        else:
            return basic_logp
    except:
        return Descriptors.MolLogP(mol)

def get_enhanced_logp(mol, method='consensus'):
    """Получение улучшенного LogP различными методами"""
    if method == 'basic':
        return Descriptors.MolLogP(mol)
    elif method == 'crippen':
        try:
            from rdkit.Chem import Crippen
            return Crippen.MolLogP(mol)
        except:
            return Descriptors.MolLogP(mol)
    elif method == 'mlogp':
        return calculate_mlogp(mol)
    elif method == 'consensus':
        return calculate_consensus_logp(mol)
    else:
        return Descriptors.MolLogP(mol)

                

def muegge(smiles):
    mol=Chem.MolFromSmiles(smiles)
    desc_MolWt = Descriptors.MolWt(mol)
    desc_MolLogP = Descriptors.MolLogP(mol)
    
    desc_NumHDonors = Descriptors.NumHDonors(mol)
    desc_NumHAcceptors = Descriptors.NumHAcceptors(mol)
    desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
    desc_TPSA = Descriptors.TPSA(mol)
    desc_NumRings = Descriptors.RingCount(mol)

    # Нормализация для радара (масштабирование к 0-10)
    # MW: 200-600 -> 0-10 (200=0, 600=10)
    mw_normalized = max(0, min(10, (desc_MolWt - 200) / 40))
    # LogP: 0-5 -> 0-10 (0=0, 5=10)
    logp_normalized = max(0, min(10, desc_MolLogP / 0.5))
    # HBD: 0-5 -> 0-10 (0=0, 5=10)
    hbd_normalized = desc_NumHDonors * 2
    # HBA: 0-10 -> 0-10 (0=0, 10=10)
    hba_normalized = desc_NumHAcceptors
    # RotBonds: 0-15 -> 0-10 (0=0, 15=10)
    rotbonds_normalized = desc_NumRotatableBonds * 2/3
    # TPSA: 0-150 -> 0-10 (0=0, 150=10)
    tpsa_normalized = desc_TPSA / 15
    # Rings: 0-7 -> 0-10 (0=0, 7=10)
    rings_normalized = desc_NumRings * 10/7

    df = pd.DataFrame({
    'group': ['A','B'],
    'MW-200/40': [10, mw_normalized],
    'LogP/0.5': [10, logp_normalized],
    'HBD*2': [10, hbd_normalized],
    'HBA': [10, hba_normalized],
    'RotBonds*2/3': [10, rotbonds_normalized],
    'TPSA/15': [10, tpsa_normalized],
    'Rings*10/7': [10, rings_normalized]})
    categories=list(df)[1:]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
                
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5,6,7,8,9,10], ["1","2","3",'4','5','6','7','8','9','10'], color="grey", size=7)
    plt.ylim(0, 10)
                
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="The area of Muegge's rules")
    ax.fill(angles, values, 'b', alpha=0.1)

    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Values for test substance")
    ax.fill(angles, values, 'r', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Создаем таблицу дескрипторов
    descriptors = pd.DataFrame({'Molecular weight(MW), Da': [desc_MolWt, '200-600'],
                 'Octanol-water coefficient(LogP)': [desc_MolLogP, '≤5'], 
                 'Number of hydrogen bond donors (HBD)': [desc_NumHDonors, '≤5'],
                  'Number of hydrogen bond acceptors(HBAs)':[desc_NumHAcceptors, '≤10'],
                  'Number of rotatable bonds': [desc_NumRotatableBonds, '≤15'],
                  'Topological polar surface area (TPSA), Å²': [desc_TPSA, '≤150'],
                  'Number of rings': [desc_NumRings, '≤7'],
                   'Val.': ['Values for the test substance',
                   'Reference values of Muegge rules']}, index=None).set_index('Val.').T
    
    return st.pyplot(plt), st.dataframe(descriptors)





st.write("<h3 style='text-align: center; color: black;'> Step 1. Draw molecule or select input molecular files.</h3>", unsafe_allow_html=True)
files_option1 = st.selectbox('Select input method', ('Draw the molecule and click the "Apply" button','SMILES', '*CSV file containing SMILES', 'MDL multiple SD file (*.sdf)'), label_visibility='collapsed')
if files_option1 == 'Draw the molecule and click the "Apply" button':
    smiles = st_ketcher(height=400)
    st.write('''N.B. To start the step 2 (prediction), don't forget to click the "Apply" button''')
    st.write('If you want to create a new chemical structure, press the "Reset" button')
    st.write(f'The SMILES of the created  chemical: "{smiles}"')
    if len(smiles)!=0:
        # Suppress RDKit terminal messages during parse (e.g. Can't kekulize)
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        try:
            mol = try_parse_smiles_from_ketcher(smiles)
        finally:
            lg.setLevel(RDLogger.WARNING)

        if mol is None:
            st.warning(
                "**Ketcher export could not be read by RDKit.** "
                "The SMILES string from the editor sometimes fails aromatic bond assignment (kekulization), "
                "so the structure cannot be used for the next steps."
            )
            st.info(
                "**What to do:** In the menu above, select **SMILES** and paste the same structure as a SMILES string "
                "(often the same compound parses correctly when entered as text). "
                "You can copy the Ketcher string below."
            )
            st.code(smiles, language=None)
        else:
            try:
                # Same as 'SMILES' input: InChI before standardization, then molvs
                inchi = str(Chem.MolToInchi(mol))
                canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                smiles = standardize_smiles(canon_smi)
                m = Chem.MolFromSmiles(smiles)
                if m is None:
                    st.warning(
                        "**Standardization failed after Ketcher import.** "
                        "Please use the **SMILES** input method and paste your structure there."
                    )
                    st.code(Chem.MolToSmiles(mol, isomericSmiles=True), language=None)
                else:
                    im = Draw.MolToImage(m)
                    st.image(im)
            except Exception:
                st.warning(
                    "**Could not finalize the structure from Ketcher.** "
                    "Please enter the same compound using the **SMILES** input method."
                )
                st.code(smiles, language=None)
        
if files_option1 == 'SMILES':
    SMILES_input = ""
    compound_smiles = st.text_area("Enter only one structure as a SMILES", SMILES_input)
    if len(compound_smiles)!=0:
        try:
            mol = Chem.MolFromSmiles(compound_smiles)
            if mol is None:
                st.error("RDKit can't process your molecule. You might have an error in the chemical structure.")
            else:
                # Создаем InChI напрямую из молекулы (с сохранением стереохимии)
                inchi = str(Chem.MolToInchi(mol))
                # Стандартизируем для дальнейшего использования
                canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                smiles = standardize_smiles(canon_smi)
                m = Chem.MolFromSmiles(smiles)
                if m is None:
                    st.error("RDKit can't process your molecule. You might have an error in the chemical structure.")
                else:
                    im = Draw.MolToImage(m)
                    st.image(im)
        except Exception as e:
            st.error("RDKit can't process your molecule. You might have an error in the chemical structure.")

if files_option1 == '*CSV file containing SMILES':     
    # Read input
    uploaded_file = st.file_uploader('The file should contain only one column with the name "SMILES"')
    if uploaded_file is not None:
        df_ws=pd.read_csv(uploaded_file, sep=';')
        count=0
        failed_mols = []
        bad_index=[]
        index=0
        for i in df_ws.SMILES: 
            index+=1           
            try:
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(i),isomericSmiles = True)
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)             
            except:
                failed_mols.append(i)
                bad_index.append(index)
                canon_smi='wrong_smiles'
                count+=1
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)
        st.write('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        st.write(f'Original data: {len(df_ws)} molecules')
        st.write(f'Failed data: {count} molecules')

        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': bad_index, 'SMILES of wrong structure: ': failed_mols, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)


        moldf = []
        errors = []
        for i,record in enumerate(df_ws.SMILES, start=1):
            if record!='wrong_smiles':
                try:
                    mol_raw = Chem.MolFromSmiles(record, sanitize=False)
                    Chem.SanitizeMol(mol_raw)
                    # Сохраняем canonical молекулу с изомерной информацией (как в CSV базе данных)
                    canon_smi = Chem.MolToSmiles(mol_raw, isomericSmiles=True)
                    m = Chem.MolFromSmiles(canon_smi, sanitize=True)
                    moldf.append(m)
                except Exception as e:
                    failed_mols.append(record)
                    bad_index.append(i)
                    count += 1
                    errors.append(str(e))
                    st.warning(f"Failed at row {i}: {record} | {e}")
        
        st.write('Kept data: ', len(moldf), 'molecules') 
        if len(failed_mols)!=0 and len(errors)==len(failed_mols):
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': bad_index, 'SMILES of wrong structure: ': failed_mols, 'Error': errors}, index=None)
            bad_molecules = bad_molecules.set_index('No. failed molecule in original set')
            st.dataframe(bad_molecules)

# Read SDF file 
if files_option1 == 'MDL multiple SD file (*.sdf)':
    uploaded_file = st.file_uploader("Choose a SDF file")
    if uploaded_file is not None:
        st.header('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
        failed_mols = []
        all_mols =[]
        wrong_structure=[]
        wrong_smiles=[]
        bad_index=[]
        errors = []
        for i, m in enumerate(supplier):
            structure = Chem.Mol(m)
            all_mols.append(structure)
            try:
                Chem.SanitizeMol(structure)
            except Exception as e:
                failed_mols.append(m)
                wrong_smiles.append(Chem.MolToSmiles(m) if m else '')
                wrong_structure.append(str(i+1))
                bad_index.append(i)
                errors.append(str(e))
                st.warning(f"Failed SDF at idx {i+1}: {wrong_smiles[-1]} | {e}")

        
        st.write('Original data: ', len(all_mols), 'molecules')
        st.write('Failed data: ', len(failed_mols), 'molecules')
        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)

        # Standardization SDF file
        all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
        records = []
        for i in range(len(all_mols)):
            record = Chem.MolToSmiles(all_mols[i])
            records.append(record)
        
        moldf = []
        for i,record in enumerate(records, start=1):
            try:
                mol_raw = Chem.MolFromSmiles(record, sanitize=False)
                Chem.SanitizeMol(mol_raw)
                # Сохраняем canonical молекулу с изомерной информацией (как в CSV базе данных)
                canon_smi = Chem.MolToSmiles(mol_raw, isomericSmiles=True)
                m = Chem.MolFromSmiles(canon_smi, sanitize=True)
                moldf.append(m)
            except Exception as e:
                failed_mols.append(record)
                wrong_structure.append(str(i))
                wrong_smiles.append(record)
                errors.append(str(e))
                st.warning(f"Failed after SDF standardization at idx {i}: {record} | {e}")
        
        st.write('Kept data: ', len(moldf), 'molecules') 
        if len(errors)>0:
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'Error': errors}, index=None)
            bad_molecules = bad_molecules.set_index('No. failed molecule in original set')
            st.dataframe(bad_molecules)


class Models:
    def __init__(self, mol=None):
        self.mol = mol
        self.engine = load_consensus_model()
        self.res = load_hdac_data()


class one_molecules(Models):
    def seach_predic(self, inchi=None, smiles=None):
        if inchi in self.res:
            exp = round(self.res[inchi][0]["pchembl_value_mean"], 2)
            std = round(self.res[inchi][0]["pchembl_value_std"], 4)
            chembl_id = str(self.res[inchi][0]["molecule_chembl_id"])
            value_pred_act = "see experimental value"
            d_nearest = "-"
            cpd_AD_vs_act = "-"
        else:
            if self.mol is None:
                st.error("No valid molecule is available for prediction.")
                return
            rec = self.engine.predict_mols([self.mol])[0]
            if not rec.valid:
                st.error("Could not compute consensus descriptors for this molecule.")
                return
            value_pred_act = rec.format_pic50(digits=3)
            d_nearest = round(rec.distance, 4)
            cpd_AD_vs_act = rec.ad_status_label()
            exp = "-"
            std = "-"
            chembl_id = "not detected"

        st.header("**Prediction results:**")
        common_inf = pd.DataFrame(
            {
                "SMILES": smiles,
                "Predicted pIC50 ± σ_consensus": value_pred_act,
                "d_nearest (Tanimoto)": d_nearest,
                "Applicability domain_HDAC8": cpd_AD_vs_act,
                "Experimental value pIC50": exp,
                "STD": std,
                "chembl_ID": chembl_id,
            },
            index=[1],
        )
        st.dataframe(common_inf.astype(str))


class set_molecules(Models):
    def seach_predic_csv(self, moldf=None):
        if moldf is None:
            return

        total_molecules = len(moldf)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Computing consensus descriptors for {total_molecules} molecules...")
        progress_bar.progress(0.15)

        recs = self.engine.predict_mols(moldf)
        progress_bar.progress(0.75)
        status_text.text("Matching experimental HDAC8 records...")

        struct, y_pred_con_act, d_nearest, cpd_AD_vs_act, exp_act, chembl_id, number = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for count, (mol, rec) in enumerate(zip(moldf, recs), 1):
            number.append(count)
            inchi = str(Chem.MolToInchi(mol)) if mol is not None else ""
            struct.append(rec.smiles if rec.smiles else (Chem.MolToSmiles(mol) if mol is not None else ""))
            if inchi in self.res:
                exp_act.append(self.res[inchi][0]["pchembl_value_mean"])
                chembl_id.append(str(self.res[inchi][0]["molecule_chembl_id"]))
                y_pred_con_act.append("see experimental value")
                d_nearest.append("-")
                cpd_AD_vs_act.append("-")
            elif not rec.valid:
                y_pred_con_act.append("Error")
                d_nearest.append("Error")
                cpd_AD_vs_act.append("Error")
                exp_act.append("-")
                chembl_id.append("not detected")
            else:
                y_pred_con_act.append(rec.format_pic50(digits=3))
                d_nearest.append(round(rec.distance, 4))
                cpd_AD_vs_act.append(rec.ad_status_label())
                exp_act.append("-")
                chembl_id.append("not detected")

        common_inf = pd.DataFrame(
            {
                "SMILES": struct,
                "No.": number,
                "Predicted pIC50 ± σ_consensus": y_pred_con_act,
                "d_nearest (Tanimoto)": d_nearest,
                "Applicability domain_HDAC8": cpd_AD_vs_act,
                "Experimental value pIC50": exp_act,
                "chembl_ID": chembl_id,
            },
            index=None,
        )
        progress_bar.empty()
        status_text.empty()

        predictions_pred = common_inf.set_index("No.").astype(str)
        st.dataframe(predictions_pred)
        st.download_button(
            label="Download results of prediction as CSV",
            data=predictions_pred.to_csv().encode("utf-8"),
            file_name="Results.csv",
            mime="text/csv",
        )


class Med_chem_one():
    def __init__(self, propetis:str, way_exp_data:list, mol=None):
        self.propetis=propetis
        self.way_exp_data=way_exp_data
        self.mol = mol
        
        # Используем кэшированные данные для структурных алертов
        alerts_data = load_structural_alerts()
        
        if 'vip' in way_exp_data:
            self.substructure_mols = alerts_data['vip']
            self.substructures_df = None  # Не нужен для кэшированных данных
        elif 'unwanted' in way_exp_data:
            self.substructure_mols = alerts_data['brenk']
            self.substructures_df = None  # Не нужен для кэшированных данных
        elif 'PAINS' in way_exp_data:
            self.substructure_mols = alerts_data['pains']
            self.substructures_df = None  # Не нужен для кэшированных данных
        elif 'tox' in way_exp_data:
            self.substructure_mols = alerts_data['tox']
            self.substructures_df = None  # Не нужен для кэшированных данных
        else:
            # Fallback к старому методу
            self.substructures_df = pd.read_csv(self.way_exp_data, sep=r"\s+")
            self.substructure_mols = [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in self.substructures_df.iterrows()]
        if self.propetis=='structural alerts' or 'tox' in way_exp_data:
            # Creating a topological fingerprint for the original molecule
            self.mol_fp = FingerprintMols.FingerprintMol(m)
        
        # Check Muegge's drug-likeness rules for the molecule
        self.muegge_result = self.check_muegge_rule(m)
        
        # A dictionary for found substructures with their atomic indexes
        self.found_substructures = {}
        for name, substructure in self.substructure_mols:
            if substructure:
                match = m.GetSubstructMatch(substructure)
                if match:
                    self.found_substructures[name] = match
        
        
        # Checking if substructures are found
        if self.found_substructures:
            # A passage through each found substructure and a display of a molecule with isolated atoms
                for name, atoms in self.found_substructures.items():
                    st.write(f"The found {self.propetis}: {name}")
                    # Calculating the Tanimoto coefficient
                    if self.substructures_df is not None:
                        # Используем старый метод с DataFrame
                        self.substructure_mol = Chem.MolFromSmarts(self.substructures_df[self.substructures_df['name'] == name]['smarts'].values[0])
                    else:
                        # Используем кэшированные данные - находим SMARTS по имени
                        substructure_mol = None
                        for sub_name, sub_mol in self.substructure_mols:
                            if sub_name == name:
                                substructure_mol = sub_mol
                                break
                        self.substructure_mol = substructure_mol
                    
                    if (self.propetis=='structural alerts' or 'tox' in way_exp_data) and self.substructure_mol is not None:
                        self.sub_fp = FingerprintMols.FingerprintMol(self.substructure_mol)
                        self.tanimoto_similarity = DataStructs.TanimotoSimilarity(self.mol_fp, self.sub_fp)
                        st.write(f"Tanimoto coefficient: {self.tanimoto_similarity:.2f}")
                    if self.propetis=='Brenk_SA':
                        st.header('The Structural Alerts or Brenk filters [DOI:10.1002/cmdc.200700139] contain substructures with undesirable pharmacokinetics or toxicity*')
                    if self.propetis=='Pains':
                        st.header('*Filter for PAINS*')
                    if 'tox' in way_exp_data:
                        st.header('*Toxicophore Alert*')
                    # visualization of a molecule with a highlighted sub-structure
                    img = Draw.MolToImage(m, highlightAtoms=atoms, size=(300, 300))
                    st.image(img)  # Display an image
        else:
            st.write(f"The {self.propetis} are not found in the molecule.")
    
    def check_muegge_rule(self, mol):
        """Check compliance with Muegge's drug-likeness rules"""
        violations = 0
        details = []
        
        # 1. Molecular weight: 200-600
        mol_weight = Descriptors.MolWt(mol)
        if mol_weight < 200:
            violations += 1
            details.append(f"MW={mol_weight:.1f}<200")
        elif mol_weight > 600:
            violations += 1
            details.append(f"MW={mol_weight:.1f}>600")
        
        # 2. LogP: ≤ 5
        logp = Descriptors.MolLogP(mol)
        if logp > 5:
            violations += 1
            details.append(f"LogP={logp:.1f}>5")
        
        # 3. Hydrogen bond donors ≤ 5
        h_bond_donors = Descriptors.NumHDonors(mol)
        if h_bond_donors > 5:
            violations += 1
            details.append(f"HBD={h_bond_donors}>5")
        
        # 4. Hydrogen bond acceptors ≤ 10
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        if h_bond_acceptors > 10:
            violations += 1
            details.append(f"HBA={h_bond_acceptors}>10")
        
        # 5. Rotatable bonds ≤ 15
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if rotatable_bonds > 15:
            violations += 1
            details.append(f"RotBonds={rotatable_bonds}>15")
        
        # 6. TPSA ≤ 150
        tpsa = Descriptors.TPSA(mol)
        if tpsa > 150:
            violations += 1
            details.append(f"TPSA={tpsa:.1f}>150")
        
        # 7. Number of rings ≤ 7
        num_rings = Descriptors.RingCount(mol)
        if num_rings > 7:
            violations += 1
            details.append(f"Rings={num_rings}>7")
        
        if violations == 0:
            return "Compliant (0 violations)"
        else:
            return f"Non-compliant ({violations} violations: {', '.join(details)})"

st.write("<h3 style='text-align: center; color: black;'> Step 2. Select prediction of HDAC8 inhibitor activity, molecular generation, or substructural analysis</h3>", unsafe_allow_html=True)
files_option2 = st.selectbox(
    'Select prediction type',
    (
        'HDAC8',
        'Molecule generation (SMILES-RNN)',
        'Muegge rules, PAINS, Brenk structural alerts, Substructural search',
    ),
    label_visibility='collapsed'
)
if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 =='HDAC8':
    if st.button('Run predictions!'):
        # Создаем progress bar для загрузки модели
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Loading HDAC8 consensus model...')
        progress_bar.progress(0.2)
        
        HDAC8_one = one_molecules(m if 'm' in locals() else None)
        
        status_text.text('Calculating consensus descriptors (PaDEL + fingerprints)...')
        progress_bar.progress(0.4)
        
        status_text.text('Making HDAC8 activity prediction...')
        progress_bar.progress(0.6)
        
        HDAC8_one.seach_predic(inchi=inchi if 'inchi' in locals() else None, smiles=smiles if 'smiles' in locals() else None)
        
        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()

 
if (files_option1  =='*CSV file containing SMILES' or files_option1=='MDL multiple SD file (*.sdf)')  and files_option2 =='HDAC8':
    if st.button('Run predictions!'):
        # Создаем progress bar для загрузки модели
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Loading HDAC8 consensus model...')
        progress_bar.progress(0.1)
        
        HDAC8_set = set_molecules()
        
        status_text.text('Starting HDAC8 consensus predictions for multiple molecules...')
        progress_bar.progress(0.2)
        
        HDAC8_set.seach_predic_csv(moldf=moldf if 'moldf' in locals() else None)
        
        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()


if files_option2 == 'Molecule generation (SMILES-RNN)':
    st.write(
        "Run SMILES-RNN RL locally. Generation is driven solely by the HDAC8 consensus QSAR "
        "reward (weighted pIC50 + Tanimoto applicability domain); no template molecule is required. "
        "Final results include only molecules inside AD, and exclude compounds that match "
        "the experimental pIC50 dataset (HDAC8_exp_data_inchi.csv). After each run the app "
        "reports publication metrics: total generated structures, validity, uniqueness, "
        "novelty versus the experimental HDAC8 set, applicability-domain coverage, and a "
        "step 0 (prior) versus RL steps 1–N comparison table including mean predicted pIC50, "
        "plus a third column for the RL step with peak mean predicted pIC50."
    )
    st.caption(
        "Mode: Run SMILES-RNN RL now (GPU/CPU auto-selection). "
        "There is no wall-clock timeout — a full run can take several hours on CPU."
    )
    max_generated = st.number_input(
        "Maximum molecules to return",
        min_value=50,
        max_value=10000,
        value=500,
        step=50,
        help="Max AD-passing rows kept from the RL run.",
    )
    n_steps_ui = st.number_input(
        "RL n_steps",
        min_value=1,
        max_value=5000,
        value=250,
        step=10,
    )
    batch_size_ui = st.number_input(
        "RL batch_size",
        min_value=8,
        max_value=2048,
        value=128,
        step=8,
    )

    if st.button('Run generation!'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("Preparing SMILES-RNN assets...")
            progress_bar.progress(0.0)

            def _on_rl_progress(current_step, expected_steps):
                expected = max(int(expected_steps), 1)
                current = min(max(int(current_step), 0), expected)
                progress_bar.progress(current / expected)
                status_text.text(f"Running RL training: step {current}/{expected}")

            generated_df, summary = run_smiles_rnn_pipeline(
                max_molecules=max_generated,
                n_steps=n_steps_ui,
                batch_size=batch_size_ui,
                progress_callback=_on_rl_progress,
            )

            status_text.text("Rendering results...")
            progress_bar.progress(1.0)

            st.header("**Generation results:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Initial generated", summary["initial_count"])
            c2.metric("Valid unique", summary["valid_unique_count"])
            c3.metric("Inside AD (returned)", summary["inside_ad_count"])
            c4.metric(
                "Excluded (exp. dataset)",
                summary.get("experimental_excluded_count", 0),
            )
            if summary.get("device_used"):
                st.caption(f"Execution mode/device: **{summary['device_used']}**")

            if summary.get("warning"):
                st.warning(summary["warning"])

            if generated_df.empty:
                st.info("No molecules passed the selected filtering criteria.")
            else:
                st.dataframe(generated_df.set_index("No."))
                st.download_button(
                    label="Download generated molecules as CSV",
                    data=generated_df.to_csv(index=False),
                    file_name="Generated_molecules_SMILES_RNN.csv",
                    mime="text/csv",
                    key="download_generated_molecules",
                )

            _render_generation_quality_report(summary)

            with st.expander("Pipeline details"):
                st.write(f"RL n_steps: {summary['n_steps_used']}")
                st.write(f"RL batch_size: {summary['batch_size_used']}")
                st.write(f"RL seed: {summary.get('seed_used', 42)}")
                st.write(f"Run directory: `{summary['run_dir']}`")
                st.write(f"Results directory: `{summary['result_dir']}`")
                st.write(f"Unique after first deduplication: {summary['unique_count']}")
                st.write(f"Invalid SMILES removed: {summary['invalid_count']}")
                st.write(f"After quality filters: {summary['post_quality_filter_count']}")
                st.write(
                    "Excluded (match experimental pIC50 / HDAC8_exp_data_inchi.csv): "
                    f"{summary.get('experimental_excluded_count', 0)}"
                )
                st.write(
                    "Publication CSVs are also written to the results directory: "
                    "`generation_quality_comparison.csv` and `generation_quality_metrics.csv`."
                )

            progress_bar.progress(1.0)
            status_text.text("Generation completed!")
        except Exception as e:
            st.error(f"Molecule generation failed: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()



if (files_option1 == '*CSV file containing SMILES' or files_option1 == 'MDL multiple SD file (*.sdf)') \
   and files_option2 == 'Muegge rules, PAINS, Brenk structural alerts, Substructural search':
    if st.button('Run predictions!'):
        # Создаем progress bar для загрузки данных
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Используем кэшированные структурные алерты
        status_text.text('Loading structural alert databases...')
        progress_bar.progress(0.1)
        
        alerts_data = load_structural_alerts()
        pains_substructures = alerts_data['pains']
        brenk_substructures = alerts_data['brenk']
        tox_substructures = alerts_data['tox']
        
        status_text.text('Starting structural analysis...')
        progress_bar.progress(0.2)

        results = []
        
        # Используем общий progress bar для структурных алертов
        total_molecules = len(moldf)

        for idx, mol in enumerate(moldf):
            # Обновляем progress bar
            progress = 0.4 + (idx + 1) / total_molecules * 0.5  # От 40% до 90%
            progress_bar.progress(progress)
            status_text.text(f'Analyzing structural alerts: {idx + 1}/{total_molecules} molecules')
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)

            # Поиск PAINS
            pains_hits = [name for name, sub in pains_substructures if sub and mol.HasSubstructMatch(sub)]
            has_pains = bool(pains_hits)

            # Поиск Brenk
            brenk_hits = [name for name, sub in brenk_substructures if sub and mol.HasSubstructMatch(sub)]
            has_brenk = bool(brenk_hits)

            # Поиск TOX alerts
            tox_hits = [name for name, sub in tox_substructures if sub and mol.HasSubstructMatch(sub)]
            has_tox = bool(tox_hits)
            
            # Проверка правил Muegge для текущей молекулы
            muegge_result = check_muegge_rule(mol)
            


            results.append({
                'SMILES': smi,
                'Muegge rules':muegge_result,
                'PAINS': 'yes' if has_pains else 'no',
                'PAINS_names': ', '.join(pains_hits),
                'Brenk': 'yes' if has_brenk else 'no',
                'Brenk_names': ', '.join(brenk_hits),
                'Toxic_alert': 'yes' if has_tox else 'no',
                'Toxic_alert_names': ', '.join(tox_hits)
            })

        # Завершаем progress bar
        progress_bar.progress(1.0)
        status_text.text('Analysis completed!')
        
        df_result = pd.DataFrame(results, columns=[
            'SMILES',
            'Muegge rules',
            'PAINS', 'PAINS_names',
            'Brenk', 'Brenk_names',
            'Toxic_alert', 'Toxic_alert_names'
        ])
        st.dataframe(df_result)
        
        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()

        csv_data = df_result.to_csv(index=False)
        st.download_button(
            label='Download Substructural_alerts.csv',
            data=csv_data,
            file_name='Substructural_alerts.csv',
            mime='text/csv'
        )

if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 =='Muegge rules, PAINS, Brenk structural alerts, Substructural search':
    if st.button('Run predictions!'):
        # Создаем progress bar для структурных алертов одиночной молекулы
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Analyzing Muegge drug-likeness rules...')
        progress_bar.progress(0.2)
        
        # Muegge's rules
        st.header("**The Drug-likeness Radar: compliance with Muegge's rules**") 
        muegge(smiles)
        
        status_text.text('Searching for HDAC8 activity fragments...')
        progress_bar.progress(0.4)
        
        Substructural_search_one=Med_chem_one('fragments that increase the activity to inhibit HDAC8', 'datasets/vip_substructures.csv', m if 'm' in locals() else None)
        
        status_text.text('Checking Brenk filters...')
        progress_bar.progress(0.6)
        
        Brenk_SA=Med_chem_one('Brenk filter', 'datasets/unwanted_substructures.csv', m if 'm' in locals() else None)
        
        status_text.text('Checking PAINS filters...')
        progress_bar.progress(0.7)
        
        Pains=Med_chem_one('PAINS', 'datasets/PAINS.csv', m if 'm' in locals() else None)
        
        status_text.text('Checking toxicophore alerts...')
        progress_bar.progress(0.9)
        
        Toxic_alerts=Med_chem_one('toxicophore alerts', 'datasets/tox_alerts_list.csv', m if 'm' in locals() else None)
        
        # Завершаем progress bar
        progress_bar.progress(1.0)
        status_text.text('Analysis completed!')
        
        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()

st.text('© Tinkov Oleg, 2026')        