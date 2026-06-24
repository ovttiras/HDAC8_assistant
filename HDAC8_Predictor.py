######################
# Import libraries
######################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
import pickle
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from molvs import standardize_smiles
from math import pi
import zipfile
from pathlib import Path
from catboost import  CatBoostRegressor
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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


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
   st.text_area('Text to analyze', '''This application generates candidate HDAC8 inhibitors using a SMILES-RNN model pretrained on ChEMBL drug-like compounds. The model is optimized via reinforcement learning (REINVENT) to maximize predicted pIC50 values and ensure applicability domain compliance. An ECFP4-based CatBoost QSAR model serves as the core of the RL reward function. Generated structures undergo deduplication, validity checks, and optional ring-system filtering; they are then ranked by predicted activity and annotated with synthetic accessibility scores (SAScore) and Muegge drug-likeness criteria.''', height=350, label_visibility="hidden" )


with col2:
   st.header("Machine learning")
   st.image("figures/machine-learning.png", width=125)
   st.text_area('Text to analyze', '''This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets generated from scientific articles. The  models were developed using open-source chemical descriptors based on ECFP4, along with the gradient boosting method''', height=350, label_visibility="hidden" )


with col3:
   st.header("OECD rules")
   st.image("figures/target.png", width=125)
   st.text_area('Text to analyze', '''We follow the best practices for model development and validation recommended by guidelines of the Organization for Economic Cooperation and Development (OECD). The applicability domain (AD) of the models was calculated as Dcutoff = ⟨D⟩ + Zs, where «Z» is a similarity threshold parameter defined by a user (0.5 in this study) and «⟨D⟩» and «s» are the average and standard deviation, respectively, of all Euclidian distances in the multidimensional descriptor space between each compound and its nearest neighbors for all compounds in the training set. ''', height=350, label_visibility="hidden" )
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


def _build_smiles_rnn_config(work_dir: Path) -> Path:
    """Create per-run config with pIC50 + AD reward."""
    dst = work_dir / "smiles_rnn_config.json"

    input_dict = {
        "task": "HDAC8_pIC50_AD_reward",
        "output_dir": "./",
        "load_from_previous": False,
        "logging": False,
        "monitor_app": False,
        "diversity_filter": {"run": False},
        "scoring_functions": [
            {
                "name": "HDAC8QSARReward",
                "run": True,
                "parameters": {
                    "prefix": "hdac8",
                    "model_path": "CatBoost_MF.pkl",
                    "xtr_path": "x_tr_MF.csv",
                    "model_ad_limit": 4.13,
                    "nBits": 1024,
                    "radius": 2,
                    "n_jobs": 1,
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
    for filename in (
        "reinforcement_learning.py",
        "ChEMBL28pur.ckpt",
        "hdac8_qsar_reward.py",
        "sitecustomize.py",
    ):
        src = base_dir / filename
        if not src.exists():
            raise RuntimeError(f"Required asset is missing: {src}")
        shutil.copy2(src, run_dir / filename)

    # Register custom scoring class in copied reinforcement_learning.py.
    _ensure_hdac8_hook(run_dir / "reinforcement_learning.py")

    # Extract QSAR assets used by reward function.
    model_zip = Path("Models") / "CatBoost_MF.zip"
    xtr_zip = Path("Models") / "x_tr_MF.zip"
    if not model_zip.exists() or not xtr_zip.exists():
        raise RuntimeError("QSAR model assets are missing in Models directory.")
    with zipfile.ZipFile(model_zip) as zf:
        zf.extract("CatBoost_MF.pkl", path=run_dir)
    with zipfile.ZipFile(xtr_zip) as zf:
        zf.extract("x_tr_MF.csv", path=run_dir)

    return _build_smiles_rnn_config(run_dir)


def _run_smiles_rnn_rl(
    work_dir: Path,
    config_path: Path,
    n_steps: int = 250,
    batch_size: int = 128,
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
        start_ts = time.time()
        timeout_seconds = 3600
        last_reported_step = -1
        if progress_callback is not None:
            progress_callback(0, int(n_steps))

        while process.poll() is None:
            if time.time() - start_ts > timeout_seconds:
                process.kill()
                raise RuntimeError(f"SMILES-RNN RL run timed out after {timeout_seconds} seconds.")

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
    return {"device": device, "n_steps": int(n_steps), "batch_size": int(batch_size)}


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
        progress_callback=progress_callback,
    )
    result_dir = _find_smiles_rnn_result_dir(run_dir)

    if result_dir is None:
        raise RuntimeError("Could not locate SMILES-RNN result directory with scores.csv.")

    df = pd.read_csv(result_dir / "scores.csv")
    initial_count = len(df)

    # Same first cleanup stages as in notebook.
    df_unique = df.drop_duplicates(subset=["smiles"]).dropna(subset=["smiles"]).copy()
    unique_count = len(df_unique)
    df_unique["mol"] = df_unique["smiles"].apply(Chem.MolFromSmiles)
    invalid_count = int(df_unique["mol"].isna().sum())
    df_ok = df_unique.dropna(subset=["mol"]).copy()
    df_ok["cansmi"] = df_ok["mol"].apply(Chem.MolToSmiles)
    df_ok = df_ok.drop_duplicates(subset=["cansmi"]).copy()
    valid_unique_count = len(df_ok)

    # Optional quality stage (ring-system filter) if useful_rdkit_utils is available.
    warning_note = None
    df_filtered = df_ok.copy()
    try:
        uru = importlib.import_module("useful_rdkit_utils")

        ring_system_lookup = uru.RingSystemLookup()
        df_filtered["ring_systems"] = df_filtered["mol"].apply(ring_system_lookup.process_mol)
        ring_freq = [uru.get_min_ring_frequency(x) for x in df_filtered["ring_systems"]]
        df_filtered[["min_ring", "min_freq"]] = ring_freq
        df_filtered = df_filtered.query("min_freq > 100 or min_freq < 0").copy()
    except Exception:
        warning_note = (
            "Quality filters were skipped because `useful_rdkit_utils` is unavailable."
        )

    if len(df_filtered) == 0:
        _prune_datagen_runs(base_dir / "runs", keep=2)
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
            "device_used": run_meta.get("device", "unknown"),
            "n_steps_used": run_meta.get("n_steps", int(n_steps)),
            "batch_size_used": run_meta.get("batch_size", int(batch_size)),
            "warning": warning_note,
        }

    # AD is mandatory: keep only molecules inside AD.
    ad_candidates = [c for c in df_filtered.columns if c.endswith("_in_AD")]
    pic50_candidates = [c for c in df_filtered.columns if c.endswith("_pIC50")]
    if not ad_candidates or not pic50_candidates:
        raise RuntimeError(
            "RL output does not contain expected QSAR reward columns (_pIC50 and _in_AD)."
        )

    ad_col = ad_candidates[0]
    pic50_col = pic50_candidates[0]

    df_filtered[ad_col] = pd.to_numeric(df_filtered[ad_col], errors="coerce").fillna(0.0)
    df_filtered[pic50_col] = pd.to_numeric(df_filtered[pic50_col], errors="coerce").fillna(0.0)

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

    # SAScore (Ertl et al.) only for the final table rows, not during RL filtering.
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

    output_cols = [c for c in ["step", "smiles", "cansmi", pic50_col, ad_col] if c in selected.columns]
    output_df = selected[output_cols].copy()
    output_df.rename(
        columns={
            "smiles": "generated_smiles",
            "cansmi": "canonical_smiles",
            pic50_col: "predicted_pIC50",
            ad_col: "in_AD",
        },
        inplace=True,
    )
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
    if "SAScore" in output_df.columns:
        output_df["SAScore"] = pd.to_numeric(output_df["SAScore"], errors="coerce").round(3)
    if "in_AD" in output_df.columns:
        output_df["in_AD"] = output_df["in_AD"].astype(bool)

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
        "device_used": run_meta.get("device", "unknown"),
        "n_steps_used": run_meta.get("n_steps", int(n_steps)),
        "batch_size_used": run_meta.get("batch_size", int(batch_size)),
        "warning": warning_note,
    }
    _prune_datagen_runs(base_dir / "runs", keep=2)
    return output_df, summary

# Расчет молекулярных дескрипторов
#Расчет для HDAC8 активности 
@lru_cache(maxsize=1000)
def calcfp_cached(smiles, radius=2, nBits=1024, useFeatures=False, useChirality=False):
    """Кэшированная версия расчета Morgan fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, 
                                               useFeatures=useFeatures, useChirality=useChirality)
    fp = pd.Series(np.asarray(fp))
    fp = fp.add_prefix('Bit_')
    return fp

def calcfp(mol,funcFPInfo=dict(radius=2, nBits=1024, useFeatures=False, useChirality=False)):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
    fp = pd.Series(np.asarray(fp))
    fp = fp.add_prefix('Bit_')
    return fp

# Глобальные кэши для моделей и данных

@st.cache_data
def load_hdac_model():
    """Кэшированная загрузка модели HDAC8"""
    zf = zipfile.ZipFile('Models/CatBoost_MF.zip')
    return pickle.load(zf.open('CatBoost_MF.pkl'))


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
def load_training_data_hdac():
    """Кэшированная загрузка тренировочных данных HDAC8"""
    zf = zipfile.ZipFile('Models/x_tr_MF.zip')
    df = pd.read_csv(zf.open('x_tr_MF.csv'))
    return df.to_numpy()


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


def process_single_molecule_parallel(args):
    """Функция для параллельной обработки одной молекулы"""
    mol, activity, model, x_tr, model_AD_limit, res = args
    
    # Вычисляем дескрипторы
    if mol is not None:
        desc_ws = calcfp(mol)
        X = desc_ws.to_numpy().reshape(1, -1)
    
    inchi = str(Chem.MolToInchi(mol))
    smiles = Chem.MolToSmiles(mol)
    
    # Инициализируем переменные
    exp = "-"
    chembl_id = "not detected"
    
    # Проверяем наличие в экспериментальных данных
    if inchi in res:
        exp = res[inchi][0]['pchembl_value_mean']
        std = res[inchi][0]['pchembl_value_std']
        chembl_id = str(res[inchi][0]['molecule_chembl_id'])
        pred_value = 'see experimental value'
        ad_status = '-'
    else:
        # Оптимизированный расчет предсказания и AD
        # Используем более эффективный алгоритм для поиска ближайших соседей
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=1)
        nbrs.fit(x_tr)
        distances, indices = nbrs.kneighbors(X)
        similarity = distances[0, 0]
        ad_status = "Inside AD" if similarity <= model_AD_limit else "Outside AD"
        
        pred = model.predict(X)[0]
        pred_value = round(pred, 4)
    
    return {
        'smiles': smiles,
        'pred_value': pred_value,
        'ad_status': ad_status,
        'exp_value': exp,
        'chembl_id': chembl_id
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


class Models():
    def __init__(self, activity:str, way_exp_data:str, way_model:str, descripror_way_zip:str, 
                 descripror_way_csv:str, model_AD_limit:float, mol=None):
        self.activity = activity
        self.way_exp_data = way_exp_data
        self.way_model = way_model
        self.descripror_way_zip = descripror_way_zip
        self.descripror_way_csv = descripror_way_csv
        self.model_AD_limit = model_AD_limit
        self.mol = mol  
        
        # Используем кэшированные данные вместо повторной загрузки
        self.model = load_hdac_model()
        self.res = load_hdac_data()
        self.x_tr = load_training_data_hdac()

        # Calculate molecular descriptors
        if mol is not None:
            self.desc_ws = calcfp(mol)
            self.X = self.desc_ws.to_numpy().reshape(1, -1)
     

class one_molecules(Models):
    def seach_predic(self, inchi=None, smiles=None):
        # Инициализация переменных
        if inchi in self.res:
            #Поиск экспериментальных значений HDAC8
            exp = round(self.res[inchi][0]['pchembl_value_mean'], 2)
            std = round(self.res[inchi][0]['pchembl_value_std'], 4)
            chembl_id = str(self.res[inchi][0]['molecule_chembl_id']) 
            value_pred_act = 'see experimental value'
            cpd_AD_vs_act = '-' 
        else:
            #Прогнозирование  HDAC8
            y_pred_con_act = self.model.predict(self.X)           
            value_pred_act = round(y_pred_con_act[0], 3)            
            neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
            neighbors_k_vs_tox.sort(0)
            cpd_value_vs_tox = neighbors_k_vs_tox[0, :]
            cpd_AD_vs_act = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
            exp = "-"
            std = '-'
            chembl_id = "not detected"
                

        st.header('**Prediction results:**')             
        common_inf = pd.DataFrame({
            'SMILES': smiles,
            'Predicted value pIC50': value_pred_act,
            'Applicability domain_HDAC8': cpd_AD_vs_act,
            'Experimental value value pIC50': exp,
            'STD': std, 
            'chembl_ID': chembl_id
        }, index=[1])
        predictions_pred = common_inf.astype(str) 
        st.dataframe(predictions_pred)
          
   

class set_molecules(Models):    
    def seach_predic_csv(self, moldf=None, use_parallel=True):
        if moldf is None:
            return
        
        # Создаем progress bar для множественных молекул
        total_molecules = len(moldf)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f'Processing molecules: 0/{total_molecules}')
        
        if use_parallel and total_molecules > 5:  # Используем параллелизм для больших наборов
            status_text.text('Using parallel processing for faster calculations...')
            
            # Подготавливаем аргументы для параллельной обработки
            args_list = [(mol, self.activity, self.model, self.x_tr, self.model_AD_limit, self.res) 
                        for mol in moldf]
            
            # Используем ThreadPoolExecutor для I/O операций
            max_workers = min(mp.cpu_count(), 8)  # Ограничиваем количество потоков
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Отправляем задачи
                future_to_mol = {executor.submit(process_single_molecule_parallel, args): i 
                                for i, args in enumerate(args_list)}
                
                # Собираем результаты
                for i, future in enumerate(future_to_mol):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Обновляем progress bar
                        progress = (i + 1) / total_molecules
                        progress_bar.progress(progress)
                        status_text.text(f'Processing molecules: {i + 1}/{total_molecules}')
                        
                    except Exception as e:
                        st.error(f"Error processing molecule {i}: {e}")
                        results.append({
                            'smiles': 'Error',
                            'pred_value': 'Error',
                            'ad_status': 'Error',
                            'exp_value': 'Error',
                            'cas_id': 'Error',
                            'chembl_id': 'Error'
                        })
            
            # Формируем результаты
            struct = [r['smiles'] for r in results]
            y_pred_con_act = [r['pred_value'] for r in results]
            cpd_AD_vs_act = [r['ad_status'] for r in results]
            exp_act = [r['exp_value'] for r in results]
            chembl_id = [r['chembl_id'] for r in results]
            number = list(range(1, len(results) + 1))
            
        else:
            # Последовательная обработка для малых наборов
            exp_act = []
            std = []
            chembl_id = []
            y_pred_con_act = []
            cpd_AD_vs_act = []
            struct = []  
            number = []

            for count, m in enumerate(moldf, 1):
                # Обновляем progress bar
                progress = count / total_molecules
                progress_bar.progress(progress)
                status_text.text(f'Processing molecules: {count}/{total_molecules}')
                
                # Вычисляем дескрипторы для КАЖДОЙ молекулы
                if m is not None:
                    self.desc_ws = calcfp(m)
                    self.X = self.desc_ws.to_numpy().reshape(1, -1)
                                
                inchi = str(Chem.MolToInchi(m))
                i = Chem.MolToSmiles(m)
                struct.append(i)
                number.append(count)

                if inchi in self.res:
                    exp_act.append(self.res[inchi][0]['pchembl_value_mean'])
                    std.append(self.res[inchi][0]['pchembl_value_std'])
                    chembl_id.append(str(self.res[inchi][0]['molecule_chembl_id']))
                    y_pred_con_act.append('see experimental value')
                    cpd_AD_vs_act.append('-')
                else:
                    # Расчет предсказания и AD для каждой молекулы
                    neighbors_k_vs_act = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                    neighbors_k_vs_act.sort(0)
                    similarity_vs_act = neighbors_k_vs_act
                    cpd_value_vs_act = similarity_vs_act[0, :]
                    cpd_AD_vs_act_r = np.where(cpd_value_vs_act <= self.model_AD_limit, "Inside AD", "Outside AD")
                    
                    y_pred_act = self.model.predict(self.X)                
                    value_pred_act = round(y_pred_act[0], 4)
                        
                    y_pred_con_act.append(value_pred_act)
                    cpd_AD_vs_act.append(cpd_AD_vs_act_r[0])
                    
                    exp_act.append("-")
                    chembl_id.append("not detected")

        # visualization of the results
        common_inf = pd.DataFrame({
            'SMILES': struct,
            'No.': number,
            'Predicted value pIC50': y_pred_con_act,
            'Applicability domain_HDAC8': cpd_AD_vs_act,
            'Experimental value pIC50': exp_act,
            'chembl_ID': chembl_id
        }, index=None)

        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()
        
        predictions_pred = common_inf.set_index('No.')
        predictions_pred = predictions_pred.astype(str)
        st.dataframe(predictions_pred)

        def convert_df(df):
            return df.to_csv().encode('utf-8')  
        csv = convert_df(predictions_pred)

        st.download_button(
            label="Download results of prediction as CSV",
            data=csv,
            file_name='Results.csv',
            mime='text/csv',
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
        
        status_text.text('Loading HDAC8 model...')
        progress_bar.progress(0.2)
        
        HDAC8_one=one_molecules('HDAC8', 'datasets/HDAC8_exp_data_inchi.csv', 'Models/CatBoost_MF.pkl', 'Models/x_tr_MF.zip',
                             'x_tr_MF.csv',  4.13, m if 'm' in locals() else None)
        
        status_text.text('Calculating molecular descriptors...')
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
        
        status_text.text('Loading HDAC8 model...')
        progress_bar.progress(0.1)
        
        HDAC8_set=set_molecules('HDAC8', 'datasets/HDAC8_exp_data_inchi.csv', 'Models/CatBoost_MF.pkl', 'Models/x_tr_MF.zip',
                             'x_tr_MF.csv', 4.13, m if 'm' in locals() else None)
        
        status_text.text('Starting HDAC8 predictions for multiple molecules...')
        progress_bar.progress(0.2)
        
        HDAC8_set.seach_predic_csv(moldf=moldf if 'moldf' in locals() else None)
        
        # Очищаем progress bar
        progress_bar.empty()
        status_text.empty()


if files_option2 == 'Molecule generation (SMILES-RNN)':
    st.write(
        "Run SMILES-RNN RL locally. Generation is driven solely by the HDAC8 QSAR "
        "reward (predicted pIC50 + applicability domain); no template molecule is required. "
        "Final results include only molecules inside AD, and exclude compounds that match "
        "the experimental pIC50 dataset (HDAC8_exp_data_inchi.csv)."
    )
    st.caption("Mode: Run SMILES-RNN RL now (GPU/CPU auto-selection).")
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
                )

            with st.expander("Pipeline details"):
                st.write(f"RL n_steps: {summary['n_steps_used']}")
                st.write(f"RL batch_size: {summary['batch_size_used']}")
                st.write(f"Run directory: `{summary['run_dir']}`")
                st.write(f"Results directory: `{summary['result_dir']}`")
                st.write(f"Unique after first deduplication: {summary['unique_count']}")
                st.write(f"Invalid SMILES removed: {summary['invalid_count']}")
                st.write(f"After quality filters: {summary['post_quality_filter_count']}")
                st.write(
                    "Excluded (match experimental pIC50 / HDAC8_exp_data_inchi.csv): "
                    f"{summary.get('experimental_excluded_count', 0)}"
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