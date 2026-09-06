"""HDAC8 weighted CatBoost consensus QSAR and dual-component UQ.

Late fusion of family models (ECFP4, MACCS, RDKit, PubChem, Klekota–Roth):

    y_hat = sum_f w_f * y_hat_f

Epistemic uncertainty is the weighted consensus standard deviation:

    sigma_consensus = sqrt(sum_f w_f * (y_hat_f - y_hat)^2)

Aleatoric / domain uncertainty uses concatenated SHAP–mRMR pooled binary
fingerprints (ECFP4, MACCS, PubChem, Klekota–Roth) and nearest-neighbour
Tanimoto distance D = 1 - Tc, with threshold mean(d_NN) + Z * sd(d_NN),
Z = 0.5. Predictions with D > threshold are outside the AD (low confidence).
RDKit 2D descriptors are not used in the AD metric. Families with weight 0
are skipped at prediction time.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

FAMILIES = ("ECFP4", "MACCS", "RDKit", "PubChem", "KlekotaRoth")
BINARY_FAMILIES = ("ECFP4", "MACCS", "PubChem", "KlekotaRoth")

FP_RADIUS = 2
FP_SIZE_ECFP = 2048
FP_SIZE_MACCS = 167
FP_SIZE_PUBCHEM = 881
FP_SIZE_KR = 4860
AD_Z = 0.5
DESC_SKIP = {"Ipc"}

_MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE_ECFP)


def default_models_dir() -> Path:
    return Path(__file__).resolve().parent / "Models"


def _padel_jar() -> Path:
    import padelpy

    jar = Path(padelpy.__file__).resolve().parent / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar"
    if not jar.is_file():
        raise FileNotFoundError(f"PaDEL JAR not found: {jar}")
    return jar


def _decode_java(raw: bytes) -> str:
    if not raw:
        return ""
    for enc in ("utf-8", "cp1251", "cp866"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _write_smi(path: Path, smiles: list[str], names: list[str]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for smi, name in zip(smiles, names):
            fh.write(f"{smi}\t{name}\n")


def padel_fingerprints(
    smiles: list[str],
    xml_src: Path,
    expected_size: int,
    tag: str,
    threads: int = 2,
) -> tuple[np.ndarray, list[str]]:
    if not xml_src.is_file():
        raise FileNotFoundError(f"PaDEL XML not found: {xml_src.resolve()}")
    jar = _padel_jar()
    work = Path(tempfile.mkdtemp(prefix=f"hdac8_{tag}_"))
    try:
        smi_path = work / f"{tag}.smi"
        csv_path = work / f"{tag}.csv"
        xml_path = work / xml_src.name
        shutil.copy2(xml_src, xml_path)
        names = [f"mol_{i + 1}" for i in range(len(smiles))]
        _write_smi(smi_path, smiles, names)
        cmd = [
            "java",
            "-Djava.awt.headless=true",
            "-Dfile.encoding=UTF-8",
            "-jar",
            str(jar),
            "-maxruntime",
            "-1",
            "-threads",
            str(int(threads)),
            "-descriptortypes",
            str(xml_path),
            "-detectaromaticity",
            "-dir",
            str(smi_path),
            "-file",
            str(csv_path),
            "-fingerprints",
            "-retainorder",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            raise RuntimeError(
                f"PaDEL produced no output for {tag} (exit {proc.returncode}).\n"
                f"stderr:\n{_decode_java(proc.stderr)[:2000]}\n"
                f"stdout:\n{_decode_java(proc.stdout)[:2000]}"
            )
        import pandas as pd

        desc = pd.read_csv(csv_path)
        fp_cols = [c for c in desc.columns if c != "Name"]
        if len(fp_cols) != expected_size:
            raise ValueError(f"Expected {expected_size} bits for {tag}, got {len(fp_cols)}")
        if len(desc) != len(smiles):
            raise ValueError(f"PaDEL returned {len(desc)} rows for {len(smiles)} molecules ({tag})")
        X = desc[fp_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite bits in {tag}")
        return X, list(fp_cols)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def compute_ecfp4(mols) -> tuple[np.ndarray, list[str]]:
    X = np.vstack([np.asarray(_MFPGEN.GetFingerprint(mol), dtype=np.float32) for mol in mols])
    names = [f"ECFP4::ECFP4_{i}" for i in range(X.shape[1])]
    return X, names


def compute_maccs(mols) -> tuple[np.ndarray, list[str]]:
    X = np.vstack([np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32) for mol in mols])
    names = [f"MACCS::MACCS_{i}" for i in range(X.shape[1])]
    return X, names


def compute_rdkit_named(mols, wanted: list[str]) -> np.ndarray:
    """Compute selected RDKit 2D descriptors. `wanted` uses Family::RDKit_<name> labels."""
    raw_names = []
    for label in wanted:
        name = label.split("::", 1)[-1]
        if name.startswith("RDKit_"):
            name = name[len("RDKit_") :]
        raw_names.append(name)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(raw_names)
    X = np.asarray([calc.CalcDescriptors(m) for m in mols], dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
    return X.astype(np.float32)


def mol_to_smiles(mol) -> str:
    return Chem.MolToSmiles(mol)


def tanimoto_nn_distance(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    Xq = np.asarray(X_query, dtype=np.float32)
    Xt = np.asarray(X_train, dtype=np.float32)
    intersection = Xq @ Xt.T
    q_pop = Xq.sum(axis=1, keepdims=True)
    t_pop = Xt.sum(axis=1, keepdims=True)
    union = q_pop + t_pop.T - intersection
    tc = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
    return (1.0 - tc).min(axis=1)


def compute_ad_limit(X_train: np.ndarray, z: float = AD_Z) -> float:
    Xt = np.asarray(X_train, dtype=np.float32)
    dist = 1.0 - np.divide(
        Xt @ Xt.T,
        Xt.sum(1, keepdims=True) + Xt.sum(1, keepdims=True).T - Xt @ Xt.T,
        out=np.ones((Xt.shape[0], Xt.shape[0]), dtype=np.float32),
        where=(Xt.sum(1, keepdims=True) + Xt.sum(1, keepdims=True).T - Xt @ Xt.T) > 0,
    )
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    return float(nn.mean() + z * nn.std(ddof=0))


def _pool_indices(raw_names: list[str], pool_names: list[str]) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(raw_names)}
    missing = [n for n in pool_names if n not in lookup]
    if missing:
        raise KeyError(f"Pool features not found in raw names: {missing[:5]}")
    return np.asarray([lookup[n] for n in pool_names], dtype=np.int32)


def _load_estimator(path: Path):
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as fh:
            return pickle.load(fh)


def weighted_consensus_sigma(
    family_hat: dict[str, np.ndarray],
    weights: dict,
    consensus: np.ndarray,
) -> np.ndarray:
    """Weighted std of family predictions around the consensus mean (epistemic UQ)."""
    y = np.asarray(consensus, dtype=np.float64).reshape(-1)
    var = np.zeros_like(y, dtype=np.float64)
    for fam, hat in family_hat.items():
        w = float(weights.get(fam, 0.0))
        if w <= 0.0:
            continue
        hat_arr = np.asarray(hat, dtype=np.float64).reshape(-1)
        var += w * np.square(hat_arr - y)
    return np.sqrt(var)


@dataclass
class ConsensusPrediction:
    smiles: str
    pred: float
    sigma: float
    in_ad: bool
    distance: float
    valid: bool
    family_preds: dict

    @property
    def low_confidence(self) -> bool:
        """True when the molecule falls outside the Tanimoto AD threshold."""
        return self.valid and not self.in_ad

    def format_pic50(self, digits: int = 3) -> str:
        """Report activity as pIC50 ± sigma_consensus."""
        if not self.valid:
            return ""
        return f"{self.pred:.{digits}f} ± {self.sigma:.{digits}f}"

    def ad_status_label(self) -> str:
        if not self.valid:
            return "Error"
        if self.in_ad:
            return "Inside AD"
        return "Outside AD (low confidence)"


class HDAC8ConsensusModel:
    def __init__(self, models_dir: str | Path | None = None, padel_threads: int = 2):
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.padel_threads = int(padel_threads)
        self._load()

    def _load(self) -> None:
        d = self.models_dir
        weights_path = d / "consensus_simplex_weights.json"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing consensus weights: {weights_path}")
        self.weights = json.loads(weights_path.read_text(encoding="utf-8"))
        with (d / "pool_feature_names.pkl").open("rb") as fh:
            self.pool_names = pickle.load(fh)

        self.active_families = [f for f in FAMILIES if float(self.weights.get(f, 0.0)) > 0]
        self.models: dict[str, CatBoostRegressor] = {}
        for fam in self.active_families:
            path = d / f"CatBoost_{fam}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing family model: {path}")
            self.models[fam] = _load_estimator(path)

        ad_path = d / "x_ad_tr.npz"
        meta_path = d / "ad_meta.json"
        if not ad_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"AD artifacts missing in {d}. Expected x_ad_tr.npz and ad_meta.json."
            )
        packed = np.load(ad_path)
        self.X_ad_tr = packed["X"].astype(np.float32, copy=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.ad_limit = float(meta["ad_limit"])
        self.ad_z = float(meta.get("ad_z", AD_Z))
        expected_ad_bits = sum(len(self.pool_names[f]) for f in BINARY_FAMILIES)
        if int(self.X_ad_tr.shape[1]) != expected_ad_bits:
            raise ValueError(
                f"AD matrix width {self.X_ad_tr.shape[1]} does not match pooled binary "
                f"features ({expected_ad_bits}). Rebuild x_ad_tr.npz from SHAP–mRMR pools."
            )
        self.xml_pubchem = d / "fingerprints_xml" / "PubchemFingerprinter.xml"
        self.xml_kr = d / "fingerprints_xml" / "KlekotaRothFingerprinter.xml"

    def _featurize(self, mols, smiles: list[str]) -> dict:
        # AD always needs the four binary families, including PubChem / KR.
        blocks = {}
        blocks["ECFP4"] = compute_ecfp4(mols)
        blocks["MACCS"] = compute_maccs(mols)
        if "RDKit" in self.active_families:
            blocks["RDKit"] = (
                compute_rdkit_named(mols, self.pool_names["RDKit"]),
                list(self.pool_names["RDKit"]),
            )
        X_pc, pc_names = padel_fingerprints(
            smiles, self.xml_pubchem, FP_SIZE_PUBCHEM, "pubchem", threads=self.padel_threads
        )
        blocks["PubChem"] = (X_pc, [f"PubChem::{n}" for n in pc_names])
        X_kr, kr_names = padel_fingerprints(
            smiles, self.xml_kr, FP_SIZE_KR, "klekotaroth", threads=self.padel_threads
        )
        blocks["KlekotaRoth"] = (X_kr, [f"KlekotaRoth::{n}" for n in kr_names])
        return blocks

    def _family_matrix(self, blocks: dict, family: str) -> np.ndarray:
        X, names = blocks[family]
        if family == "RDKit":
            return X
        idx = _pool_indices(names, self.pool_names[family])
        return X[:, idx]

    def _ad_matrix(self, blocks: dict) -> np.ndarray:
        """Concatenated SHAP–mRMR binary pools (same bits as consensus CatBoost inputs)."""
        parts = []
        for fam in BINARY_FAMILIES:
            X, names = blocks[fam]
            idx = _pool_indices(names, self.pool_names[fam])
            parts.append((X[:, idx] > 0).astype(np.float32))
        return np.hstack(parts)

    def predict_mols(self, mols) -> list[ConsensusPrediction]:
        valid_idx = []
        valid_mols = []
        valid_smiles = []
        all_smiles = []
        for i, mol in enumerate(mols):
            smi = mol_to_smiles(mol) if mol is not None else ""
            all_smiles.append(smi)
            if mol is None:
                continue
            valid_idx.append(i)
            valid_mols.append(mol)
            valid_smiles.append(smi)

        out = [
            ConsensusPrediction(
                smiles=smi,
                pred=0.0,
                sigma=0.0,
                in_ad=False,
                distance=float("inf"),
                valid=False,
                family_preds={},
            )
            for smi in all_smiles
        ]
        if not valid_mols:
            return out

        blocks = self._featurize(valid_mols, valid_smiles)
        family_hat = {}
        y = np.zeros(len(valid_mols), dtype=np.float64)
        for fam in self.active_families:
            Xf = self._family_matrix(blocks, fam)
            hat = np.asarray(self.models[fam].predict(Xf), dtype=np.float64).reshape(-1)
            family_hat[fam] = hat
            y += float(self.weights[fam]) * hat

        sigma = weighted_consensus_sigma(family_hat, self.weights, y)
        X_ad = self._ad_matrix(blocks)
        dist = tanimoto_nn_distance(X_ad, self.X_ad_tr)
        in_ad = dist <= self.ad_limit

        for j, i in enumerate(valid_idx):
            out[i] = ConsensusPrediction(
                smiles=valid_smiles[j],
                pred=float(y[j]),
                sigma=float(sigma[j]),
                in_ad=bool(in_ad[j]),
                distance=float(dist[j]),
                valid=True,
                family_preds={fam: float(family_hat[fam][j]) for fam in family_hat},
            )
        return out

    def predict_smiles(self, smiles_list) -> list[ConsensusPrediction]:
        mols = [Chem.MolFromSmiles(str(s)) if s else None for s in smiles_list]
        recs = self.predict_mols(mols)
        for rec, smi in zip(recs, smiles_list):
            rec.smiles = str(smi)
        return recs


def _pool_bit_index_in_raw_family(family: str, label: str) -> int:
    """Map Family::name labels to 0-based column indices in the raw family matrix."""
    raw = label.split("::", 1)[-1]
    if family == "ECFP4":
        prefix = "ECFP4_"
        if not raw.startswith(prefix):
            raise ValueError(f"Unexpected ECFP4 label: {label}")
        return int(raw[len(prefix) :])
    if family == "MACCS":
        prefix = "MACCS_"
        if not raw.startswith(prefix):
            raise ValueError(f"Unexpected MACCS label: {label}")
        return int(raw[len(prefix) :])
    if family == "PubChem":
        prefix = "PubchemFP"
        if not raw.startswith(prefix):
            raise ValueError(f"Unexpected PubChem label: {label}")
        return int(raw[len(prefix) :])
    if family == "KlekotaRoth":
        prefix = "KRFP"
        if not raw.startswith(prefix):
            raise ValueError(f"Unexpected KlekotaRoth label: {label}")
        # PaDEL KRFP names are 1-indexed.
        return int(raw[len(prefix) :]) - 1
    raise ValueError(f"Family {family} is not used in the binary AD matrix")


def subset_raw_ad_to_pools(
    X_raw: np.ndarray,
    pool_names: dict,
    z: float = AD_Z,
) -> tuple[np.ndarray, dict]:
    """Subset a full-binary AD matrix (7956 bits) to SHAP–mRMR pooled columns."""
    X_raw = np.asarray(X_raw)
    expected = FP_SIZE_ECFP + FP_SIZE_MACCS + FP_SIZE_PUBCHEM + FP_SIZE_KR
    if X_raw.shape[1] != expected:
        raise ValueError(f"Expected raw AD width {expected}, got {X_raw.shape[1]}")

    offsets = {
        "ECFP4": 0,
        "MACCS": FP_SIZE_ECFP,
        "PubChem": FP_SIZE_ECFP + FP_SIZE_MACCS,
        "KlekotaRoth": FP_SIZE_ECFP + FP_SIZE_MACCS + FP_SIZE_PUBCHEM,
    }
    sizes = {
        "ECFP4": FP_SIZE_ECFP,
        "MACCS": FP_SIZE_MACCS,
        "PubChem": FP_SIZE_PUBCHEM,
        "KlekotaRoth": FP_SIZE_KR,
    }
    parts = []
    pool_sizes = {}
    for fam in BINARY_FAMILIES:
        names = list(pool_names[fam])
        cols = []
        for label in names:
            local = _pool_bit_index_in_raw_family(fam, label)
            if local < 0 or local >= sizes[fam]:
                raise ValueError(f"Pool bit out of range for {fam}: {label} -> {local}")
            cols.append(offsets[fam] + local)
        parts.append(X_raw[:, cols])
        pool_sizes[fam] = len(names)
    X_ad = np.hstack(parts)
    X_bin = (X_ad > 0).astype(np.uint8)
    limit = compute_ad_limit(X_bin.astype(np.float32), z=z)
    meta = {
        "ad_limit": limit,
        "ad_z": float(z),
        "n_train": int(X_bin.shape[0]),
        "n_bits": int(X_bin.shape[1]),
        "binary_families": list(BINARY_FAMILIES),
        "pool_sizes": pool_sizes,
        "metric": "tanimoto_distance",
        "feature_space": "shap_mrmr_binary_pools",
    }
    return X_bin, meta


def export_ad_matrix(
    work_sdf: Path,
    pubchem_csv: Path,
    kr_csv: Path,
    out_dir: Path,
    pool_names: dict | None = None,
    z: float = AD_Z,
) -> dict:
    """Build packed AD training matrix from the modelling work set + PaDEL caches.

    If ``pool_names`` is provided (family -> list of Family::name labels), the AD
    matrix is restricted to those SHAP–mRMR pooled binary bits. Otherwise the full
    concatenated fingerprints are written (legacy).
    """
    import pandas as pd
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    mols = []
    for mol in Chem.ForwardSDMolSupplier(str(work_sdf), sanitize=False):
        if mol is None:
            raise ValueError(f"Unreadable record in {work_sdf}")
        Chem.SanitizeMol(mol)
        mols.append(mol)

    X_ecfp, ecfp_names = compute_ecfp4(mols)
    X_maccs, maccs_names = compute_maccs(mols)
    pc = pd.read_csv(pubchem_csv)
    kr = pd.read_csv(kr_csv)
    pc_cols = [c for c in pc.columns if c != "Name"]
    kr_cols = [c for c in kr.columns if c != "Name"]
    if len(pc_cols) != FP_SIZE_PUBCHEM:
        raise ValueError(f"PubChem cache has {len(pc_cols)} bits, expected {FP_SIZE_PUBCHEM}")
    if len(kr_cols) != FP_SIZE_KR:
        raise ValueError(f"KlekotaRoth cache has {len(kr_cols)} bits, expected {FP_SIZE_KR}")
    if len(pc) != len(mols) or len(kr) != len(mols):
        raise ValueError(
            f"AD row mismatch: sdf={len(mols)} pubchem={len(pc)} klekota={len(kr)}"
        )
    X_pc = pc[pc_cols].to_numpy(dtype=np.float32)
    X_kr = kr[kr_cols].to_numpy(dtype=np.float32)
    blocks = {
        "ECFP4": ((X_ecfp > 0).astype(np.uint8), list(ecfp_names)),
        "MACCS": ((X_maccs > 0).astype(np.uint8), list(maccs_names)),
        "PubChem": (
            (X_pc > 0).astype(np.uint8),
            [f"PubChem::{n}" for n in pc_cols],
        ),
        "KlekotaRoth": (
            (X_kr > 0).astype(np.uint8),
            [f"KlekotaRoth::{n}" for n in kr_cols],
        ),
    }

    if pool_names is None:
        X_ad = np.hstack([blocks[f][0] for f in BINARY_FAMILIES])
        limit = compute_ad_limit(X_ad.astype(np.float32), z=z)
        meta = {
            "ad_limit": limit,
            "ad_z": z,
            "n_train": int(X_ad.shape[0]),
            "n_bits": int(X_ad.shape[1]),
            "binary_families": list(BINARY_FAMILIES),
            "metric": "tanimoto_distance",
            "feature_space": "raw_binary_fingerprints",
            "source_sdf": str(work_sdf),
        }
    else:
        parts = []
        pool_sizes = {}
        for fam in BINARY_FAMILIES:
            X, names = blocks[fam]
            # PubChem/KR columns from CSV may lack Family:: prefix in raw names list
            # for ECFP4/MACCS compute_* already returns Family:: labels.
            lookup_names = names
            if fam in ("PubChem", "KlekotaRoth") and names and not names[0].startswith(f"{fam}::"):
                lookup_names = [f"{fam}::{n}" for n in names]
            idx = _pool_indices(lookup_names, list(pool_names[fam]))
            parts.append(X[:, idx])
            pool_sizes[fam] = len(pool_names[fam])
        X_ad = np.hstack(parts)
        limit = compute_ad_limit(X_ad.astype(np.float32), z=z)
        meta = {
            "ad_limit": limit,
            "ad_z": z,
            "n_train": int(X_ad.shape[0]),
            "n_bits": int(X_ad.shape[1]),
            "binary_families": list(BINARY_FAMILIES),
            "pool_sizes": pool_sizes,
            "metric": "tanimoto_distance",
            "feature_space": "shap_mrmr_binary_pools",
            "source_sdf": str(work_sdf),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "x_ad_tr.npz", X=X_ad)
    (out_dir / "ad_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
