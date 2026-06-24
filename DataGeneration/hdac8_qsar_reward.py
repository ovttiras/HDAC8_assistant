import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors


class HDAC8QSARReward:
    return_metrics = ["pIC50", "in_AD"]

    def __init__(
        self,
        prefix,
        model_path,
        xtr_path,
        model_ad_limit=4.13,
        nBits=1024,
        radius=2,
        n_jobs=1,
        **kwargs,
    ):
        self.prefix = prefix.replace(" ", "_")
        self.model_ad_limit = float(model_ad_limit)
        self.nBits = int(nBits)
        self.radius = int(radius)

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        x_tr = pd.read_csv(xtr_path).to_numpy()
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=1)
        self.nbrs.fit(x_tr)

    def _fp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.nBits, useFeatures=False, useChirality=False
        )
        return np.asarray(fp, dtype=float).reshape(1, -1)

    def __call__(self, smiles, **kwargs):
        results = []
        for smi in smiles:
            rec = {
                "smiles": smi,
                f"{self.prefix}_pIC50": 0.0,
                f"{self.prefix}_in_AD": 0.0,
            }
            X = self._fp(smi)
            if X is None:
                results.append(rec)
                continue

            pred = float(self.model.predict(X)[0])
            dist, _ = self.nbrs.kneighbors(X)
            in_ad = 1.0 if float(dist[0, 0]) <= self.model_ad_limit else 0.0

            rec[f"{self.prefix}_pIC50"] = pred
            rec[f"{self.prefix}_in_AD"] = in_ad
            results.append(rec)
        return results
