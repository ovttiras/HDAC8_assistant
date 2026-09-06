from hdac8_consensus import HDAC8ConsensusModel


class HDAC8QSARReward:
    return_metrics = ["pIC50", "sigma_consensus", "d_nearest", "in_AD"]

    def __init__(self, prefix, models_dir, padel_threads=2, **kwargs):
        self.prefix = str(prefix).replace(" ", "_")
        self.model = HDAC8ConsensusModel(models_dir, padel_threads=int(padel_threads))

    def __call__(self, smiles, **kwargs):
        recs = self.model.predict_smiles(list(smiles))
        results = []
        for rec in recs:
            results.append(
                {
                    "smiles": rec.smiles,
                    f"{self.prefix}_pIC50": float(rec.pred) if rec.valid else 0.0,
                    f"{self.prefix}_sigma_consensus": float(rec.sigma) if rec.valid else 0.0,
                    f"{self.prefix}_d_nearest": float(rec.distance) if rec.valid else float("inf"),
                    f"{self.prefix}_in_AD": 1.0 if rec.valid and rec.in_ad else 0.0,
                }
            )
        return results
