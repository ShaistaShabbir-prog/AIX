"""
Fix for issue #4: logging results module bug.
Provides a clean, working results logger for MIA experiments.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


class ResultsLogger:
    """
    Thread-safe results logger for MIA experiments.
    Saves to JSON + prints structured summary.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def log(
        self,
        experiment_name: str,
        model_type: str,
        attack_type: str,
        train_acc: float,
        val_acc: float,
        mia_accuracy: float,
        mia_auc: float | None = None,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment_name,
            "model_type": model_type,
            "attack_type": attack_type,
            "train_accuracy": round(float(train_acc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "overfitting_gap": round(float(train_acc) - float(val_acc), 4),
            "mia_accuracy": round(float(mia_accuracy), 4),
            "mia_auc": round(float(mia_auc), 4) if mia_auc is not None else None,
            "privacy_risk": self._privacy_risk(mia_accuracy),
            **(extra or {}),
        }
        self.records.append(record)
        self._print(record)
        return record

    def save(self, filename: str | None = None) -> str:
        fname = filename or f"mia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path  = os.path.join(self.output_dir, fname)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        log.info("Results saved to %s", path)
        return path

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {}
        mia_accs = [r["mia_accuracy"] for r in self.records]
        return {
            "total_experiments": len(self.records),
            "avg_mia_accuracy": round(sum(mia_accs) / len(mia_accs), 4),
            "max_mia_accuracy": max(mia_accs),
            "min_mia_accuracy": min(mia_accs),
            "high_risk_count":  sum(1 for r in self.records if r["privacy_risk"] == "HIGH"),
        }

    @staticmethod
    def _privacy_risk(mia_acc: float) -> str:
        if mia_acc >= 0.75: return "HIGH"
        if mia_acc >= 0.60: return "MEDIUM"
        return "LOW"

    @staticmethod
    def _print(r: dict) -> None:
        print(
            f"[{r['experiment']}] model={r['model_type']} attack={r['attack_type']} "
            f"train={r['train_accuracy']:.3f} val={r['val_accuracy']:.3f} "
            f"gap={r['overfitting_gap']:.3f} MIA={r['mia_accuracy']:.3f} "
            f"risk={r['privacy_risk']}"
        )


# Convenience function
def log_result(experiment: str, **kwargs) -> dict:
    logger = ResultsLogger()
    return logger.log(experiment, **kwargs)
