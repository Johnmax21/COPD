"""
app/management/commands/export_training_data.py
==================================================
Exports all labeled BreathRecords from the database into a CSV formatted
the same way as patient_features.csv, so you can point train_model.py at
real MQ135/MQ137 data instead of the lab dataset.

USAGE (from the folder with manage.py):
  python manage.py export_training_data

Only records where true_label is set (not blank) are included -- label
your collected breaths via /admin/ first (BreathRecord -> edit true_label).
"""

from django.core.management.base import BaseCommand
from app.models import BreathRecord
import csv


class Command(BaseCommand):
    help = "Export labeled breath records to a training CSV"

    def handle(self, *args, **options):
        records = BreathRecord.objects.exclude(true_label="").order_by("timestamp")

        if not records.exists():
            self.stdout.write(self.style.WARNING(
                "No labeled records found. Go to /admin/, open BreathRecords, "
                "and fill in true_label for each breath before exporting."
            ))
            return

        out_path = "patient_features_live.csv"
        fieldnames = [
            "patient_id", "class", "timestamp",
            "mq135_mean", "mq135_std", "mq135_min", "mq135_max", "mq135_slope",
            "mq135_peak_minus_baseline",
            "mq137_mean", "mq137_std", "mq137_min", "mq137_max", "mq137_slope",
            "mq137_peak_minus_baseline",
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, r in enumerate(records, start=1):
                mq135 = r.mq135_readings
                mq137 = r.mq137_readings

                def stats(x):
                    import numpy as np
                    x = np.array(x, dtype=float)
                    t = np.arange(len(x))
                    slope = np.polyfit(t, x, 1)[0] if len(x) > 1 else 0.0
                    baseline = float(np.mean(x[: max(1, len(x) // 20)]))
                    peak = float(np.max(x))
                    return {
                        "mean": float(np.mean(x)), "std": float(np.std(x)),
                        "min": float(np.min(x)), "max": peak, "slope": float(slope),
                        "peak_minus_baseline": peak - baseline,
                    }

                s135 = stats(mq135)
                s137 = stats(mq137)

                writer.writerow({
                    "patient_id": i,
                    "class": r.true_label,
                    "timestamp": r.timestamp.isoformat(),
                    "mq135_mean": s135["mean"], "mq135_std": s135["std"],
                    "mq135_min": s135["min"], "mq135_max": s135["max"],
                    "mq135_slope": s135["slope"],
                    "mq135_peak_minus_baseline": s135["peak_minus_baseline"],
                    "mq137_mean": s137["mean"], "mq137_std": s137["std"],
                    "mq137_min": s137["min"], "mq137_max": s137["max"],
                    "mq137_slope": s137["slope"],
                    "mq137_peak_minus_baseline": s137["peak_minus_baseline"],
                })

        self.stdout.write(self.style.SUCCESS(
            f"Exported {records.count()} labeled breaths -> {out_path}"
        ))
