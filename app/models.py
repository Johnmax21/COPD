"""
app/models.py
=============
Stores one row per breath measurement: when it happened, the raw sensor
readings, the extracted features, and the model's prediction.
"""

from django.db import models


class BreathRecord(models.Model):
    # Automatically set the first time the row is created -- this is your
    # "datetime of the breath" field.
    timestamp = models.DateTimeField(auto_now_add=True)

    # Ground truth label for this breath, for building a retraining dataset.
    # Fill this in via Django admin after collecting each breath (e.g. "CONTROL",
    # "COPD", "SMOKERS"). Leave blank for breaths where you don't know/haven't
    # recorded the true class yet -- those just won't be usable for training.
    true_label = models.CharField(max_length=50, blank=True, default="")

    # Raw sensor data for this breath, stored as JSON arrays (e.g. [1820, 1875, ...])
    mq135_readings = models.JSONField()
    mq137_readings = models.JSONField()

    # Full set of summary stats per sensor, matching the ESP32's printed output
    # (Minimum / Maximum / Mean / Baseline / Change)
    mq135_min = models.FloatField(default=0.0)
    mq135_max = models.FloatField()
    mq135_mean = models.FloatField()
    mq135_baseline = models.FloatField(default=0.0)
    mq135_change = models.FloatField(default=0.0)   # = max - baseline; kept for display, not sent to model

    mq137_min = models.FloatField(default=0.0)
    mq137_max = models.FloatField()
    mq137_mean = models.FloatField()
    mq137_baseline = models.FloatField(default=0.0)
    mq137_change = models.FloatField(default=0.0)

    # Kept for backward compatibility with earlier records/queries
    mq135_peak_minus_baseline = models.FloatField()
    mq137_peak_minus_baseline = models.FloatField()

    # Model output
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()

    class Meta:
        ordering = ["-timestamp"]   # newest first

    def __str__(self):
        return f"{self.timestamp:%Y-%m-%d %H:%M:%S} - {self.prediction} ({self.confidence:.2f})"


class DoseRecord(models.Model):
    """One row per detected inhaler dose (cartridge press)."""
    timestamp = models.DateTimeField(auto_now_add=True)   # exact moment Django received it
    dose_number = models.IntegerField()                    # running count from the ESP32 (resets on reboot)
    interval_seconds = models.IntegerField(default=0)      # seconds since the previous dose (0 for the first)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"Dose {self.dose_number} at {self.timestamp:%Y-%m-%d %H:%M:%S}"