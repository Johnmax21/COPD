"""
app/views.py
============
Four views:
  - predict(request):       POST endpoint the ESP32 (or manual form) sends
                             breath data to. Runs the model, saves a
                             BreathRecord, returns JSON.
  - record_dose(request):   POST endpoint the ESP32 calls on each inhaler dose.
  - results(request):       GET page showing the dashboard (breath log, dose
                             log, charts).
  - manual_entry(request):  GET/POST page for typing in values by hand.

------------------------------------------------------------------------
v4 PIPELINE (from your friend): CLASS_LABELS are now confirmed
(0 = control, 1 = copd), and the 8-feature order is loaded dynamically from
training metadata (data/processed/train_selected_mq_2class.csv) instead of
being guessed -- this replaces the earlier FEATURE_ORDER placeholder and
resolves the class-meaning question that was open before.

Model file location changed too: now expected at
<project_root>/models/champion_model_mq_2class.joblib, falling back to
<project_root>/models/champion_model.joblib if the first isn't found.
Make sure the model file is moved into a "models" folder in your project
root (next to manage.py), not left loose in the root itself.

parse_payload() accepts EITHER nested JSON ({"mq135": {"baseline": ...}})
OR flat JSON ({"mq135_baseline": ...}) from the ESP32/manual form, and
tolerates both "min"/"minimum" and "max"/"maximum" key spellings -- so if
your ESP32 sketch's key names don't exactly match, this still works.

NOTE: This version does NOT go through the CSV->Parquet detour from the
previous message -- it predicts directly from the parsed values in memory.
If your friend's Parquet requirement is still in effect, let me know and
I'll layer that back in on top of this; as given, this file does in-memory
inference only.
------------------------------------------------------------------------
"""

import json
import logging
from datetime import datetime, date as date_cls, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import BreathRecord, DoseRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---- Model paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model_mq_2class.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "champion_model.joblib"

TRAIN_SELECTED_PATH = BASE_DIR / "data" / "processed" / "train_selected_mq_2class.csv"

_model = None
_feature_order = None

# Confirmed mapping (was a placeholder before -- now resolved)
CLASS_LABELS = {
    0: "control",
    1: "copd",
}


def get_model():
    """Lazy load and cache the champion model in memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Champion model artifact not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logging.info(f"Loaded champion model from {MODEL_PATH}")
    return _model


def get_feature_order():
    """
    Dynamically load the exact 8 surviving feature column names from Phase 3
    training metadata, preventing feature-order drift between training and
    inference. Falls back to an explicit hardcoded order if the metadata
    file isn't present.
    """
    global _feature_order
    if _feature_order is None:
        if TRAIN_SELECTED_PATH.exists():
            df_sample = pd.read_csv(TRAIN_SELECTED_PATH, nrows=1)
            _feature_order = [c for c in df_sample.columns if c.startswith("MQ")]
            logging.info(f"Dynamically loaded feature order from training metadata: {_feature_order}")
        else:
            _feature_order = [
                "MQ135_baseline", "MQ135_minimum", "MQ135_maximum", "MQ135_mean",
                "MQ137_baseline", "MQ137_minimum", "MQ137_maximum", "MQ137_mean",
            ]
            logging.warning(f"Metadata file not found. Using explicit feature order: {_feature_order}")
    return _feature_order


def parse_payload(body_dict: dict) -> dict:
    """
    Parse ESP32 JSON payload supporting both nested and flat JSON structures.

    Nested: {"mq135": {"baseline": ..., "minimum": ..., "maximum": ..., "mean": ..., "change": ...}, "mq137": {...}}
    Flat:   {"mq135_baseline": ..., "mq135_min": ..., "mq135_max": ..., "mq135_mean": ..., "mq135_change": ..., ...}
    (accepts "min"/"minimum" and "max"/"maximum" interchangeably)
    """
    extracted = {}

    for sensor in ["mq135", "mq137"]:
        sensor_upper = sensor.upper()

        if sensor in body_dict and isinstance(body_dict[sensor], dict):
            s_data = body_dict[sensor]
            base_val = float(s_data.get("baseline", 0.0))
            min_val = float(s_data.get("minimum", s_data.get("min", 0.0)))
            max_val = float(s_data.get("maximum", s_data.get("max", 0.0)))
            mean_val = float(s_data.get("mean", 0.0))

            if "change" in s_data and s_data["change"] is not None:
                chg_val = float(s_data["change"])
            else:
                chg_val = float(max_val - base_val)

        else:
            base_val = float(body_dict.get(f"{sensor}_baseline", 0.0))
            min_val = float(body_dict.get(f"{sensor}_minimum", body_dict.get(f"{sensor}_min", 0.0)))
            max_val = float(body_dict.get(f"{sensor}_maximum", body_dict.get(f"{sensor}_max", 0.0)))
            mean_val = float(body_dict.get(f"{sensor}_mean", 0.0))

            if f"{sensor}_change" in body_dict and body_dict[f"{sensor}_change"] is not None:
                chg_val = float(body_dict[f"{sensor}_change"])
            else:
                chg_val = float(max_val - base_val)

        extracted[f"{sensor}_baseline"] = base_val
        extracted[f"{sensor}_min"] = min_val
        extracted[f"{sensor}_max"] = max_val
        extracted[f"{sensor}_mean"] = mean_val
        extracted[f"{sensor}_change"] = chg_val

        extracted[f"{sensor_upper}_baseline"] = base_val
        extracted[f"{sensor_upper}_minimum"] = min_val
        extracted[f"{sensor_upper}_maximum"] = max_val
        extracted[f"{sensor_upper}_mean"] = mean_val

    return extracted


def run_prediction_and_save(raw_payload: dict, save_to_db: bool = True):
    """
    Shared logic used by both the ESP32/manual predict() endpoint:
    1. Parses payload to extract all 10 values (including change).
    2. Builds a single-row DataFrame with the exact 8 surviving features,
       in the exact order the model was trained on.
    3. Runs in-memory prediction and confidence computation.
    4. Optionally saves a BreathRecord to the DB.
    """
    values = parse_payload(raw_payload)
    logging.info(f"Parsed values: {values}")
    model = get_model()
    feature_order = get_feature_order()

    df_single = pd.DataFrame([values], columns=feature_order)
    logging.info(f"Model input row:\n{df_single.to_string()}")

    pred_raw = int(model.predict(df_single)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_single)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 1.0

    pred_label = CLASS_LABELS.get(pred_raw, f"CLASS_{pred_raw}")

    record = None
    if save_to_db:
        try:
            record = BreathRecord.objects.create(
                mq135_readings=[values["mq135_min"], values["mq135_max"]],
                mq137_readings=[values["mq137_min"], values["mq137_max"]],
                mq135_min=values["mq135_min"],
                mq135_max=values["mq135_max"],
                mq135_mean=values["mq135_mean"],
                mq135_baseline=values["mq135_baseline"],
                mq135_change=values["mq135_change"],
                mq135_peak_minus_baseline=values["mq135_change"],
                mq137_min=values["mq137_min"],
                mq137_max=values["mq137_max"],
                mq137_mean=values["mq137_mean"],
                mq137_baseline=values["mq137_baseline"],
                mq137_change=values["mq137_change"],
                mq137_peak_minus_baseline=values["mq137_change"],
                prediction=pred_label,
                confidence=confidence,
            )
        except Exception as db_err:
            logging.warning(f"BreathRecord DB creation skipped/failed: {db_err}")

    return pred_label, confidence, pred_raw, record, values


@csrf_exempt  # ESP32 can't send a Django CSRF token; fine for a local/trusted network
def predict(request):
    """Endpoint for ESP32 (JSON) and manual form (form-encoded) POST requests."""
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    # Try JSON first regardless of the Content-Type header -- some HTTP client
    # libraries (including ESP32's HTTPClient in some configurations) don't
    # send a header that matches exactly, which previously caused this to
    # silently fall through to an EMPTY request.POST.dict() and predict from
    # all-zero values. Trying JSON unconditionally and only falling back to
    # form data if that fails avoids that whole class of bug.
    body = None
    if request.body:
        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            body = None

    if body is None:
        body = request.POST.dict()

    if not body:
        return JsonResponse({"error": "Empty or unparseable request body"}, status=400)

    logging.info(f"predict() received body: {body}")

    try:
        pred_label, confidence, pred_raw, record, values = run_prediction_and_save(body, save_to_db=True)
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return JsonResponse({"error": f"Model prediction failed: {e}"}, status=500)

    return JsonResponse({
        "prediction": pred_label,
        "class_id": pred_raw,
        "confidence": round(confidence, 4),
        "status": "success",
        "record_id": record.id if record else None,
    })


DOSE_SPIKE_THRESHOLD = 0.30   # confidence increase from previous reading that triggers an extra dose
BASE_DOSES_PER_SLOT = 1       # doctor-prescribed: 1 dose each at morning/afternoon/night


@csrf_exempt
def record_dose(request):
    """POST endpoint the ESP32 calls every time it detects an inhaler dose."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        body = json.loads(request.body)
        dose_number = int(body["dose_number"])
        interval_seconds = int(body.get("interval_seconds", 0))
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        return JsonResponse({"error": f"bad payload: {e}"}, status=400)

    record = DoseRecord.objects.create(
        dose_number=dose_number,
        interval_seconds=interval_seconds,
    )

    return JsonResponse({"status": "logged", "record_id": record.id})


def results(request):
    """
    HTML page listing all past breath records, oldest-first internally (for
    delta calculation) but displayed newest-first, plus a confidence-over-time
    chart and a dosage recommendation per reading.

    Dosage rule (as specified): if a reading's confidence score jumps by
    DOSE_SPIKE_THRESHOLD (0.30) or more compared to the immediately preceding
    reading, recommend 2 doses for that slot instead of the normal 1 --
    a rising confidence is being used here as a proxy for worsening/more
    clearly-detected symptoms. This is a simple placeholder rule, not a
    clinically validated one -- flag this clearly in your report.

    Dose log supports filtering by date via ?date=YYYY-MM-DD (defaults to
    today), and shows the total number of doses taken on that date.
    """
    records_chrono = list(BreathRecord.objects.all().order_by("timestamp"))

    annotated = []
    prev_confidence = None
    for r in records_chrono:
        if prev_confidence is None:
            delta = 0.0
        else:
            delta = r.confidence - prev_confidence

        recommended_dose = BASE_DOSES_PER_SLOT + (1 if delta >= DOSE_SPIKE_THRESHOLD else 0)

        annotated.append({
            "record": r,
            "delta": round(delta, 3),
            "recommended_dose": recommended_dose,
            "spike": delta >= DOSE_SPIKE_THRESHOLD,
        })
        prev_confidence = r.confidence

    annotated.reverse()

    CHART_DAYS = 14
    start_day = date_cls.today() - timedelta(days=CHART_DAYS - 1)

    daily_counts = (
        DoseRecord.objects.filter(timestamp__date__gte=start_day)
        .annotate(day=TruncDate("timestamp"))
        .values("day")
        .annotate(count=Count("id"))
    )
    counts_by_day = {row["day"]: row["count"] for row in daily_counts}

    dose_chart_labels = []
    dose_chart_counts = []
    for i in range(CHART_DAYS):
        day = start_day + timedelta(days=i)
        dose_chart_labels.append(day.strftime("%d %b"))
        dose_chart_counts.append(counts_by_day.get(day, 0))

    date_param = request.GET.get("date", "")
    if date_param:
        try:
            selected_date = datetime.strptime(date_param, "%Y-%m-%d").date()
        except ValueError:
            selected_date = date_cls.today()
    else:
        selected_date = date_cls.today()

    dose_records = DoseRecord.objects.filter(timestamp__date=selected_date).order_by("-timestamp")
    total_doses_for_day = dose_records.count()

    return render(request, "results.html", {
        "annotated": annotated,
        "dose_chart_labels": json.dumps(dose_chart_labels),
        "dose_chart_counts": json.dumps(dose_chart_counts),
        "dose_records": dose_records,
        "selected_date": selected_date.strftime("%Y-%m-%d"),
        "total_doses_for_day": total_doses_for_day,
    })


# Field definitions for the manual entry form
MANUAL_FIELDS = [
    ("mq135_min", "MQ135 Minimum (V)"),
    ("mq135_max", "MQ135 Maximum (V)"),
    ("mq135_mean", "MQ135 Mean (V)"),
    ("mq135_baseline", "MQ135 Baseline (V)"),
    ("mq135_change", "MQ135 Change (V)"),
    ("mq137_min", "MQ137 Minimum (V)"),
    ("mq137_max", "MQ137 Maximum (V)"),
    ("mq137_mean", "MQ137 Mean (V)"),
    ("mq137_baseline", "MQ137 Baseline (V)"),
    ("mq137_change", "MQ137 Change (V)"),
]


def manual_entry(request):
    """
    HTML form for typing in the 10 Serial-Monitor values by hand and running
    them through the model -- useful for testing without the ESP32 connected.
    """
    result = None
    error = None

    if request.method == "POST":
        try:
            raw_values = {}
            for field_name, _ in MANUAL_FIELDS:
                raw = request.POST.get(field_name, "").strip()
                raw_values[field_name] = float(raw)

            pred_label, confidence, pred_raw, record, _ = run_prediction_and_save(raw_values, save_to_db=True)
            result = {
                "prediction": pred_label,
                "raw_class": pred_raw,
                "confidence": confidence,
                "record_id": record.id if record else None,
            }
        except (KeyError, ValueError):
            error = "Please fill in all 10 fields with valid numbers."

    return render(request, "manual_entry.html", {
        "fields": MANUAL_FIELDS,
        "result": result,
        "error": error,
    })