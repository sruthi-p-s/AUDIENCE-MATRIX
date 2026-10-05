"""AUDIENCE MATRIX - REST API & Dashboard Service
Containerized OTT Audience Segmentation & Personalization Engine.
"""

import os
import sys
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder="templates", static_folder="static")

# Environment & Path Resolutions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
CATALOG_PATH = os.environ.get("CATALOG_PATH", os.path.join(BASE_DIR, "recommendation_catalog.json"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(MODELS_DIR, "metrics.json"))
SUMMARY_PATH = os.path.join(MODELS_DIR, "training_summary.json")

SUPPORTED_GENRES = ["Action", "Comedy", "Thriller", "Drama", "Sci-Fi"]
FEATURE_COLUMNS = [
    "watch_time_hours",
    "avg_session_minutes",
    "sessions_per_week",
    "completion_rate",
    "weekend_ratio",
    "genre_diversity",
    "action_affinity",
    "comedy_affinity",
    "thriller_affinity",
    "drama_affinity",
    "sci_fi_affinity",
]

# State
model_artifacts = {
    "kmeans": None,
    "scaler": None,
    "metadata": None,
    "catalog": [],
    "loaded": False,
    "load_error": None,
}


def load_artifacts():
    """Loads persisted KMeans model, scaler, metadata, and catalog."""
    global model_artifacts
    kmeans_file = os.path.join(MODELS_DIR, "kmeans_model.joblib")
    scaler_file = os.path.join(MODELS_DIR, "scaler.joblib")
    metadata_file = os.path.join(MODELS_DIR, "cluster_metadata.joblib")

    try:
        if os.path.exists(kmeans_file) and os.path.exists(scaler_file) and os.path.exists(metadata_file):
            model_artifacts["kmeans"] = joblib.load(kmeans_file)
            model_artifacts["scaler"] = joblib.load(scaler_file)
            model_artifacts["metadata"] = joblib.load(metadata_file)
            model_artifacts["loaded"] = True
            model_artifacts["load_error"] = None
            print(f"[AUDIENCE MATRIX] Model artifacts loaded successfully from: {MODELS_DIR}")
        else:
            missing = [f for f in [kmeans_file, scaler_file, metadata_file] if not os.path.exists(f)]
            model_artifacts["loaded"] = False
            model_artifacts["load_error"] = f"Missing model artifacts: {missing}"
            print(f"[AUDIENCE MATRIX] Warning: {model_artifacts['load_error']}")
    except Exception as e:
        model_artifacts["loaded"] = False
        model_artifacts["load_error"] = str(e)
        print(f"[AUDIENCE MATRIX] Error loading models: {e}")

    try:
        if os.path.exists(CATALOG_PATH):
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                model_artifacts["catalog"] = json.load(f)
            print(f"[AUDIENCE MATRIX] Loaded {len(model_artifacts['catalog'])} catalog titles.")
        else:
            model_artifacts["catalog"] = []
            print(f"[AUDIENCE MATRIX] Catalog file not found at: {CATALOG_PATH}")
    except Exception as e:
        model_artifacts["catalog"] = []
        print(f"[AUDIENCE MATRIX] Error loading catalog: {e}")


# Initialize artifacts on startup
load_artifacts()


@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        "error": "Bad Request",
        "message": getattr(e, "description", "The request payload was invalid or malformed.")
    }), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource could not be found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method Not Allowed",
        "message": "The HTTP method is not supported for this endpoint."
    }), 405


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred while processing the request."
    }), 500


@app.route("/", methods=["GET"])
def index():
    """Serves the AUDIENCE MATRIX dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck endpoint for container orchestration and monitors."""
    if not model_artifacts["loaded"]:
        # Attempt reload if previously unready
        load_artifacts()

    status_code = 200 if model_artifacts["loaded"] else 503
    selected_k = model_artifacts["metadata"].get("selected_k") if model_artifacts["metadata"] else None

    return jsonify({
        "status": "healthy" if model_artifacts["loaded"] else "degraded",
        "model_loaded": model_artifacts["loaded"],
        "selected_k": selected_k,
        "total_catalog_titles": len(model_artifacts["catalog"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brand": "AUDIENCE MATRIX"
    }), status_code


@app.route("/api/overview", methods=["GET"])
def get_overview():
    """Provides complete dashboard metrics, segment details, and evaluator robustness."""
    if not model_artifacts["loaded"]:
        load_artifacts()

    if not model_artifacts["loaded"]:
        return jsonify({
            "error": "Service Unavailable",
            "message": "Model artifacts have not been initialized."
        }), 503

    meta = model_artifacts["metadata"]

    # Read evaluator metrics if present
    eval_metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                eval_metrics = json.load(f)
        except Exception:
            eval_metrics = None

    # Construct human-friendly segment list
    segments = []
    selected_k = meta["selected_k"]
    centroids = meta.get("cluster_centroids_unscaled", {})

    for cid in range(selected_k):
        cname = meta["cluster_names"].get(cid) or meta["cluster_names"].get(str(cid), f"Segment {cid}")
        cdesc = meta["cluster_descriptions"].get(cid) or meta["cluster_descriptions"].get(str(cid), "")
        ctraits = meta["cluster_characteristics"].get(cid) or meta["cluster_characteristics"].get(str(cid), [])
        csize = meta["cluster_sizes"].get(cid) or meta["cluster_sizes"].get(str(cid), {"count": 0, "percentage": 0})
        centroid = centroids.get(cid) or centroids.get(str(cid), {})

        segments.append({
            "segment_id": cid,
            "name": cname,
            "description": cdesc,
            "traits": ctraits,
            "user_count": csize.get("count", 0),
            "percentage": csize.get("percentage", 0),
            "centroid": centroid,
        })

    return jsonify({
        "project_name": "AUDIENCE MATRIX",
        "subtitle": "Containerized OTT Audience Segmentation & Personalization",
        "total_users": meta.get("total_users", 0),
        "selected_k": meta.get("selected_k", 0),
        "best_silhouette_score": meta.get("best_silhouette_score", 0.0),
        "inertia": meta.get("inertia", 0.0),
        "cluster_balance_ratio": meta.get("balance_ratio", 0.0),
        "k_sweep": meta.get("k_sweep", []),
        "feature_names": meta.get("feature_names", FEATURE_COLUMNS),
        "segments": segments,
        "evaluator_robustness": eval_metrics,
        "model_loaded": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns actual evaluator execution metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        except Exception as e:
            return jsonify({"error": "Failed to read metrics file", "details": str(e)}), 500
    else:
        return jsonify({
            "message": "Evaluator has not yet executed. Run evaluator service to populate metrics.",
            "status": "pending"
        }), 404


def validate_recommend_payload(data) -> tuple[bool, str]:
    """Strictly validates the POST /recommend payload."""
    if not isinstance(data, dict):
        return False, "Payload must be a valid JSON object."

    required_fields = [
        "user_id",
        "watch_time_hours",
        "avg_session_minutes",
        "sessions_per_week",
        "completion_rate",
        "weekend_ratio",
        "genre_diversity",
        "top_genres"
    ]

    for rf in required_fields:
        if rf not in data:
            return False, f"Missing required field: '{rf}'."

    # Type & Range Checks: user_id
    if not isinstance(data["user_id"], str) or not data["user_id"].strip():
        return False, "Field 'user_id' must be a non-empty string."

    # Numeric fields
    num_fields = ["watch_time_hours", "avg_session_minutes", "sessions_per_week", "completion_rate", "weekend_ratio"]
    for nf in num_fields:
        val = data[nf]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False, f"Field '{nf}' must be a numerical value (got {type(val).__name__})."

    # Genre diversity
    gd = data["genre_diversity"]
    if not isinstance(gd, int) or isinstance(gd, bool):
        return False, f"Field 'genre_diversity' must be an integer (got {type(gd).__name__})."
    if gd < 1 or gd > 5:
        return False, "Field 'genre_diversity' must be an integer between 1 and 5."

    # Specific Range Constraints
    wt = float(data["watch_time_hours"])
    if wt < 0:
        return False, "Field 'watch_time_hours' cannot be negative."
    if wt > 168:
        return False, "Field 'watch_time_hours' exceeds weekly maximum (168 hours)."

    sess_dur = float(data["avg_session_minutes"])
    if sess_dur <= 0:
        return False, "Field 'avg_session_minutes' must be strictly positive."
    if sess_dur > 720:
        return False, "Field 'avg_session_minutes' exceeds maximum threshold (720 minutes)."

    spw = float(data["sessions_per_week"])
    if spw < 0:
        return False, "Field 'sessions_per_week' cannot be negative."
    if spw > 100:
        return False, "Field 'sessions_per_week' exceeds realistic threshold (100 sessions)."

    comp = float(data["completion_rate"])
    if comp < 0.0 or comp > 1.0:
        return False, "Field 'completion_rate' must be between 0.0 and 1.0."

    wknd = float(data["weekend_ratio"])
    if wknd < 0.0 or wknd > 1.0:
        return False, "Field 'weekend_ratio' must be between 0.0 and 1.0."

    # Top Genres validation
    top_genres = data["top_genres"]
    if not isinstance(top_genres, list):
        return False, "Field 'top_genres' must be a list of genre strings."
    if len(top_genres) == 0:
        return False, "Field 'top_genres' cannot be empty. Must include at least 1 genre."
    if len(top_genres) > len(SUPPORTED_GENRES):
        return False, f"Field 'top_genres' contains more genres than supported ({len(SUPPORTED_GENRES)})."

    for g in top_genres:
        if not isinstance(g, str):
            return False, f"Invalid genre format: '{g}' must be a string."
        if g not in SUPPORTED_GENRES:
            return False, f"Invalid genre '{g}'. Supported genres are: {', '.join(SUPPORTED_GENRES)}."

    return True, ""


def rank_recommendations(user_data, segment_id, segment_centroid):
    """
    Ranks catalog titles transparently using:
    - predicted segment profile
    - user's top_genres preference
    - behavioral viewing format (session duration vs Movie/Series)
    - title popularity
    """
    catalog = model_artifacts["catalog"]
    if not catalog:
        return []

    top_genres = user_data["top_genres"]
    avg_sess = user_data["avg_session_minutes"]
    comp_rate = user_data["completion_rate"]

    scored_titles = []

    for item in catalog:
        title_genres = item.get("genres", [])
        content_type = item.get("type", "Movie")
        pop = float(item.get("popularity", 0.5))

        # 1. Genre Preference Alignment (Weight: 0.45)
        # Check intersection with top_genres
        match_count = sum(1 for g in title_genres if g in top_genres)
        if match_count > 0:
            # Primary match: first preference gets highest weight
            primary_bonus = 0.5 if (len(top_genres) > 0 and top_genres[0] in title_genres) else 0.3
            genre_score = min(1.0, primary_bonus + 0.3 * (match_count - 1))
        else:
            genre_score = 0.10

        # 2. Behavioral Format Match (Weight: 0.25)
        # Long sessions match movies or heavy series; short sessions match agile series
        if avg_sess >= 65:
            format_score = 0.95 if content_type == "Movie" else 0.85
        elif avg_sess <= 35:
            format_score = 0.95 if content_type == "Series" and "mins" in item.get("duration", "") and any(int(s) <= 30 for s in item.get("duration", "").split() if s.isdigit()) else 0.60
        else:
            format_score = 0.80

        # High completion rate boost for acclaimed narrative content
        if comp_rate >= 0.80 and ("Drama" in title_genres or "Thriller" in title_genres):
            format_score = min(1.0, format_score + 0.10)

        # 3. Segment Archetype Affinity (Weight: 0.20)
        # Check if title's genres align with centroid dominant genres
        seg_score = 0.2
        if segment_centroid:
            for tg in title_genres:
                affinity_key = f"{tg.lower().replace('-', '_')}_affinity"
                if affinity_key in segment_centroid:
                    seg_score = max(seg_score, float(segment_centroid[affinity_key]))

        # 4. Content Popularity (Weight: 0.10)
        pop_score = pop

        total_score = (0.45 * genre_score) + (0.25 * format_score) + (0.20 * seg_score) + (0.10 * pop_score)

        # Match reason
        reasons = []
        if match_count > 0:
            reasons.append(f"Matches preferred genre ({', '.join([g for g in title_genres if g in top_genres])})")
        if content_type == "Movie" and avg_sess >= 65:
            reasons.append("Optimized for long-session movie viewing")
        elif content_type == "Series" and avg_sess < 40:
            reasons.append("Paced for bite-sized episodic streaming")
        if pop >= 0.90:
            reasons.append("Trending platform title")

        scored_titles.append({
            "title": item["title"],
            "genres": item["genres"],
            "type": item["type"],
            "duration": item["duration"],
            "popularity": item["popularity"],
            "description": item.get("description", ""),
            "match_score": round(float(total_score), 3),
            "match_reason": " • ".join(reasons) if reasons else "Catalog baseline recommendation"
        })

    # Sort descending by match score and return top 5
    scored_titles.sort(key=lambda x: x["match_score"], reverse=True)
    return scored_titles[:5]


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend endpoint
    Performs real-time segmentation and personalized recommendation.
    """
    if not model_artifacts["loaded"]:
        load_artifacts()

    if not model_artifacts["loaded"]:
        return jsonify({
            "error": "Model Unavailable",
            "message": "Audience segmentation models are not ready. Please verify trainer execution."
        }), 503

    # Safely parse JSON
    try:
        if not request.is_json:
            return jsonify({
                "error": "Bad Request",
                "message": "Content-Type must be 'application/json'."
            }), 400
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({
                "error": "Bad Request",
                "message": "Malformed JSON payload provided."
            }), 400
    except Exception:
        return jsonify({
            "error": "Bad Request",
            "message": "Unable to decode JSON body."
        }), 400

    # Payload validation
    is_valid, validation_error = validate_recommend_payload(payload)
    if not is_valid:
        return jsonify({
            "error": "Validation Error",
            "message": validation_error
        }), 400

    try:
        kmeans = model_artifacts["kmeans"]
        scaler = model_artifacts["scaler"]
        metadata = model_artifacts["metadata"]

        user_id = str(payload["user_id"]).strip()
        top_genres = payload["top_genres"]

        # Derive genre affinities from top_genres
        # Primary gets 0.85, secondary 0.70, others baseline 0.15
        affinities = {}
        for g in SUPPORTED_GENRES:
            key = f"{g.lower().replace('-', '_')}_affinity"
            if g in top_genres:
                idx = top_genres.index(g)
                affinities[key] = max(0.60, 0.90 - (idx * 0.15))
            else:
                affinities[key] = 0.18

        # Build feature vector matching FEATURE_COLUMNS
        vector = [
            float(payload["watch_time_hours"]),
            float(payload["avg_session_minutes"]),
            float(payload["sessions_per_week"]),
            float(payload["completion_rate"]),
            float(payload["weekend_ratio"]),
            int(payload["genre_diversity"]),
            affinities["action_affinity"],
            affinities["comedy_affinity"],
            affinities["thriller_affinity"],
            affinities["drama_affinity"],
            affinities["sci_fi_affinity"],
        ]

        X_input = np.array([vector])
        X_scaled = scaler.transform(X_input)

        # Predict cluster
        segment_id = int(kmeans.predict(X_scaled)[0])

        # Calculate Euclidean distance to centroid in scaled feature space
        # NOTE: Explicitly NOT called a confidence score per specification
        centroid_scaled = kmeans.cluster_centers_[segment_id]
        distance_to_centroid = round(float(np.linalg.norm(X_scaled[0] - centroid_scaled)), 4)

        # Retrieve metadata
        segment_name = metadata["cluster_names"].get(segment_id) or metadata["cluster_names"].get(str(segment_id), f"Segment {segment_id}")
        segment_description = metadata["cluster_descriptions"].get(segment_id) or metadata["cluster_descriptions"].get(str(segment_id), "")
        segment_centroid = metadata.get("cluster_centroids_unscaled", {}).get(segment_id) or metadata.get("cluster_centroids_unscaled", {}).get(str(segment_id), {})

        # Personalized recommendations
        recommendations = rank_recommendations(payload, segment_id, segment_centroid)

        # Transparent explanation
        explanation = (
            f"Viewer '{user_id}' assigned to '{segment_name}' based on viewing pace "
            f"({payload['watch_time_hours']} hrs/wk, {payload['avg_session_minutes']} mins/session) "
            f"and weekend affinity ({float(payload['weekend_ratio']):.0%}). "
            f"Recommendations prioritized titles matching top genres ({', '.join(top_genres)}) "
            f"harmonized with the behavioral format preferences of the segment."
        )

        response = {
            "user_id": user_id,
            "segment_id": segment_id,
            "segment_name": segment_name,
            "segment_description": segment_description,
            "recommendations": recommendations,
            "distance_to_centroid": distance_to_centroid,
            "explanation": explanation
        }

        return jsonify(response), 200

    except Exception as e:
        # Never expose raw stack traces
        return jsonify({
            "error": "Inference Error",
            "message": "An error occurred while generating subscriber segmentation recommendations."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
