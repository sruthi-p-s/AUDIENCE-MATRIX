"""AUDIENCE MATRIX - Evaluator Service
Automated Robustness, Regression, and Edge-Case Test Suite.
Generates metrics.json from actual live execution.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
import requests

API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "5000")
API_URL = f"http://{API_HOST}:{API_PORT}"
METRICS_OUTPUT = os.environ.get("METRICS_OUTPUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "metrics.json"))


def wait_for_api_healthy(max_attempts=30, delay_seconds=2):
    """Waits for the API service to become healthy and models to load."""
    health_url = f"{API_URL}/health"
    print(f"[AUDIENCE MATRIX EVALUATOR] Connecting to API service at {health_url}...")

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(health_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded") is True:
                    print(f"[AUDIENCE MATRIX EVALUATOR] Connected successfully! Model loaded (K={data.get('selected_k')}).")
                    return True
        except requests.exceptions.RequestException:
            pass

        print(f"  Attempt {attempt}/{max_attempts}: Waiting for API /health (retry in {delay_seconds}s)...")
        time.sleep(delay_seconds)

    print("[AUDIENCE MATRIX EVALUATOR] ERROR: Timed out waiting for API service.")
    return False


def run_evaluation():
    """Executes the test suite against the live AUDIENCE MATRIX API."""
    print("=" * 65)
    print("AUDIENCE MATRIX - AUTOMATED ROBUSTNESS & VALIDATION EVALUATION")
    print("=" * 65)

    if not wait_for_api_healthy():
        sys.exit(1)

    recommend_url = f"{API_URL}/recommend"

    # 1. Representative Persona Tests
    representative_test_cases = [
        {
            "id": "high_engagement_viewer",
            "name": "High-Engagement Sci-Fi/Action Viewer",
            "payload": {
                "user_id": "eval_high_01",
                "watch_time_hours": 36.5,
                "avg_session_minutes": 88.0,
                "sessions_per_week": 8,
                "completion_rate": 0.94,
                "weekend_ratio": 0.62,
                "genre_diversity": 3,
                "top_genres": ["Sci-Fi", "Action"]
            }
        },
        {
            "id": "casual_viewer",
            "name": "Casual Quick-Bite Comedy Viewer",
            "payload": {
                "user_id": "eval_casual_02",
                "watch_time_hours": 6.2,
                "avg_session_minutes": 24.0,
                "sessions_per_week": 4,
                "completion_rate": 0.55,
                "weekend_ratio": 0.28,
                "genre_diversity": 2,
                "top_genres": ["Comedy", "Drama"]
            }
        },
        {
            "id": "genre_focused_viewer",
            "name": "Intense Thriller Purist",
            "payload": {
                "user_id": "eval_focused_03",
                "watch_time_hours": 19.0,
                "avg_session_minutes": 58.0,
                "sessions_per_week": 6,
                "completion_rate": 0.89,
                "weekend_ratio": 0.38,
                "genre_diversity": 1,
                "top_genres": ["Thriller"]
            }
        },
        {
            "id": "balanced_viewer",
            "name": "Multi-Genre Balanced Omnivore",
            "payload": {
                "user_id": "eval_balanced_04",
                "watch_time_hours": 21.0,
                "avg_session_minutes": 50.0,
                "sessions_per_week": 9,
                "completion_rate": 0.77,
                "weekend_ratio": 0.42,
                "genre_diversity": 5,
                "top_genres": ["Drama", "Action", "Comedy", "Thriller", "Sci-Fi"]
            }
        },
        {
            "id": "weekend_heavy_viewer",
            "name": "Marathon Weekend Binge Viewer",
            "payload": {
                "user_id": "eval_weekend_05",
                "watch_time_hours": 29.5,
                "avg_session_minutes": 115.0,
                "sessions_per_week": 4,
                "completion_rate": 0.88,
                "weekend_ratio": 0.85,
                "genre_diversity": 3,
                "top_genres": ["Action", "Thriller"]
            }
        }
    ]

    api_results = []
    print("\n--- Running Representative Persona Tests ---")
    for tc in representative_test_cases:
        t0 = time.perf_counter()
        resp = requests.post(recommend_url, json=tc["payload"], timeout=5)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        passed = False
        error_msg = None
        data = None

        if resp.status_code == 200:
            data = resp.json()
            required_keys = ["user_id", "segment_id", "segment_name", "segment_description", "recommendations", "distance_to_centroid", "explanation"]
            has_keys = all(k in data for k in required_keys)
            has_recs = isinstance(data.get("recommendations"), list) and len(data["recommendations"]) > 0
            has_dist = isinstance(data.get("distance_to_centroid"), (int, float)) and data["distance_to_centroid"] >= 0

            if has_keys and has_recs and has_dist:
                passed = True
            else:
                error_msg = f"Response missing fields or malformed structure. Keys present: {list(data.keys())}"
        else:
            error_msg = f"HTTP {resp.status_code}: {resp.text}"

        status_str = "PASS" if passed else "FAIL"
        seg_str = data.get("segment_name") if data else "N/A"
        dist_str = data.get("distance_to_centroid") if data else "N/A"
        recs_count = len(data.get("recommendations", [])) if data else 0

        print(f"  [{status_str}] {tc['name']} -> Segment: '{seg_str}' | Dist: {dist_str} | Recs: {recs_count} ({latency_ms}ms)")

        api_results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "passed": passed,
            "latency_ms": latency_ms,
            "status_code": resp.status_code,
            "segment_name": seg_str,
            "distance_to_centroid": dist_str,
            "recommendations_count": recs_count,
            "error": error_msg
        })

    # 2. Edge Case & Fault-Injection Tests
    edge_test_cases = [
        {
            "id": "empty_genres",
            "description": "Empty genre list (top_genres=[])",
            "payload": {
                "user_id": "edge_01",
                "watch_time_hours": 15.0,
                "avg_session_minutes": 45.0,
                "sessions_per_week": 5,
                "completion_rate": 0.8,
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": []
            },
            "expect_status": 400
        },
        {
            "id": "unseen_genre",
            "description": "Unsupported/unseen genre ('Horror', 'Documentary')",
            "payload": {
                "user_id": "edge_02",
                "watch_time_hours": 15.0,
                "avg_session_minutes": 45.0,
                "sessions_per_week": 5,
                "completion_rate": 0.8,
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": ["Horror", "Documentary"]
            },
            "expect_status": 400
        },
        {
            "id": "zero_watch_time",
            "description": "Zero watch time (boundary case)",
            "payload": {
                "user_id": "edge_03",
                "watch_time_hours": 0.0,
                "avg_session_minutes": 30.0,
                "sessions_per_week": 1,
                "completion_rate": 0.2,
                "weekend_ratio": 0.1,
                "genre_diversity": 1,
                "top_genres": ["Drama"]
            },
            "expect_status": 200
        },
        {
            "id": "very_large_values",
            "description": "Exorbitant watch time (>168 hrs/week)",
            "payload": {
                "user_id": "edge_04",
                "watch_time_hours": 999.0,
                "avg_session_minutes": 45.0,
                "sessions_per_week": 5,
                "completion_rate": 0.8,
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": ["Action"]
            },
            "expect_status": 400
        },
        {
            "id": "missing_field",
            "description": "Missing completion_rate field",
            "payload": {
                "user_id": "edge_05",
                "watch_time_hours": 15.0,
                "avg_session_minutes": 45.0,
                "sessions_per_week": 5,
                # missing completion_rate
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": ["Action"]
            },
            "expect_status": 400
        },
        {
            "id": "wrong_type",
            "description": "Wrong type for numeric field ('eighteen')",
            "payload": {
                "user_id": "edge_06",
                "watch_time_hours": "eighteen",
                "avg_session_minutes": 45.0,
                "sessions_per_week": 5,
                "completion_rate": 0.8,
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": ["Action"]
            },
            "expect_status": 400
        },
        {
            "id": "negative_value",
            "description": "Negative session duration (-45 mins)",
            "payload": {
                "user_id": "edge_07",
                "watch_time_hours": 15.0,
                "avg_session_minutes": -45.0,
                "sessions_per_week": 5,
                "completion_rate": 0.8,
                "weekend_ratio": 0.3,
                "genre_diversity": 2,
                "top_genres": ["Action"]
            },
            "expect_status": 400
        },
        {
            "id": "malformed_json",
            "description": "Corrupted non-JSON syntax payload",
            "raw_payload": "{'user_id': broken_json_no_quotes, watch_time: ...",
            "expect_status": 400
        }
    ]

    edge_results = []
    print("\n--- Running Edge Case & Fault Injection Tests ---")
    for tc in edge_test_cases:
        t0 = time.perf_counter()
        if "raw_payload" in tc:
            resp = requests.post(
                recommend_url,
                data=tc["raw_payload"],
                headers={"Content-Type": "application/json"},
                timeout=5
            )
        else:
            resp = requests.post(recommend_url, json=tc["payload"], timeout=5)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        passed = (resp.status_code == tc["expect_status"])
        status_str = "PASS" if passed else "FAIL"

        print(f"  [{status_str}] {tc['description']} (Expected: {tc['expect_status']}, Got: {resp.status_code})")

        edge_results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "expected_status": tc["expect_status"],
            "actual_status": resp.status_code,
            "passed": passed,
            "latency_ms": latency_ms,
            "response_snippet": resp.text[:120]
        })

    # Summary Statistics
    api_passed = sum(1 for r in api_results if r["passed"])
    edge_passed = sum(1 for r in edge_results if r["passed"])
    total_tests = len(api_results) + len(edge_results)
    total_passed = api_passed + edge_passed
    pass_rate = round((total_passed / total_tests) * 100, 1)

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "AUDIENCE MATRIX",
        "model_loaded": True,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "pass_rate_percent": pass_rate,
        "api_tests": {
            "total": len(api_results),
            "passed": api_passed,
            "pass_rate_percent": round((api_passed / len(api_results)) * 100, 1),
            "details": api_results
        },
        "edge_case_tests": {
            "total": len(edge_results),
            "passed": edge_passed,
            "pass_rate_percent": round((edge_passed / len(edge_results)) * 100, 1),
            "details": edge_results
        },
        "overall_status": "PASSED" if total_passed == total_tests else "DEGRADED"
    }

    # Persist metrics.json
    os.makedirs(os.path.dirname(os.path.abspath(METRICS_OUTPUT)), exist_ok=True)
    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 65)
    print(f"EVALUATION COMPLETE: {total_passed}/{total_tests} Tests Passed ({pass_rate}%)")
    print(f"  - Representative Persona Tests: {api_passed}/{len(api_results)}")
    print(f"  - Edge Case & Validation Tests: {edge_passed}/{len(edge_results)}")
    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print("=" * 65 + "\n")

    return metrics_payload


if __name__ == "__main__":
    run_evaluation()
