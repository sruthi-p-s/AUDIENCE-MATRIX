"""AUDIENCE MATRIX - Model Trainer
Unsupervised KMeans Clustering Pipeline for OTT Audience Segmentation.
"""

import os
import sys
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

RANDOM_SEED = 42
TOTAL_USERS = 8500
GENRES = ["Action", "Comedy", "Thriller", "Drama", "Sci-Fi"]
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


def generate_synthetic_ott_data(num_users=TOTAL_USERS, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a reproducible synthetic OTT viewer telemetry dataset.
    Note: Does NOT inject artificial cluster labels into the dataset.
    Generates realistic multivariate behavioral distributions with continuous variance.
    """
    np.random.seed(seed)
    print(f"[AUDIENCE MATRIX] Generating {num_users} synthetic OTT viewer telemetry records (seed={seed})...")

    # We synthesize heterogeneous subscriber behavior using natural behavioral archetypes with realistic noise
    # Distribution ratios for population components
    proportions = [0.24, 0.22, 0.20, 0.18, 0.16]
    counts = [int(p * num_users) for p in proportions]
    counts[-1] = num_users - sum(counts[:-1])

    data_records = []
    user_counter = 1

    # Archetype 1: Marathon Weekend Binge Viewers
    # High weekend ratio, long sessions, moderate-to-high watch time, high completion, likes Action/Thriller/Sci-Fi
    for _ in range(counts[0]):
        wt = np.random.normal(loc=26.0, scale=6.0)
        sess_dur = np.random.normal(loc=95.0, scale=20.0)
        sess_pw = np.random.normal(loc=4.5, scale=1.2)
        comp = np.random.normal(loc=0.86, scale=0.07)
        wknd = np.random.normal(loc=0.74, scale=0.10)
        div = int(np.clip(np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]), 1, 5))
        act = np.clip(np.random.normal(0.78, 0.12), 0.1, 0.99)
        com = np.clip(np.random.normal(0.28, 0.12), 0.05, 0.8)
        thr = np.clip(np.random.normal(0.82, 0.11), 0.1, 0.99)
        dra = np.clip(np.random.normal(0.42, 0.14), 0.05, 0.8)
        sci = np.clip(np.random.normal(0.72, 0.13), 0.1, 0.99)
        data_records.append([wt, sess_dur, sess_pw, comp, wknd, div, act, com, thr, dra, sci])

    # Archetype 2: Casual Daily Mobile & Comedy Streamers
    # Short sessions, moderate frequency, lower watch time, high comedy/drama, lower completion
    for _ in range(counts[1]):
        wt = np.random.normal(loc=6.8, scale=2.5)
        sess_dur = np.random.normal(loc=26.0, scale=7.0)
        sess_pw = np.random.normal(loc=4.2, scale=1.5)
        comp = np.random.normal(loc=0.58, scale=0.12)
        wknd = np.random.normal(loc=0.32, scale=0.11)
        div = int(np.clip(np.random.choice([1, 2, 3], p=[0.4, 0.45, 0.15]), 1, 5))
        act = np.clip(np.random.normal(0.25, 0.12), 0.05, 0.7)
        com = np.clip(np.random.normal(0.85, 0.10), 0.2, 0.99)
        thr = np.clip(np.random.normal(0.22, 0.11), 0.05, 0.6)
        dra = np.clip(np.random.normal(0.68, 0.13), 0.1, 0.95)
        sci = np.clip(np.random.normal(0.20, 0.10), 0.05, 0.6)
        data_records.append([wt, sess_dur, sess_pw, comp, wknd, div, act, com, thr, dra, sci])

    # Archetype 3: Prime-Time Narrative Drama Enthusiasts
    # Balanced watch time, steady weekday/evening sessions, high completion on story-driven series
    for _ in range(counts[2]):
        wt = np.random.normal(loc=17.5, scale=4.0)
        sess_dur = np.random.normal(loc=55.0, scale=10.0)
        sess_pw = np.random.normal(loc=5.8, scale=1.2)
        comp = np.random.normal(loc=0.88, scale=0.06)
        wknd = np.random.normal(loc=0.36, scale=0.09)
        div = int(np.clip(np.random.choice([2, 3, 4], p=[0.25, 0.55, 0.20]), 1, 5))
        act = np.clip(np.random.normal(0.35, 0.12), 0.05, 0.8)
        com = np.clip(np.random.normal(0.40, 0.14), 0.05, 0.8)
        thr = np.clip(np.random.normal(0.55, 0.14), 0.1, 0.9)
        dra = np.clip(np.random.normal(0.90, 0.08), 0.4, 0.99)
        sci = np.clip(np.random.normal(0.30, 0.12), 0.05, 0.8)
        data_records.append([wt, sess_dur, sess_pw, comp, wknd, div, act, com, thr, dra, sci])

    # Archetype 4: Speculative Sci-Fi & Action Devotees
    # High watch time, long sessions, high sci-fi and action, high completion rate
    for _ in range(counts[3]):
        wt = np.random.normal(loc=23.0, scale=5.5)
        sess_dur = np.random.normal(loc=72.0, scale=15.0)
        sess_pw = np.random.normal(loc=6.2, scale=1.4)
        comp = np.random.normal(loc=0.84, scale=0.08)
        wknd = np.random.normal(loc=0.48, scale=0.12)
        div = int(np.clip(np.random.choice([2, 3], p=[0.55, 0.45]), 1, 5))
        act = np.clip(np.random.normal(0.86, 0.09), 0.3, 0.99)
        com = np.clip(np.random.normal(0.22, 0.10), 0.05, 0.7)
        thr = np.clip(np.random.normal(0.60, 0.14), 0.1, 0.9)
        dra = np.clip(np.random.normal(0.30, 0.12), 0.05, 0.8)
        sci = np.clip(np.random.normal(0.92, 0.07), 0.4, 0.99)
        data_records.append([wt, sess_dur, sess_pw, comp, wknd, div, act, com, thr, dra, sci])

    # Archetype 5: High-Frequency Eclectic Omnivores
    # High frequency, varied sessions, highest diversity, balanced genres across the spectrum
    for _ in range(counts[4]):
        wt = np.random.normal(loc=19.5, scale=5.0)
        sess_dur = np.random.normal(loc=48.0, scale=12.0)
        sess_pw = np.random.normal(loc=8.4, scale=1.8)
        comp = np.random.normal(loc=0.74, scale=0.10)
        wknd = np.random.normal(loc=0.42, scale=0.12)
        div = int(np.clip(np.random.choice([4, 5], p=[0.5, 0.5]), 1, 5))
        act = np.clip(np.random.normal(0.65, 0.14), 0.15, 0.95)
        com = np.clip(np.random.normal(0.62, 0.14), 0.15, 0.95)
        thr = np.clip(np.random.normal(0.64, 0.13), 0.15, 0.95)
        dra = np.clip(np.random.normal(0.68, 0.12), 0.15, 0.95)
        sci = np.clip(np.random.normal(0.58, 0.14), 0.15, 0.95)
        data_records.append([wt, sess_dur, sess_pw, comp, wknd, div, act, com, thr, dra, sci])

    df = pd.DataFrame(data_records, columns=FEATURE_COLUMNS)

    # Shuffle the dataset so rows are thoroughly randomized
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Post-processing and physical clipping
    df["watch_time_hours"] = df["watch_time_hours"].clip(1.0, 75.0).round(2)
    df["avg_session_minutes"] = df["avg_session_minutes"].clip(10.0, 180.0).round(1)
    df["sessions_per_week"] = df["sessions_per_week"].clip(1.0, 30.0).round(1)
    df["completion_rate"] = df["completion_rate"].clip(0.10, 0.99).round(3)
    df["weekend_ratio"] = df["weekend_ratio"].clip(0.05, 0.95).round(3)
    df["genre_diversity"] = df["genre_diversity"].astype(int).clip(1, 5)

    for g in ["action_affinity", "comedy_affinity", "thriller_affinity", "drama_affinity", "sci_fi_affinity"]:
        df[g] = df[g].clip(0.05, 0.99).round(3)

    # Insert user_id as the first column - NO cluster labels!
    user_ids = [f"USR_{i+1:05d}" for i in range(len(df))]
    df.insert(0, "user_id", user_ids)

    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validates data integrity, nulls, and schema constraints."""
    print("[AUDIENCE MATRIX] Validating dataset integrity...")
    assert len(df) >= 5000, f"Expected >=5000 users, got {len(df)}"
    assert df["user_id"].nunique() == len(df), "user_id contains duplicates!"
    assert df[FEATURE_COLUMNS].isnull().sum().sum() == 0, "Missing values detected!"
    assert (df["completion_rate"] >= 0).all() and (df["completion_rate"] <= 1).all()
    assert (df["weekend_ratio"] >= 0).all() and (df["weekend_ratio"] <= 1).all()
    print(f"[AUDIENCE MATRIX] Validation passed successfully ({len(df)} rows, {len(df.columns)} columns).")
    return True


def name_clusters(centroids_df: pd.DataFrame, overall_means: pd.Series) -> tuple:
    """
    Generates human-readable segment names, descriptions, and characteristic bullets
    based on actual centroid traits compared to the overall population averages.
    """
    cluster_names = {}
    cluster_descriptions = {}
    cluster_characteristics = {}

    for cluster_id, row in centroids_df.iterrows():
        cid = int(cluster_id)
        # Trait comparisons
        high_weekend = row["weekend_ratio"] > overall_means["weekend_ratio"] * 1.15
        high_session = row["avg_session_minutes"] > overall_means["avg_session_minutes"] * 1.15
        short_session = row["avg_session_minutes"] < overall_means["avg_session_minutes"] * 0.75
        high_frequency = row["sessions_per_week"] > overall_means["sessions_per_week"] * 1.2
        high_completion = row["completion_rate"] > overall_means["completion_rate"] * 1.05
        high_diversity = row["genre_diversity"] >= 4

        # Affinity dominance
        genre_affinities = {
            "Action": float(row["action_affinity"]),
            "Comedy": float(row["comedy_affinity"]),
            "Thriller": float(row["thriller_affinity"]),
            "Drama": float(row["drama_affinity"]),
            "Sci-Fi": float(row["sci_fi_affinity"]),
        }
        sorted_genres = sorted(genre_affinities.items(), key=lambda x: x[1], reverse=True)
        top_genre, top_val = sorted_genres[0]
        second_genre, sec_val = sorted_genres[1]

        # Semantic Rule Derivation
        if high_weekend and high_session:
            name = "Weekend Marathon Streamers"
            desc = "Long-duration weekend marathon viewers favoring immersive storylines, high completion, and high-intensity genres."
            traits = [
                f"Peak weekend ratio ({row['weekend_ratio']:.0%})",
                f"Extended avg session duration ({row['avg_session_minutes']:.0f} mins)",
                f"Top affinity for {top_genre} and {second_genre}",
                f"High completion fidelity ({row['completion_rate']:.0%})",
            ]
        elif short_session and top_genre == "Comedy":
            name = "Casual Quick-Bite Streamers"
            desc = "Bite-sized mobile & snackable session viewers with low watch commitment and strong inclination toward light entertainment and comedy."
            traits = [
                f"Short agile viewing sessions ({row['avg_session_minutes']:.0f} mins)",
                f"High preference for Comedy ({row['comedy_affinity']:.2f}) and light formats",
                f"Moderate completion rate ({row['completion_rate']:.0%})",
                "Predominantly weekday commuter / evening viewing",
            ]
        elif top_genre == "Drama" and ((top_val - sec_val > 0.15) or high_completion):
            name = "Primetime Narrative Devotees"
            desc = "Dedicated serialized drama and story-driven enthusiasts characterized by high completion rates and consistent evening sessions."
            traits = [
                f"Heavy Drama affinity ({row['drama_affinity']:.2f})",
                f"High completion loyalty ({row['completion_rate']:.0%})",
                f"Consistent weekly pacing ({row['sessions_per_week']:.1f} sessions/week)",
                f"Evening focus with moderate weekend ratio ({row['weekend_ratio']:.0%})",
            ]
        elif top_genre in ["Sci-Fi", "Action"] and second_genre in ["Sci-Fi", "Action", "Thriller"]:
            name = "Sci-Fi & Action Purists"
            desc = "High-engagement cinephiles focused heavily on speculative science fiction, action set-pieces, and suspense thrillers."
            traits = [
                f"Dominant affinities: {top_genre} ({top_val:.2f}) & {second_genre} ({sec_val:.2f})",
                f"Substantial total watch time ({row['watch_time_hours']:.1f} hrs/week)",
                f"High engagement sessions ({row['avg_session_minutes']:.0f} mins)",
                "Low affinity for light comedy or domestic drama",
            ]
        elif high_diversity or high_frequency:
            name = "Eclectic Multi-Genre Omnivores"
            desc = "High-frequency streaming enthusiasts who consume across all genre verticals with high weekly sessions and broad appetite."
            traits = [
                f"High genre diversity score ({row['genre_diversity']:.1f} of 5 genres)",
                f"High frequency streaming ({row['sessions_per_week']:.1f} sessions/week)",
                f"Balanced multi-genre profile across {top_genre}, {second_genre}, and others",
                "Resilient viewing habits spanning weekdays and weekends",
            ]
        else:
            name = f"{top_genre} & {second_genre} Explorers"
            desc = f"Balanced subscriber cohort with primary engagement in {top_genre} and {second_genre} content."
            traits = [
                f"Primary genre: {top_genre} ({top_val:.2f})",
                f"Secondary genre: {second_genre} ({sec_val:.2f})",
                f"Average session duration: {row['avg_session_minutes']:.0f} mins",
                f"Weekly watch time: {row['watch_time_hours']:.1f} hrs",
            ]

        cluster_names[cid] = name
        cluster_descriptions[cid] = desc
        cluster_characteristics[cid] = traits

    return cluster_names, cluster_descriptions, cluster_characteristics


def train_pipeline(data_path: str, models_dir: str):
    """Executes the complete unsupervised KMeans pipeline and artifact persistence."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # 1. Dataset Generation
    df = generate_synthetic_ott_data(num_users=TOTAL_USERS, seed=RANDOM_SEED)
    df.to_csv(data_path, index=False)
    print(f"[AUDIENCE MATRIX] Saved audience telemetry to: {data_path}")

    # 2. Validation
    validate_dataset(df)

    # 3. Preprocessing
    X = df[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. K-Sweep Evaluation (K=2 through K=8)
    print("\n[AUDIENCE MATRIX] Evaluating KMeans clustering across K=2 to K=8...")
    k_range = range(2, 9)
    k_metrics = []
    best_k = 5
    best_score = -1.0
    models_by_k = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)
        inertia = float(kmeans.inertia_)
        sil_score = float(silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=RANDOM_SEED))

        # Check cluster balance
        counts = pd.Series(labels).value_counts().to_dict()
        min_size = min(counts.values())
        max_size = max(counts.values())
        balance_ratio = float(min_size / max_size)

        k_metrics.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4),
            "cluster_sizes": [int(counts.get(i, 0)) for i in range(k)],
            "balance_ratio": round(balance_ratio, 3),
        })

        models_by_k[k] = (kmeans, labels, sil_score, inertia, balance_ratio)
        print(f"  -> K={k} | Inertia={inertia:10.2f} | Silhouette={sil_score:.4f} | Min/Max Balance={balance_ratio:.2f}")

    # 5. K Selection (Silhouette score + cluster balance + interpretability)
    # Filter for candidates with good cluster balance (> 0.25)
    valid_candidates = [m for m in k_metrics if m["balance_ratio"] >= 0.25]
    if valid_candidates:
        # Choose candidate maximizing silhouette score with penalty for fragmentation
        best_candidate = max(valid_candidates, key=lambda x: x["silhouette_score"])
        selected_k = best_candidate["k"]
    else:
        selected_k = 5

    print(f"\n[AUDIENCE MATRIX] Selected Optimal K: {selected_k} (Silhouette: {models_by_k[selected_k][2]:.4f})")

    final_kmeans, final_labels, final_sil, final_inertia, final_balance = models_by_k[selected_k]

    # Calculate centroids in original unscaled domain
    unscaled_centroids = scaler.inverse_transform(final_kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(unscaled_centroids, columns=FEATURE_COLUMNS)
    overall_means = df[FEATURE_COLUMNS].mean()

    # 6. Generate Human-Readable Segment Names
    cluster_names, cluster_descriptions, cluster_characteristics = name_clusters(centroids_df, overall_means)

    # Calculate cluster sizes and percentages
    label_counts = pd.Series(final_labels).value_counts().sort_index()
    cluster_sizes = {}
    for cid in range(selected_k):
        cnt = int(label_counts.get(cid, 0))
        cluster_sizes[cid] = {
            "count": cnt,
            "percentage": round(cnt / len(df) * 100, 2),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("AUDIENCE MATRIX - DISCOVERED SEGMENTS:")
    print("=" * 60)
    for cid in range(selected_k):
        cname = cluster_names[cid]
        cstats = cluster_sizes[cid]
        print(f"Segment {cid}: {cname}")
        print(f"  Audience Share: {cstats['count']} users ({cstats['percentage']}%)")
        print(f"  Description: {cluster_descriptions[cid]}")
        for t in cluster_characteristics[cid]:
            print(f"    - {t}")
        print("-" * 60)

    # 7. Persistence
    cluster_metadata = {
        "selected_k": selected_k,
        "feature_names": FEATURE_COLUMNS,
        "cluster_names": cluster_names,
        "cluster_descriptions": cluster_descriptions,
        "cluster_characteristics": cluster_characteristics,
        "cluster_centroids_unscaled": centroids_df.to_dict(orient="index"),
        "cluster_sizes": cluster_sizes,
        "total_users": len(df),
        "best_silhouette_score": round(final_sil, 4),
        "inertia": round(final_inertia, 2),
        "balance_ratio": round(final_balance, 3),
        "k_sweep": k_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Joblib artifacts
    kmeans_file = os.path.join(models_dir, "kmeans_model.joblib")
    scaler_file = os.path.join(models_dir, "scaler.joblib")
    metadata_file = os.path.join(models_dir, "cluster_metadata.joblib")
    summary_json_file = os.path.join(models_dir, "training_summary.json")

    joblib.dump(final_kmeans, kmeans_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(cluster_metadata, metadata_file)

    with open(summary_json_file, "w") as f:
        json.dump(cluster_metadata, f, indent=2)

    print(f"\n[AUDIENCE MATRIX] Artifacts successfully persisted:")
    print(f"  - Model:     {kmeans_file}")
    print(f"  - Scaler:    {scaler_file}")
    print(f"  - Metadata:  {metadata_file}")
    print(f"  - Summary:   {summary_json_file}")
    print("[AUDIENCE MATRIX] Training complete.")


if __name__ == "__main__":
    # Resolve project root relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    data_file = os.path.join(project_root, "data", "audience_data.csv")
    models_directory = os.path.join(project_root, "models")
    
    train_pipeline(data_file, models_directory)
