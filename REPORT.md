# AUDIENCE MATRIX — Technical Engineering & Methodology Report

> **Containerized OTT Audience Segmentation & Personalization**  
> *Deterministic Scaled KMeans Clustering & Production Inference Engine*

---

## 1. Problem Understanding

Modern Over-The-Top (OTT) streaming platforms face significant challenges in audience retention, content discovery, and churn mitigation. Streaming telemetry data arrives at immense scale, capturing nuanced viewer habits across multiple dimensions—such as session duration, viewing regularity, weekend concentration, completion rates, and genre affinities.

Traditional rule-based categorization creates rigid, fragile audience buckets that fail to adapt to continuous behavioral variances. Supervised machine learning approaches often suffer from lack of true ground-truth labels for "subscriber mindset." 

To address this, **AUDIENCE MATRIX** implements an unsupervised machine learning architecture utilizing standardized **KMeans Clustering** to discover organic subscriber behavioral archetypes directly from multivariate telemetry. The discovered segments feed directly into a transparent, multi-factor personalization engine that re-ranks content catalogs and generates explainable recommendations in real time.

---

## 2. Solution Overview

**AUDIENCE MATRIX** is built as an end-to-end, reproducible, containerized streaming intelligence platform composed of:
1. **Synthetic Telemetry Generator**: Produces 8,500 subscriber records reflecting realistic streaming behaviors without artificial cluster labels.
2. **Standardized KMeans Clustering Pipeline**: Performs feature scaling, hyperparameter exploration across $K \in [2, 8]$, metric evaluation (Inertia, Silhouette score, cluster balance), and semantic centroid translation into human-interpretable segment names.
3. **Model & Metadata Persistence**: Serializes model parameters, scalers, and centroid profiles using `joblib`.
4. **Production REST API & Dashboard Service**: Built with Flask, serving a dark navy/violet enterprise analytics interface, exposing `/health` and `/recommend` endpoints.
5. **Transparent Personalization Engine**: Re-ranks a 20-title content catalog using a multi-factor scoring function incorporating predicted segment archetypes, explicit user genre preferences, session duration compatibility, and baseline popularity.
6. **Automated Evaluator Service**: Runs regression persona benchmarks and 8 edge-case fault-injection tests, serializing verified results into `metrics.json`.

---

## 3. System Architecture

```
                               +-------------------------------------+
                               |          AUDIENCE MATRIX            |
                               +-------------------------------------+
                                                  |
                 +--------------------------------+-------------------------------+
                 |                                |                               |
                 v                                v                               v
       [ trainer/train.py ]              [ api/app.py ]                [ evaluator/evaluate.py ]
                 |                                |                               |
        (KMeans Pipeline)                 (Flask REST API)                (Validation Suite)
                 |                                |                               |
        +--------+--------+                       |                               |
        |                 |                       |                               |
        v                 v                       v                               v
 [data/audience_data.csv] [models/] <------+------+-------------------------------+
   (8,500 Telemetry Rows)    |             | (Loads Models / Catalogs)   (Writes metrics.json)
                             |             v
                             +----> [ Dashboard UI (HTML/CSS/JS) ]
```

---

## 4. Dataset & Synthetic Telemetry Assumptions

### 4.1 Synthetic Dataset Disclosure
In full compliance with project specifications, **the dataset utilized by AUDIENCE MATRIX is 100% synthetically generated**. No proprietary streaming logs or personal identifiable information (PII) were utilized.

### 4.2 Data Generation Methodology
The dataset simulates 8,500 OTT subscribers over a fixed random seed (`seed=42`). To ensure that unsupervised KMeans discovers natural, organic clusters rather than synthetic artifacts:
- **No artificial cluster labels are present in `data/audience_data.csv`**.
- Heterogeneous subscriber archetypes are synthesized using multivariate Gaussian distributions with randomized continuous noise.
- Every subscriber row represents a unique `user_id` (`USR_00001` through `USR_08500`).
- Physical bounds are enforced:
  - `watch_time_hours`: Clamped between 1.0 and 75.0 hours/week.
  - `avg_session_minutes`: Clamped between 10.0 and 180.0 minutes.
  - `sessions_per_week`: Clamped between 1.0 and 30.0 sessions.
  - `completion_rate`: Clamped between 0.10 and 0.99.
  - `weekend_ratio`: Clamped between 0.05 and 0.95.
  - `genre_diversity`: Clamped integer between 1 and 5.
  - `genre_affinities`: Clamped continuous scores between 0.05 and 0.99 across 5 supported verticals: *Action*, *Comedy*, *Thriller*, *Drama*, *Sci-Fi*.

---

## 5. Feature Selection & Preprocessing

### 5.1 Telemetry Features
The unsupervised segmentation model trains on 11 continuous and discrete numerical features:

| Feature Name | Description | Range |
| :--- | :--- | :--- |
| `watch_time_hours` | Cumulative weekly streaming duration | 1.0 – 75.0 |
| `avg_session_minutes` | Average viewing session duration | 10.0 – 180.0 |
| `sessions_per_week` | Weekly frequency of streaming sessions | 1.0 – 30.0 |
| `completion_rate` | Ratio of initiated content watched to completion | 0.10 – 0.99 |
| `weekend_ratio` | Proportion of viewing taking place on Sat/Sun | 0.05 – 0.95 |
| `genre_diversity` | Number of distinct genre categories consumed | 1 – 5 |
| `action_affinity` | Normalized engagement affinity for Action content | 0.05 – 0.99 |
| `comedy_affinity` | Normalized engagement affinity for Comedy content | 0.05 – 0.99 |
| `thriller_affinity` | Normalized engagement affinity for Thriller content | 0.05 – 0.99 |
| `drama_affinity` | Normalized engagement affinity for Drama content | 0.05 – 0.99 |
| `sci_fi_affinity` | Normalized engagement affinity for Sci-Fi content | 0.05 – 0.99 |

### 5.2 Standardization
Because distance metrics in Euclidean space are sensitive to feature variances (e.g. `avg_session_minutes` spanning up to 180 vs `completion_rate` spanning 0 to 1), all 11 features are scaled via `StandardScaler`:
$$z = \frac{x - \mu}{\sigma}$$
The fitted scaler is persisted as `models/scaler.joblib` to ensure runtime inference vectors undergo identical mathematical transformation.

---

## 6. Unsupervised KMeans Clustering & K Selection

### 6.1 Hyperparameter Sweep ($K = 2$ to $8$)
The trainer evaluated KMeans convergence across $K \in [2, 8]$ using `n_init=10`, `max_iter=300`, and `random_state=42`.

#### Actual Experimental Evaluation Results:
| $K$ | Inertia (WCSS) | Silhouette Score | Cluster Balance (Min/Max Ratio) |
| :---: | :---: | :---: | :---: |
| **2** | 56,599.64 | 0.3525 | 0.76 |
| **3** | 37,837.81 | 0.3797 | 0.52 |
| **4** | **31,222.42** | **0.3808** | **0.38** (Selected) |
| **5** | 26,642.79 | 0.3348 | 0.67 |
| **6** | 25,827.22 | 0.2578 | 0.43 |
| **7** | 25,108.39 | 0.2218 | 0.53 |
| **8** | 24,404.00 | 0.1883 | 0.38 |

### 6.2 Selection Rationale
- **Silhouette Coefficient**: Measures how similar an object is to its own cluster compared to other clusters. $K=4$ achieves the peak silhouette coefficient of **0.3808**, indicating maximum inter-cluster separation and intra-cluster cohesion.
- **Cluster Balance**: The smallest cluster at $K=4$ contains 1,352 users (15.91%) while the largest contains 3,571 users (42.01%), yielding a healthy balance ratio of 0.38 with zero degenerate or trivial clusters.
- **Interpretability**: $K=4$ yields 4 distinct, actionable operational segments. Higher values of $K$ ($K \ge 6$) produced severe cluster fragmentation and declining silhouette scores ($< 0.26$).

---

## 7. Cluster Profiles & Semantic Segment Names

By inverse-transforming cluster centroids back to unscaled telemetry space and benchmarking them against the population baseline, human-readable segment names were automatically assigned:

### Segment 0: Weekend Marathon Streamers
- **Population Share**: 3,571 users (42.01%)
- **Unscaled Centroid**: 24.8 hrs watch time, 85.0 mins/session, 5.3 sessions/wk, 85.0% completion, 62.7% weekend ratio.
- **Dominant Affinities**: Action (0.81), Sci-Fi (0.80), Thriller (0.73).
- **Core Archetype**: High-intensity weekend bingers who favor long, uninterrupted sessions and sci-fi/action epics.

### Segment 1: Primetime Narrative Devotees
- **Population Share**: 1,708 users (20.09%)
- **Unscaled Centroid**: 17.5 hrs watch time, 54.7 mins/session, 5.8 sessions/wk, 87.8% completion, 36.1% weekend ratio.
- **Dominant Affinities**: Drama (0.89), Thriller (0.55), Comedy (0.40).
- **Core Archetype**: Dedicated serialized drama viewers with consistent evening viewing patterns and superior completion loyalty.

### Segment 2: Casual Quick-Bite Streamers
- **Population Share**: 1,869 users (21.99%)
- **Unscaled Centroid**: 6.8 hrs watch time, 26.1 mins/session, 4.2 sessions/wk, 58.0% completion, 31.8% weekend ratio.
- **Dominant Affinities**: Comedy (0.84), Drama (0.68), Action (0.25).
- **Core Archetype**: Mobile, commute, and snackable session viewers with low time commitments and high appetite for light comedy.

### Segment 3: Eclectic Multi-Genre Omnivores
- **Population Share**: 1,352 users (15.91%)
- **Unscaled Centroid**: 19.5 hrs watch time, 48.1 mins/session, 8.4 sessions/wk, 73.9% completion, 42.2% weekend ratio.
- **Dominant Affinities**: Drama (0.68), Action (0.65), Thriller (0.64), Comedy (0.62), Sci-Fi (0.58), Genre Diversity (4.51 / 5).
- **Core Archetype**: High-frequency subscribers with broad, multifaceted tastes who consume widely across the entire catalog.

---

## 8. Recommendation System & Transparent Explainability

### 8.1 Fictional Catalog
A curated catalog of 20 fictional OTT titles (`recommendation_catalog.json`) spans both Movies and Episodic Series across all five supported genres (*Action*, *Comedy*, *Thriller*, *Drama*, *Sci-Fi*), featuring explicit runtimes and baseline popularity indices.

### 8.2 Transparent Ranking Formula
When a subscriber profile is submitted to `POST /recommend`, each catalog candidate $t$ is scored using a 4-component weighted function:

$$\text{Score}(t) = 0.45 \cdot S_{\text{genre}}(t) + 0.25 \cdot S_{\text{format}}(t) + 0.20 \cdot S_{\text{segment}}(t) + 0.10 \cdot S_{\text{pop}}(t)$$

1. **Genre Alignment ($S_{\text{genre}}$ - 45% weight)**: Matches candidate genres against the viewer's explicit `top_genres` array, prioritizing primary and secondary preferences.
2. **Format Compatibility ($S_{\text{format}}$ - 25% weight)**: Matches session length patterns. Subscribers with `avg_session_minutes` $\ge 65$ receive bonus weight for Movies and long-form series; subscribers with short sessions ($\le 35$ mins) receive bonus weight for episodic comedies.
3. **Segment Archetype Affinity ($S_{\text{segment}}$ - 20% weight)**: Checks alignment between candidate genres and the dominant affinities of the assigned cluster centroid.
4. **Catalog Popularity ($S_{\text{pop}}$ - 10% weight)**: Incorporates base title engagement rating.

### 8.3 Distance to Centroid vs. "Confidence"
Per project specifications, the Euclidean distance in scaled feature space:
$$d = \| \mathbf{z}_{\text{subscriber}} - \boldsymbol{\mu}_{\text{segment}} \|_2$$
is reported strictly as `distance_to_centroid`. **It is explicitly not designated as a confidence score**, preserving statistical transparency.

---

## 9. API Specification & Input Validation

The Flask service enforces strict data integrity:

| Endpoint | Method | Purpose | Input / Status |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | Serves the AUDIENCE MATRIX Analytics Dashboard | HTML (200) |
| `/health` | `GET` | Healthcheck returning model status & catalog count | JSON (200 / 503) |
| `/api/overview` | `GET` | Complete telemetry metrics, segments, & evaluator data | JSON (200) |
| `/api/metrics` | `GET` | Actual test execution report from `metrics.json` | JSON (200 / 404) |
| `/recommend` | `POST` | Real-time segmentation & personalized recommendations | JSON (200 / 400) |

### Robust Validation Layer
The `/recommend` endpoint validates:
- **Presence of required fields**: Rejects missing fields with descriptive 400 errors.
- **Type safety**: Verifies numerical types, boolean discrimination, and string structures.
- **Physical boundary constraints**:
  - `watch_time_hours`: $0.0 \le t \le 168.0$ (weekly hours limit).
  - `avg_session_minutes`: $0 < m \le 720$ (positive, 12h maximum).
  - `sessions_per_week`: $0 \le s \le 100$.
  - `completion_rate`: $0.0 \le c \le 1.0$.
  - `weekend_ratio`: $0.0 \le w \le 1.0$.
  - `genre_diversity`: integer $1 \le d \le 5$.
- **Genre validity**: Must contain at least 1 genre; all elements must be strictly members of `['Action', 'Comedy', 'Thriller', 'Drama', 'Sci-Fi']`. Rejects empty lists `[]` or unsupported genres like `'Horror'`.
- **Malformed JSON safety**: Traps syntax errors and returns clean JSON without exposing stack traces.

---

## 10. Container Architecture & Orchestration

The application is containerized into three decoupled microservices in `docker-compose.yml`:

```yaml
services:
  trainer:   # Generates dataset, trains KMeans, saves artifacts, exits cleanly
  api:       # Loads artifacts, serves dashboard, exposes /health and /recommend
  evaluator: # Waits for API health, executes test suite, outputs metrics.json
```

- **Docker Healthchecks**: The API container exposes a native Python healthcheck:
  `["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]`
- **Volume Sharing**: `./data` and `./models` are mounted across containers to share dataset records, serializations (`kmeans_model.joblib`), and live evaluation metrics (`metrics.json`).
- **One-Command Startup**: Fully launches via `docker compose up --build`.

---

## 11. Automated Evaluation & Live Results

The automated evaluator executed **13 distinct test scenarios** against the live API, achieving a **100% pass rate**:

### 11.1 Representative Persona Tests (5 / 5 Passed)
1. **High-Engagement Sci-Fi/Action Viewer** (`36.5h`, `88m`, `Sci-Fi/Action`): Assigned to *Weekend Marathon Streamers* (Dist: 3.2321, 5 recommendations, 2072ms).
2. **Casual Quick-Bite Comedy Viewer** (`6.2h`, `24m`, `Comedy/Drama`): Assigned to *Casual Quick-Bite Streamers* (Dist: 0.6440, 5 recommendations, 2063ms).
3. **Intense Thriller Purist** (`19.0h`, `58m`, `Thriller`): Assigned to *Primetime Narrative Devotees* (Dist: 3.9668, 5 recommendations, 2069ms).
4. **Multi-Genre Balanced Omnivore** (`21.0h`, `50m`, All 5 Genres): Assigned to *Eclectic Multi-Genre Omnivores* (Dist: 1.1821, 5 recommendations, 2055ms).
5. **Marathon Weekend Binge Viewer** (`29.5h`, `115m`, `85% Weekend`): Assigned to *Weekend Marathon Streamers* (Dist: 2.9802, 5 recommendations, 2041ms).

### 11.2 Edge Case & Fault Injection Tests (8 / 8 Passed)
1. `empty_genres` (`top_genres: []`): Expected 400 &rarr; **Got 400** (PASS).
2. `unseen_genre` (`top_genres: ["Horror", "Documentary"]`): Expected 400 &rarr; **Got 400** (PASS).
3. `zero_watch_time` (`watch_time_hours: 0.0`): Expected 200 &rarr; **Got 200** (PASS, graceful handling).
4. `very_large_values` (`watch_time_hours: 999.0`): Expected 400 &rarr; **Got 400** (PASS).
5. `missing_field` (omitted `completion_rate`): Expected 400 &rarr; **Got 400** (PASS).
6. `wrong_type` (`watch_time_hours: "eighteen"`): Expected 400 &rarr; **Got 400** (PASS).
7. `negative_value` (`avg_session_minutes: -45`): Expected 400 &rarr; **Got 400** (PASS).
8. `malformed_json` (corrupted raw string): Expected 400 &rarr; **Got 400** (PASS).

---

## 12. Reproducibility & Verification

To reproduce results identically:
1. Random seed `seed=42` guarantees consistent telemetry dataset generation.
2. `KMeans(random_state=42, n_init=10)` ensures deterministic cluster assignments.
3. Model artifacts and metrics are persisted and reloaded without re-training during inference.

---

## 13. Limitations & Rejected Approaches

### 13.1 Limitations
- **Synthetic Data**: While distributions mimic realistic streaming telemetry, real-world platforms exhibit higher churn seasonality, holiday surges, and noisy multi-user account sharing.
- **Cold-Start Content**: Recommender relies on fixed catalog genres and popularity ratings; dynamic content embeddings could improve unseen title ranking.

### 13.2 Rejected / Prohibited Approaches
- **Deep Neural Embeddings**: Rejected as needlessly complex, computationally opaque, and in violation of project constraints.
- **RFM (Recency, Frequency, Monetary)**: Rejected because streaming platforms are subscription-based rather than transactional retail; watch time, session length, and completion rate provide superior behavioral signals.
- **Churn Prediction**: Supervised churn classification was excluded to maintain pure unsupervised audience segmentation focus.
- **External APIs / LLMs**: Prohibited to guarantee container independence, deterministic execution, zero latency overhead, and no external API billing dependencies.

---

## 14. Future Improvements

1. **Temporal Drift Monitoring**: Implement Kolmogorov-Smirnov drift detection on streaming feature vectors to trigger periodic retraining.
2. **Contextual Dayparting**: Expand time-of-day feature ingestion (e.g., lunch-break mobile sessions vs. late-night TV viewing).
3. **A/B Testing Container Service**: Implement multi-model shadow testing comparing alternative distance metrics (e.g. Cosine vs Euclidean).
