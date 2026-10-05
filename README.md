# AUDIENCE MATRIX

> **Containerized OTT Audience Segmentation & Personalization**  
> *Unsupervised KMeans Telemetry Clustering, Real-Time Recommendation Engine, and Enterprise Analytics*

---

**DEMO : https://audience-matrix-vwsy.onrender.com/**

## 1. Overview

**AUDIENCE MATRIX** is an end-to-end, production-ready audience segmentation and personalization system engineered for OTT streaming platforms. Operating on multivariate subscriber telemetry (watch duration, session lengths, completion rates, weekend ratios, and genre affinities), **AUDIENCE MATRIX** leverages standardized **unsupervised KMeans clustering** to discover organic viewer cohorts and powers a multi-factor recommendation engine with transparent, explainable recommendations.

---

## 2. Key Features

- **Unsupervised KMeans Segmentation**: Evaluates $K=2$ to $K=8$ and deterministically converges on the optimal $K$ based on silhouette coefficients, cluster balance, and interpretability.
- **Dynamic Segment Interpretation**: Converts unscaled cluster centroids into intuitive human-readable subscriber cohorts (e.g., *Weekend Marathon Streamers*, *Primetime Narrative Devotees*, *Casual Quick-Bite Streamers*, *Eclectic Multi-Genre Omnivores*).
- **Explainable Content Personalization**: Multi-factor recommender scoring over a 20-title fictional catalog integrating segment archetypes, explicit genre affinities, viewing format compatibility, and title popularity.
- **Enterprise Dark Analytics Dashboard**: Minimal, professional interface themed in obsidian navy and violet accents, featuring real-time KPI metrics, segment cards, an interactive subscriber simulator, an Audience DNA matrix, and live evaluator pass ledgers.
- **Strict Validation & Error Resilience**: Validates ranges, types, unsupported genres, boundary conditions, and malformed JSON without exposing internal stack traces.
- **Fully Containerized & Reproducible**: Docker Compose orchestration decoupling the `trainer`, `api`, and `evaluator` microservices with healthcheck dependency guarantees.

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.11 / 3.12
- **Data & Machine Learning**: Pandas, NumPy, Scikit-learn, Joblib
- **Web & Serving**: Flask, Gunicorn, Requests
- **Frontend**: Vanilla HTML5, Modern CSS3 (Grid & Custom Properties), JavaScript (ES6+ Fetch API)
- **Containerization**: Docker, Docker Compose

---

## 4. Repository Structure

```
AUDIENCE MATRIX
│
├── data/
│   └── audience_data.csv             # 8,500 synthetic viewer telemetry records
│
├── models/
│   ├── kmeans_model.joblib           # Trained KMeans model artifact
│   ├── scaler.joblib                 # Fitted StandardScaler artifact
│   ├── cluster_metadata.joblib       # Centroids, segment names, and metrics
│   ├── training_summary.json         # Human-readable training summary
│   └── metrics.json                  # Actual live test execution ledger
│
├── trainer/
│   ├── Dockerfile                    # Container definition for model trainer
│   ├── requirements.txt              # ML pipeline dependencies
│   └── train.py                      # Unsupervised KMeans training pipeline
│
├── api/
│   ├── Dockerfile                    # Container definition for API service
│   ├── requirements.txt              # Web service dependencies
│   ├── app.py                        # REST API & dashboard controller
│   ├── recommendation_catalog.json   # 20-title fictional OTT content catalog
│   ├── templates/
│   │   └── index.html                # Enterprise analytics dashboard UI
│   └── static/
│       ├── style.css                 # Dark navy/violet design system
│       └── app.js                    # Dynamic telemetry dashboard client
│
├── evaluator/
│   ├── Dockerfile                    # Container definition for evaluator
│   ├── requirements.txt              # Test suite dependencies
│   └── evaluate.py                   # Automated robustness & persona tests
│
├── docker-compose.yml                # Microservices orchestration
├── README.md                         # Project documentation
├── REPORT.md                         # Detailed engineering report
├── BRAND_GUIDELINES.md               # Brand standards & design tokens
└── .dockerignore                     # Container build exclusions
```

---

## 5. Quick Start & Execution

### 5.1 Containerized Execution (Recommended)

Run the complete pipeline (trainer $\rightarrow$ API $\rightarrow$ evaluator) with a single command:

```bash
docker compose up --build
```

**Workflow Lifecycle**:
1. `trainer` starts, generates `data/audience_data.csv`, evaluates $K \in [2,8]$, saves models to `models/`, and exits cleanly (`code 0`).
2. `api` starts upon trainer completion, loads persisted artifacts without retraining, initializes the Flask server, and verifies its healthcheck (`GET /health`).
3. `evaluator` waits for the API healthcheck, executes 5 persona tests and 8 edge cases, and writes actual live results to `models/metrics.json`.
4. The dashboard is accessible at: **`http://localhost:5000`**.

---

### 5.2 Local Execution (Direct Python)

If running directly in a local Python virtual environment:

```bash
# 1. Install dependencies
pip install -r trainer/requirements.txt
pip install -r api/requirements.txt
pip install -r evaluator/requirements.txt

# 2. Train model and generate dataset
python trainer/train.py

# 3. Launch API & Dashboard service
python api/app.py

# 4. In a separate terminal, execute the test suite
python evaluator/evaluate.py
```

---

## 6. API Endpoints

### 6.1 `GET /health`
Returns system status, model state, and catalog count.

**Response (200 OK)**:
```json
{
  "brand": "AUDIENCE MATRIX",
  "model_loaded": true,
  "selected_k": 4,
  "status": "healthy",
  "timestamp": "2026-10-05T13:04:30.996833+00:00",
  "total_catalog_titles": 20
}
```

---

### 6.2 `POST /recommend`
Performs real-time subscriber segmentation and personalized recommendation.

**Supported Genres**: `Action`, `Comedy`, `Thriller`, `Drama`, `Sci-Fi`

**Example Request**:
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_001",
    "watch_time_hours": 18.5,
    "avg_session_minutes": 52,
    "sessions_per_week": 6,
    "completion_rate": 0.82,
    "weekend_ratio": 0.35,
    "genre_diversity": 3,
    "top_genres": ["Action", "Thriller"]
  }'
```

**Example Response**:
```json
{
  "user_id": "demo_001",
  "segment_id": 0,
  "segment_name": "Weekend Marathon Streamers",
  "segment_description": "Long-duration weekend marathon viewers favoring immersive storylines, high completion, and high-intensity genres.",
  "distance_to_centroid": 3.141,
  "explanation": "Viewer 'demo_001' assigned to 'Weekend Marathon Streamers' based on viewing pace (18.5 hrs/wk, 52 mins/session) and weekend affinity (35%). Recommendations prioritized titles matching top genres (Action, Thriller) harmonized with the behavioral format preferences of the segment.",
  "recommendations": [
    {
      "title": "Midnight Syndicate",
      "genres": ["Thriller", "Action"],
      "type": "Series",
      "duration": "8 Episodes (52 mins)",
      "popularity": 0.91,
      "match_score": 0.838,
      "match_reason": "Matches preferred genre (Thriller, Action) • Trending platform title",
      "description": "An undercover intelligence analyst infiltrates a high-stakes algorithmic crime syndicate in Berlin."
    },
    {
      "title": "Zero Hour Velocity",
      "genres": ["Action", "Thriller"],
      "type": "Movie",
      "duration": "112 mins",
      "popularity": 0.89,
      "match_score": 0.836,
      "match_reason": "Matches preferred genre (Action, Thriller)",
      "description": "An elite test pilot must intercept an autonomous combat drone before it breaches sovereign airspace."
    }
  ]
}
```

> **Note**: In accordance with system specifications, `distance_to_centroid` represents exact Euclidean distance in standardized feature space and is **never** referred to as a "confidence score".

---

## 7. Automated Evaluator & Quality Metrics

The test suite in `evaluator/evaluate.py` verifies both behavioral fidelity and system security:

| Category | Total Tests | Passed | Success Rate |
| :--- | :---: | :---: | :---: |
| **Representative Personas** | 5 | 5 | 100.0% |
| **Edge-Case & Fault Injection** | 8 | 8 | 100.0% |
| **Overall Robustness** | **13** | **13** | **100.0%** |

- **Edge Cases Tested**: Empty genre array, unsupported/unseen genre, zero watch-time boundary, exorbitant watch-time (>168 hrs), missing payload keys, wrong data types, negative numbers, and malformed non-JSON syntax.
- All test results are logged directly to `models/metrics.json` from actual runtime execution.

---

## 8. Dashboard Interface

Access `http://localhost:5000` to interact with:
1. **Executive Overview**: Real-time KPI tiles for Total Subscribers ($8,500$), Discovered Segments ($4$), Peak Silhouette Score ($0.3808$), and Optimal $K$ ($4$).
2. **Audience Segments**: Distribution cards showcasing population percentage, cohort sizes, and behavioral traits.
3. **Audience DNA Matrix**: Micro-bar comparisons across watch hours, session duration, weekly sessions, completion rate, and weekend share.
4. **Try A Viewer Simulator**: Interactive form featuring 1-click persona presets (*Weekend Marathoner*, *Casual Quick-Bite*, *Primetime Drama*, *Multi-Genre Omnivore*) that executes live queries against `/recommend`.
5. **Model Quality & Hyperparameter Sweep**: Inertia and Silhouette scores across $K=2$ through $K=8$.
6. **Robustness Ledger**: Live pass/fail records and latencies loaded dynamically from `metrics.json`.

---

## 9. Assumptions & Limitations

- **Synthetic Telemetry**: The dataset is completely synthetic and generated using seed-controlled Gaussian noise without artificial labels.
- **Fixed Genre Space**: The model operates on 5 supported OTT genres (`Action`, `Comedy`, `Thriller`, `Drama`, `Sci-Fi`).
- **Static Catalog**: The recommendation engine re-ranks an internal catalog of 20 fictional titles using transparent heuristics.
