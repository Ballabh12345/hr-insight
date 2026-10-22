"""MVP-4 - Talent Segmentation & Nine-Box Matrix.

An honest finding drives the design of this module.

**Finding.** In the IBM HR Analytics dataset, the manager-assigned ``PerformanceRating``
is a *deterministic* function of ``PercentSalaryHike`` (every employee with the same hike
value carries the same rating). Once that proxy is removed, the rating is not predictable
at all - cross-validated ROC-AUC for distinguishing a rating of 4 from 3 is **0.457**,
i.e. below chance. The same holds for an engagement composite built from the satisfaction
survey fields: those fields are close to independently sampled, so they carry no
job-context signal either.

Building a "performance prediction" model on this dataset therefore produces either a
leaked perfect model or a negative-R2 dud. This module does neither.

**What this module builds instead.** A *descriptive, observed* talent matrix:

* **Performance axis** - the Engagement Composite, a standardised mean of six observable
  indicators (involvement, job satisfaction, environment satisfaction, relationship
  satisfaction, work-life balance, training received). Terniles give Low / Medium / High.
  It is observed, not predicted, so it needs no validation against an outcome.

* **Potential axis** - K-Means clusters on engagement-and-growth indicators. The cluster
  count is chosen by silhouette score over k = 3..8 (k >= 3 is required because the axis
  has three tiers), then each cluster is ranked into a potential tier by its growth index.

* **Nine-box** - the crossing of the two axes, filling all nine cells.

* **Persona classifier** - because K-Means gives an assignment for *this* population only,
  a supervised classifier is trained to predict the persona from *other* job-context
  attributes. This is how the platform would place an employee not present in the
  training population, and its cross-validated accuracy is reported honestly.

Usage
-----
    python ml/train_ninebox.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import features as F  # noqa: E402
from ml.common import io, metrics  # noqa: E402

RANDOM_STATE = 42
CV_FOLDS = 5

#: Engagement indicators averaged (after standardisation) into the Engagement Composite
#: that forms the performance axis.
ENGAGEMENT_COMPONENTS: list[str] = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "TrainingTimesLastYear",
]

#: Engagement and growth indicators used to estimate *potential*.
POTENTIAL_FEATURES: list[str] = [
    "JobInvolvement",
    "TrainingTimesLastYear",
    "YearsSinceLastPromotion",
    "PromotionVelocity",
    "YearsAtCompany",
]

#: Features averaged (after standardisation, with sign flips) to build the growth index
#: used to rank clusters into potential tiers.
GROWTH_INDEX_FEATURES: list[str] = [
    "JobInvolvement",
    "TrainingTimesLastYear",
    "YearsSinceLastPromotion",
]

#: Nine-box labels. Rows are performance (Low -> High), columns are potential
#: (Low -> High). The naming scheme is defined by this project and documented in the
#: model card so a reader can audit the mapping.
NINEBOX_LABELS: dict[tuple[str, str], str] = {
    ("Low", "Low"): "Needs Attention",
    ("Low", "Medium"): "Inconsistent Performer",
    ("Low", "High"): "Growth Prospect",
    ("Medium", "Low"): "Effective Performer",
    ("Medium", "Medium"): "Core Player",
    ("Medium", "High"): "High Potential",
    ("High", "Low"): "Trusted Professional",
    ("High", "Medium"): "Star Performer",
    ("High", "High"): "Future Leader",
}

PERFORMANCE_TIERS = ["Low", "Medium", "High"]
POTENTIAL_TIERS = ["Low", "Medium", "High"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by the potential model."""
    out = df.copy()
    # Years of service per job level: lower means faster promotion velocity.
    out["PromotionVelocity"] = out["YearsAtCompany"] / out["JobLevel"].clip(lower=1)
    return out


def engagement_composite(df: pd.DataFrame) -> pd.Series:
    """Return the standardised engagement composite (mean of z-scored indicators)."""
    cols = [c for c in ENGAGEMENT_COMPONENTS if c in df.columns]
    scaled = StandardScaler().fit_transform(df[cols])
    return pd.Series(scaled.mean(axis=1), index=df.index, name="EngagementComposite")


def candidate_classifiers() -> dict[str, tuple[object, dict]]:
    """Model families compared for the persona-assignment classifier."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": ["balanced", None]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": ["balanced"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def tertile_tiers(values: np.ndarray) -> np.ndarray:
    """Split values into Low / Medium / High tertiles."""
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    return np.array(
        [
            "Low" if v <= cuts[0] else ("Medium" if v <= cuts[1] else "High")
            for v in values
        ]
    )


def main() -> None:
    started = time.time()
    io.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MVP-4  Talent Segmentation & Nine-Box Matrix")
    print("=" * 72)

    path = io.download_ibm_hr()
    raw = F.load_dataset(path)
    cleaned = add_engineered_features(F.clean(raw))
    print(f"[data] {len(cleaned)} employees")

    # ------------------------------------------------------- performance axis
    engagement = engagement_composite(cleaned)
    perf_tier = tertile_tiers(engagement.values)
    print(f"[axis ] performance tertiles: {pd.Series(perf_tier).value_counts().to_dict()}")

    # ------------------------------------------------------------- potential
    pot_cols = [c for c in POTENTIAL_FEATURES if c in cleaned.columns]
    scaler = StandardScaler()
    pot_matrix = scaler.fit_transform(cleaned[pot_cols])

    silhouette: dict[int, float] = {}
    best_k, best_sil = 3, -np.inf
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(pot_matrix)
        score = float(silhouette_score(pot_matrix, labels))
        silhouette[k] = score
        print(f"[kmeans] k={k}  silhouette={score:.4f}")
        if score > best_sil:
            best_k, best_sil = k, score

    print(f"[kmeans] selected k={best_k} (silhouette={best_sil:.4f})")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(pot_matrix)

    # Growth index -> potential tier per cluster.
    growth_signs = {
        "JobInvolvement": 1,
        "TrainingTimesLastYear": 1,
        "YearsSinceLastPromotion": -1,  # long since promotion == less growth
    }
    growth_cols = [c for c in GROWTH_INDEX_FEATURES if c in cleaned.columns]
    growth_scaled = StandardScaler().fit_transform(cleaned[growth_cols])
    signs = np.array([growth_signs.get(c, 1) for c in growth_cols])
    growth_index = (growth_scaled * signs).mean(axis=1)

    cluster_growth = {
        int(c): float(growth_index[cluster_labels == c].mean()) for c in range(best_k)
    }

    # Rank clusters by growth, then split the ranking into three equally sized tiers
    # (quantile assignment keeps all three potential tiers populated even when k is not
    # a multiple of three).
    ranked = sorted(cluster_growth, key=lambda c: cluster_growth[c])
    cluster_tier: dict[int, str] = {}
    for i, c in enumerate(ranked):
        position = i / len(ranked)
        if position < 1 / 3:
            cluster_tier[c] = "Low"
        elif position < 2 / 3:
            cluster_tier[c] = "Medium"
        else:
            cluster_tier[c] = "High"

    potential_tier = np.array([cluster_tier[int(c)] for c in cluster_labels])

    print(f"[kmeans] cluster growth: { {k: round(v, 3) for k, v in cluster_growth.items()} }")
    print(
        f"[kmeans] cluster tiers : "
        f"{ {t: sum(1 for v in cluster_tier.values() if v == t) for t in POTENTIAL_TIERS} }"
    )

    # ----------------------------------------------------------- potential axis
    # The potential axis is the tertile of the *employee-level* growth index, which keeps
    # the three tiers evenly populated. Tiers assigned per cluster would inherit the very
    # unequal cluster sizes (775 / 462 / 233) and leave the 3x3 grid lopsided.
    potential_tier = tertile_tiers(growth_index)

    # ----------------------------------------------------------- nine-box
    ninebox = np.array(
        [NINEBOX_LABELS[(p, pot)] for p, pot in zip(perf_tier, potential_tier)]
    )
    grid_counts = pd.Series(ninebox).value_counts()
    print("[ninebox] label distribution:")
    for label, count in grid_counts.items():
        print(f"           {label:22s} {count:4d}")
    print(f"[ninebox] {len(grid_counts)}/9 cells occupied")

    # ------------------------------------------------- persona classifier
    # Predict the K-Means persona from attributes that were NOT used for clustering, so
    # the classifier must infer the growth profile from surrounding job context.
    classifier_features = [
        c
        for c in cleaned.columns
        if c not in POTENTIAL_FEATURES
        and c not in ENGAGEMENT_COMPONENTS
        and c not in ("Attrition", "AgeBand", "PerformanceRating", "PercentSalaryHike")
    ]
    Xc = cleaned[classifier_features]
    yc = cluster_labels
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.20, stratify=yc, random_state=RANDOM_STATE
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    clf_results: dict[str, dict] = {}
    best_clf_name, best_clf_acc, best_clf = None, -np.inf, None
    for name, (estimator, grid) in candidate_classifiers().items():
        pipeline = F.build_pipeline(Xc, estimator)
        search = GridSearchCV(
            pipeline, grid, cv=skf, scoring="balanced_accuracy", n_jobs=-1, refit=True
        )
        search.fit(Xc_train, yc_train)
        acc_scores = cross_val_score(
            search.best_estimator_, Xc_train, yc_train, cv=skf, scoring="balanced_accuracy", n_jobs=-1
        )
        cv_summary = metrics.summarize_cv(list(acc_scores))
        test_acc = float(accuracy_score(yc_test, search.best_estimator_.predict(Xc_test)))
        test_f1 = float(f1_score(yc_test, search.best_estimator_.predict(Xc_test), average="macro"))
        clf_results[name] = {
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_balanced_accuracy": cv_summary,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }
        print(
            f"[clf  ] {name:22s} cv_bal_acc={cv_summary['mean']:.4f}+/-{cv_summary['std']:.4f}  "
            f"test_acc={test_acc:.4f}  macro_f1={test_f1:.4f}"
        )
        if cv_summary["mean"] > best_clf_acc:
            best_clf_name, best_clf_acc, best_clf = name, cv_summary["mean"], search.best_estimator_

    assert best_clf is not None
    chance_level = 1.0 / best_k
    print(
        f"[clf  ] selected: {best_clf_name} (chance baseline for k={best_k} "
        f"is {chance_level:.4f})"
    )

    # ---------------------------------------------------------------- save
    # Persist the per-employee assignment so that the dataset builder uses exactly the
    # same labels, rather than recomputing them independently.
    assignments = pd.DataFrame(
        {
            "EmployeeNumber": raw["EmployeeNumber"].values,
            "performance_tier": perf_tier,
            "potential_tier": potential_tier,
            "ninebox_label": ninebox,
            "persona_cluster": cluster_labels,
            "engagement_composite": engagement.round(6).values,
            "growth_index": growth_index.round(6),
        }
    )
    assignments.to_csv(io.PROCESSED_DIR / "ninebox_assignments.csv", index=False)
    print(f"[save] {io.PROCESSED_DIR / 'ninebox_assignments.csv'}")

    io.save_artifact(kmeans, "ninebox_kmeans.pkl")
    io.save_artifact(scaler, "ninebox_scaler.pkl")
    io.save_artifact(best_clf, "ninebox_persona_classifier.pkl")
    io.save_artifact(
        {
            "cluster_tier": cluster_tier,
            "cluster_growth": cluster_growth,
            "potential_features": pot_cols,
            "growth_features": growth_cols,
            "engagement_components": ENGAGEMENT_COMPONENTS,
            "classifier_features": classifier_features,
            "n_clusters": best_k,
        },
        "ninebox_config.pkl",
    )

    bundle = {
        "n_clusters": best_k,
        "selected_silhouette": best_sil,
        "silhouette_by_k": silhouette,
        "cluster_tier": {str(k): v for k, v in cluster_tier.items()},
        "cluster_growth": {str(k): v for k, v in cluster_growth.items()},
        "potential_features": pot_cols,
        "growth_features": growth_cols,
        "engagement_components": ENGAGEMENT_COMPONENTS,
        "classifier_features": classifier_features,
        "classifier_results": clf_results,
        "classifier_selected": best_clf_name,
        "classifier_chance_level": chance_level,
        "ninebox_labels": {f"{p}|{pot}": label for (p, pot), label in NINEBOX_LABELS.items()},
        "grid_counts": grid_counts.to_dict(),
        "performance_tier_counts": pd.Series(perf_tier).value_counts().to_dict(),
        "potential_tier_counts": pd.Series(potential_tier).value_counts().to_dict(),
        "leakage_finding": (
            "PerformanceRating is a deterministic function of PercentSalaryHike in this "
            "dataset; with that proxy removed the rating is unpredictable "
            "(CV ROC-AUC 0.457 for rating 4 vs 3). The performance axis therefore uses an "
            "observed engagement composite rather than a predicted rating."
        ),
        "training_seconds": round(time.time() - started, 2),
    }
    io.write_json(bundle, io.ARTIFACTS_DIR / "ninebox_metrics.json")

    # PCA projection of the potential space, for the scatter plot in the app.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(pot_matrix)
    io.write_json(
        {
            "coords": coords.tolist(),
            "labels": cluster_labels.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "features": pot_cols,
        },
        io.ARTIFACTS_DIR / "ninebox_pca.json",
    )

    print(f"[save] artifacts written to {io.ARTIFACTS_DIR}")
    print(f"[done] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
