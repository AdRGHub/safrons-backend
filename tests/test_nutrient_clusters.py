"""Uji pelatihan tipe profil hara k-means (Tahap 4 SAFRONS Jagung)."""

import json
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.hara import HaraArea
from app.models.nutrient_cluster import (
    ClusterFertilizerPackage,
    HaraClusterAssignment,
    MlModelRun,
    NutrientCluster,
)
from app.services.nutrient_clusters import (
    NutrientSample,
    TrainingError,
    acidity_label,
    draft_package,
    feature_matrix,
    load_samples,
    model_parameters,
    nutrient_level,
    persist_training,
    profile_name,
    save_artifacts,
    train_clusters,
)

REPO = Path(__file__).resolve().parents[1]
GEOJSON = REPO / "reference" / "data-hara-bogor" / "hara-bogor.geojson"
ARTIFACTS = REPO / "ml" / "artifacts"


def _dataset_rows() -> list[dict]:
    features = json.loads(GEOJSON.read_text(encoding="utf-8"))["features"]
    return [{"id": i, **f["properties"]} for i, f in enumerate(features, start=1)]


def _samples() -> list[NutrientSample]:
    return [
        NutrientSample(
            area_id=r["id"], ph=r["pH_rata2"], n=r["N_rata2"], p=r["P_rata2"], k=r["K_rata2"]
        )
        for r in _dataset_rows()
        if r["pH_rata2"] not in (None, -9999) and r["NAME"] not in ("Water", "No Data")
    ]


@pytest.fixture(scope="module")
def trained():
    return train_clusters(_samples(), model_version="uji-v1")


# ---------------------------------------------------------------------------
# Fitur dan penamaan profil
# ---------------------------------------------------------------------------


def test_feature_matrix_uses_log_for_npk_only() -> None:
    matrix = feature_matrix([NutrientSample(area_id=1, ph=5.5, n=3.0, p=8.0, k=250.0)])
    assert matrix.tolist() == [[5.5, np.log1p(3.0), np.log1p(8.0), np.log1p(250.0)]]


@pytest.mark.parametrize(
    ("ph", "label"),
    [(4.77, "sangat masam"), (5.0, "masam"), (5.49, "masam"), (5.5, "agak masam"), (6.6, "netral")],
)
def test_acidity_label(ph: float, label: str) -> None:
    assert acidity_label(ph) == label


def test_nutrient_level_uses_ratio_to_dataset_median() -> None:
    assert nutrient_level(2.0, 3.55) == "rendah"
    assert nutrient_level(3.55, 3.55) == "sedang"
    assert nutrient_level(5.0, 3.55) == "tinggi"


def test_profile_name_lists_only_non_medium_nutrients() -> None:
    levels = {"n": "rendah", "p": "sedang", "k": "tinggi"}
    assert profile_name(levels, "agak masam") == "N rendah, K tinggi, agak masam"
    assert profile_name({"n": "sedang", "p": "sedang", "k": "sedang"}, "masam") == (
        "N, P, K sedang, masam"
    )


# ---------------------------------------------------------------------------
# Pelatihan pada data asli
# ---------------------------------------------------------------------------


def test_training_on_real_data_matches_feasibility_study(trained) -> None:
    metrics = trained.metrics
    assert metrics["n_samples"] == 188
    assert metrics["n_unique"] == 41
    assert round(metrics["silhouette"], 2) == 0.37
    assert metrics["ari_mean"] == pytest.approx(1.0)
    assert metrics["cluster_sizes"] == [94, 79, 15]
    assert [p.profile_name for p in trained.profiles] == [
        "N tinggi, K rendah, masam",
        "N rendah, K tinggi, agak masam",
        "P tinggi, K tinggi, sangat masam",
    ]


def test_profile_names_are_not_fertility_levels(trained) -> None:
    forbidden = ("subur", "optimal", "defisit", "zona", "baik", "buruk")
    for profile in trained.profiles:
        assert not any(word in profile.profile_name.lower() for word in forbidden)


def test_cluster_ids_are_canonical_and_predict_consistently(trained) -> None:
    samples = _samples()
    scaled = trained.scaler.transform(feature_matrix(samples))
    predicted = trained.kmeans.predict(scaled)
    assert predicted.tolist() == [trained.labels[s.area_id] for s in samples]
    sizes = [p.n_areas for p in trained.profiles]
    assert sizes == sorted(sizes, reverse=True)


def test_training_is_deterministic(trained) -> None:
    again = train_clusters(_samples(), model_version="uji-v2")
    assert again.labels == trained.labels


def test_crosstab_covers_every_sample(trained) -> None:
    for nutrient in ("n", "p", "k"):
        table = trained.metrics["crosstab_rule_based_status"][nutrient]
        assert sum(sum(counts.values()) for counts in table.values()) == 188


def test_training_rejects_bad_parameters() -> None:
    with pytest.raises(TrainingError):
        train_clusters(_samples(), model_version="x", k=1)
    duplicate = [NutrientSample(area_id=i, ph=5.0, n=3.0, p=8.0, k=200.0) for i in range(5)]
    with pytest.raises(TrainingError):
        train_clusters(duplicate, model_version="x", k=3)


def test_draft_package_is_marked_and_points_to_fertilizer(trained) -> None:
    title, content = draft_package(trained.profiles[0])
    assert title == "Paket pemupukan: N tinggi, K rendah, masam"
    assert content.startswith("DRAF")
    assert "prioritaskan KCl" in content
    assert "kurangi atau tunda Urea" in content
    assert "pengapuran" in content


# ---------------------------------------------------------------------------
# Artefak
# ---------------------------------------------------------------------------


def _nearest_centroid(parameters: dict, samples: list[NutrientSample]) -> list[int]:
    scaled = (feature_matrix(samples) - np.array(parameters["scaler_mean"])) / np.array(
        parameters["scaler_scale"]
    )
    centroids = np.array(parameters["centroids"])
    distances = ((scaled[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).tolist()


def test_save_artifacts_writes_loadable_models(trained, tmp_path: Path) -> None:
    directory = save_artifacts(trained, tmp_path / "uji-v1")
    assert sorted(p.name for p in directory.iterdir()) == [
        "kmeans.joblib",
        "meta.json",
        "scaler.joblib",
    ]
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == "uji-v1"
    assert len(meta["profiles"]) == 3
    kmeans = joblib.load(directory / "kmeans.joblib")
    scaler = joblib.load(directory / "scaler.joblib")
    samples = _samples()
    assert kmeans.predict(scaler.transform(feature_matrix(samples))).tolist() == [
        trained.labels[s.area_id] for s in samples
    ]


def test_json_parameters_predict_without_pickle(trained) -> None:
    samples = _samples()
    assert _nearest_centroid(model_parameters(trained), samples) == [
        trained.labels[s.area_id] for s in samples
    ]


def test_committed_artifacts_match_current_code(trained) -> None:
    for folder in ("kmeans-v1", "current"):
        meta = json.loads((ARTIFACTS / folder / "meta.json").read_text(encoding="utf-8"))
        assert meta["model_version"] == "kmeans-v1"
        stored, fresh = meta["parameters"], model_parameters(trained)
        for key in ("scaler_mean", "scaler_scale", "centroids"):
            assert np.allclose(stored[key], fresh[key], atol=1e-6)


# ---------------------------------------------------------------------------
# Basis data (SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    for r in _dataset_rows():
        session.add(
            HaraArea(
                id=r["id"],
                name=r["NAME"],
                ph_rata2=Decimal(str(r["pH_rata2"])),
                n_rata2=Decimal(str(r["N_rata2"])),
                p_rata2=Decimal(str(r["P_rata2"])),
                k_rata2=Decimal(str(r["K_rata2"])),
            )
        )
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _count(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model))


def test_load_samples_skips_no_data(db: Session) -> None:
    assert len(load_samples(db)) == 188


def test_persist_training_writes_all_tables(db: Session, trained) -> None:
    persist_training(db, trained, artifact_path="ml/artifacts/uji-v1")
    db.commit()
    run = db.scalar(select(MlModelRun))
    assert run.is_active and run.model_version == "uji-v1"
    assert float(run.silhouette) == pytest.approx(trained.silhouette)
    assert _count(db, NutrientCluster) == 3
    assert _count(db, HaraClusterAssignment) == 188
    packages = db.scalars(select(ClusterFertilizerPackage)).all()
    assert len(packages) == 3 and all(p.is_draft for p in packages)


def test_new_version_deactivates_previous_and_duplicates_fail(db: Session, trained) -> None:
    persist_training(db, trained, artifact_path="ml/artifacts/uji-v1")
    db.commit()
    second = train_clusters(load_samples(db), model_version="uji-v2")
    persist_training(db, second, artifact_path="ml/artifacts/uji-v2")
    db.commit()
    active = db.scalars(select(MlModelRun.model_version).where(MlModelRun.is_active)).all()
    assert active == ["uji-v2"]
    with pytest.raises(TrainingError):
        persist_training(db, trained, artifact_path="ml/artifacts/uji-v1")
