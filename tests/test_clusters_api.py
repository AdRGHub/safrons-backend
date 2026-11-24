"""Uji API tipe profil hara dan integrasinya (Tahap 5 SAFRONS Jagung)."""

import json
from collections.abc import Generator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jagung_fixtures import seed_rows
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import create_access_token
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.hara import HaraArea
from app.models.hara_area_change import HaraAreaChange
from app.models.jagung_criteria import JagungCriterion
from app.models.nutrient_cluster import HaraClusterAssignment, MlModelRun
from app.models.user import User
from app.services.nutrient_clusters import (
    ClusterModelError,
    clear_model_cache,
    load_samples,
    persist_training,
    predict_cluster,
    run_parameters,
    train_clusters,
)

REPO = Path(__file__).resolve().parents[1]
GEOJSON = REPO / "reference" / "data-hara-bogor" / "hara-bogor.geojson"
VERSION = "uji-v1"


@dataclass
class Env:
    client: TestClient
    session: sessionmaker
    tokens: dict[str, str]
    users: dict[str, int]

    def headers(self, role: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens[role]}"}


def _seed(db: Session) -> dict[str, int]:
    features = json.loads(GEOJSON.read_text(encoding="utf-8"))["features"]
    for index, feature in enumerate(features, start=1):
        p = feature["properties"]
        db.add(
            HaraArea(
                id=index,
                name=p["NAME"],
                ph_rata2=Decimal(str(p["pH_rata2"])),
                n_rata2=Decimal(str(p["N_rata2"])),
                p_rata2=Decimal(str(p["P_rata2"])),
                k_rata2=Decimal(str(p["K_rata2"])),
                slope__=p["SLOPE__"],
                texture_of=p["TEXTURE_OF"],
                soil_great=p["SOIL_GREAT"],
            )
        )
    for r in seed_rows():
        db.add(
            JagungCriterion(
                rule_set_version=r.rule_set_version,
                parameter=r.parameter,
                suitability_class=r.suitability_class,
                rule_key=r.rule_key,
                value_type=r.value_type,
                min_value=None if r.min_value is None else Decimal(str(r.min_value)),
                max_value=None if r.max_value is None else Decimal(str(r.max_value)),
                lower_inclusive=r.lower_inclusive,
                upper_inclusive=r.upper_inclusive,
                category=r.category,
                improvable=r.improvable,
                improvement_step=r.improvement_step,
                source=r.source,
            )
        )
    users = {}
    for role in ("admin", "expert", "user"):
        user = User(email=f"{role}@uji.test", full_name=role, hashed_password="x", role=role)
        db.add(user)
        db.flush()
        users[role] = user.id
    db.commit()
    return users


def _environment(with_model: bool, tmp_path: Path, monkeypatch) -> Generator[Env, None, None]:
    monkeypatch.setenv("CLUSTER_ARTIFACT_ROOT", str(tmp_path))
    clear_model_cache()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    with factory() as db:
        users = _seed(db)
        if with_model:
            result = train_clusters(load_samples(db), model_version=VERSION)
            persist_training(db, result, artifact_path=f"ml/artifacts/{VERSION}")
            db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield Env(
            client=TestClient(app),
            session=factory,
            tokens={role: create_access_token(str(uid)) for role, uid in users.items()},
            users=users,
        )
    finally:
        app.dependency_overrides.clear()
        clear_model_cache()


@pytest.fixture()
def env(tmp_path: Path, monkeypatch) -> Generator[Env, None, None]:
    yield from _environment(True, tmp_path, monkeypatch)


@pytest.fixture()
def env_without_model(tmp_path: Path, monkeypatch) -> Generator[Env, None, None]:
    yield from _environment(False, tmp_path, monkeypatch)


def _assignment(env: Env, area_id: int, version: str = VERSION) -> int | None:
    with env.session() as db:
        row = db.get(HaraClusterAssignment, (area_id, version))
        return None if row is None else row.cluster_id


# ---------------------------------------------------------------------------
# GET /api/v1/clusters
# ---------------------------------------------------------------------------


def test_clusters_lists_profiles_and_draft_packages(env: Env) -> None:
    response = env.client.get("/api/v1/clusters")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == VERSION
    assert round(body["silhouette"], 2) == 0.37
    assert [c["profile_name"] for c in body["clusters"]] == [
        "N tinggi, K rendah, masam",
        "N rendah, K tinggi, agak masam",
        "P tinggi, K tinggi, sangat masam",
    ]
    assert [c["n_areas"] for c in body["clusters"]] == [94, 79, 15]
    assert all(c["package"]["is_draft"] for c in body["clusters"])
    assert "bukan tingkat kesuburan" in body["note"]


def test_clusters_without_model_is_empty(env_without_model: Env) -> None:
    body = env_without_model.client.get("/api/v1/clusters").json()
    assert body["model_version"] is None
    assert body["clusters"] == []


# ---------------------------------------------------------------------------
# Integrasi dengan endpoint jagung
# ---------------------------------------------------------------------------


def test_jagung_area_includes_nutrient_profile(env: Env) -> None:
    body = env.client.get("/api/v1/jagung/areas/10").json()
    profile = body["nutrient_profile"]
    assert profile["model_version"] == VERSION
    assert profile["cluster_id"] == _assignment(env, 10)
    assert profile["package"]["title"].startswith("Paket pemupukan:")


def test_no_data_area_has_no_profile(env: Env) -> None:
    water = next(
        i
        for i, f in enumerate(json.loads(GEOJSON.read_text())["features"], start=1)
        if f["properties"]["NAME"] == "Water"
    )
    assert env.client.get(f"/api/v1/jagung/areas/{water}").json()["nutrient_profile"] is None


def test_ranking_order_is_independent_of_clusters(tmp_path: Path, monkeypatch) -> None:
    orders = []
    for with_model in (True, False):
        generator = _environment(with_model, tmp_path / str(with_model), monkeypatch)
        current = next(generator)
        body = current.client.get("/api/v1/jagung/ranking", params={"limit": 200}).json()
        orders.append([item["area_id"] for item in body["items"]])
        generator.close()
    assert orders[0] == orders[1]


def test_suitability_layer_carries_cluster_id(env: Env) -> None:
    features = env.client.get("/api/v1/jagung/suitability").json()["features"]
    evaluated = [f["properties"] for f in features if f["properties"]["status"] == "evaluated"]
    assert all(p["cluster_id"] is not None for p in evaluated)


# ---------------------------------------------------------------------------
# PATCH /api/v1/expert/clusters/{id}/package
# ---------------------------------------------------------------------------

PACKAGE = {
    "title": "Paket K rendah",
    "content": "KCl prioritas; dolomit 1-2 t/ha bila pH < 5,5. Divalidasi pakar.",
}


def test_package_update_requires_expert(env: Env) -> None:
    url = "/api/v1/expert/clusters/0/package"
    assert env.client.patch(url, json=PACKAGE).status_code in {401, 403}
    assert env.client.patch(url, json=PACKAGE, headers=env.headers("user")).status_code == 403


def test_expert_validates_package(env: Env) -> None:
    response = env.client.patch(
        "/api/v1/expert/clusters/0/package", json=PACKAGE, headers=env.headers("expert")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_draft"] is False
    assert body["updated_by"] == env.users["expert"]
    listed = env.client.get("/api/v1/clusters").json()["clusters"][0]["package"]
    assert listed["title"] == "Paket K rendah"


def test_package_update_validation_and_unknown_cluster(env: Env) -> None:
    headers = env.headers("expert")
    short = {"title": "ab", "content": "terlalu pendek"}
    assert (
        env.client.patch(
            "/api/v1/expert/clusters/0/package", json=short, headers=headers
        ).status_code
        == 422
    )
    missing = env.client.patch("/api/v1/expert/clusters/9/package", json=PACKAGE, headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Klaster tidak ditemukan pada model aktif."


# ---------------------------------------------------------------------------
# F-15: klaster diperbarui saat expert mengubah atau membuat area
# ---------------------------------------------------------------------------


def _area_in_cluster(env: Env, cluster_id: int) -> int:
    with env.session() as db:
        return db.scalar(
            select(HaraClusterAssignment.hara_area_id)
            .where(HaraClusterAssignment.cluster_id == cluster_id)
            .order_by(HaraClusterAssignment.hara_area_id)
        )


def test_expert_patch_reassigns_cluster(env: Env) -> None:
    area_id = _area_in_cluster(env, 1)  # N rendah, K tinggi
    response = env.client.patch(
        f"/api/v1/expert/hara/areas/{area_id}",
        json={"n_rata2": 5.2, "k_rata2": 158.0, "ph_rata2": 5.2},
        headers=env.headers("expert"),
    )
    assert response.status_code == 200
    with env.session() as db:
        run = db.scalar(select(MlModelRun).where(MlModelRun.is_active))
        expected = predict_cluster(
            run_parameters(run), 5.2, 5.2, float(db.get(HaraArea, area_id).p_rata2), 158.0
        )
        audit = db.scalars(
            select(HaraAreaChange).where(HaraAreaChange.hara_area_id == area_id)
        ).all()
    assert expected == 0
    assert _assignment(env, area_id) == 0
    assert len(audit) == 1


def test_expert_patch_to_no_data_removes_assignment(env: Env) -> None:
    area_id = _area_in_cluster(env, 0)
    response = env.client.patch(
        f"/api/v1/expert/hara/areas/{area_id}",
        json={"ph_rata2": None},
        headers=env.headers("expert"),
    )
    assert response.status_code == 200
    assert _assignment(env, area_id) is None
    body = env.client.get(f"/api/v1/jagung/areas/{area_id}").json()
    assert body["nutrient_profile"] is None


def test_expert_created_area_gets_cluster(env: Env) -> None:
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[106.8, -6.6], [106.81, -6.6], [106.81, -6.59], [106.8, -6.6]]],
        },
        "properties": {
            "name": "Lahan Baru",
            "ph_rata2": 4.8,
            "n_rata2": 3.5,
            "p_rata2": 36.0,
            "k_rata2": 384.0,
        },
    }
    response = env.client.post(
        "/api/v1/expert/hara/areas", json=payload, headers=env.headers("expert")
    )
    assert response.status_code == 201
    new_id = response.json()["properties"]["id"]
    assert _assignment(env, new_id) == 2  # P tinggi, K tinggi, sangat masam


def test_patch_without_model_still_works(env_without_model: Env) -> None:
    response = env_without_model.client.patch(
        "/api/v1/expert/hara/areas/10",
        json={"n_rata2": 4.0},
        headers=env_without_model.headers("expert"),
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# F-16: retrain dan riwayat (admin)
# ---------------------------------------------------------------------------


def test_retrain_and_runs_are_admin_only(env: Env) -> None:
    for role in ("user", "expert"):
        headers = env.headers(role)
        assert env.client.post("/api/v1/admin/clusters/retrain", headers=headers).status_code == 403
        assert env.client.get("/api/v1/admin/clusters/runs", headers=headers).status_code == 403


def test_retrain_activates_new_version_and_keeps_validated_packages(
    env: Env, tmp_path: Path
) -> None:
    env.client.patch(
        "/api/v1/expert/clusters/0/package", json=PACKAGE, headers=env.headers("expert")
    )
    response = env.client.post("/api/v1/admin/clusters/retrain", headers=env.headers("admin"))
    assert response.status_code == 201
    run = response.json()
    assert run["is_active"] is True
    assert run["triggered_by"] == env.users["admin"]
    assert run["cluster_sizes"] == [94, 79, 15]
    assert (tmp_path / run["artifact_path"] / "meta.json").exists()

    clusters = env.client.get("/api/v1/clusters").json()
    assert clusters["model_version"] == run["model_version"]
    first = clusters["clusters"][0]["package"]
    assert (first["title"], first["is_draft"]) == ("Paket K rendah", False)
    assert clusters["clusters"][1]["package"]["is_draft"] is True

    runs = env.client.get("/api/v1/admin/clusters/runs", headers=env.headers("admin")).json()
    assert [r["is_active"] for r in runs] == [True, False]
    assert _assignment(env, 10, run["model_version"]) == _assignment(env, 10)


def test_retrain_rejects_invalid_parameters(env: Env) -> None:
    response = env.client.post(
        "/api/v1/admin/clusters/retrain", json={"k": 1}, headers=env.headers("admin")
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Parameter model dari meta.json (versi Tahap 4 tanpa parameter di DB)
# ---------------------------------------------------------------------------


def _legacy_run(artifact_path: str) -> MlModelRun:
    return MlModelRun(
        model_version="kmeans-v1",
        k=3,
        n_init=50,
        random_state=42,
        features="x",
        n_samples=188,
        n_unique=41,
        metrics={},
        artifact_path=artifact_path,
    )


def test_parameters_fall_back_to_committed_meta_json(monkeypatch) -> None:
    monkeypatch.setenv("CLUSTER_ARTIFACT_ROOT", str(REPO))
    clear_model_cache()
    parameters = run_parameters(_legacy_run("ml/artifacts/kmeans-v1"))
    assert len(parameters["centroids"]) == 3
    clear_model_cache()


def test_missing_parameters_raise_clear_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLUSTER_ARTIFACT_ROOT", str(tmp_path))
    clear_model_cache()
    with pytest.raises(ClusterModelError):
        run_parameters(_legacy_run("ml/artifacts/tidak-ada"))
