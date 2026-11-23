"""Uji endpoint /api/v1/jagung (Tahap 3 SAFRONS Jagung) dengan SQLite."""

from collections.abc import Callable, Generator
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from jagung_fixtures import seed_rows
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.hara import HaraArea
from app.models.jagung_criteria import JagungCriterion

DRY = "Dystropepts; Tropudults; Troporthents"
WET = "Tropaquepts; Tropofluvents; Eutropepts"
FINE = "mod. fine/fine; mod. fine/fine"


def _area(area_id: int, name: str, ph: str, n: str, p: str, k: str, **extra: str) -> HaraArea:
    return HaraArea(
        id=area_id,
        name=name,
        ph_rata2=Decimal(ph),
        n_rata2=Decimal(n),
        p_rata2=Decimal(p),
        k_rata2=Decimal(k),
        slope__=extra.get("slope", "<2"),
        texture_of=extra.get("texture", FINE),
        soil_great=extra.get("soil", DRY),
    )


def _areas() -> list[HaraArea]:
    """Objek baru tiap fixture; objek ORM tidak boleh dipakai ulang lintas sesi."""
    return [
        _area(1, "Datar Subur", "6.2", "5.0", "9.5", "400"),  # S1 / S1
        _area(2, "Datar P Rendah", "6.2", "5.0", "6.0", "400"),  # S3 / S1
        _area(3, "Landai", "6.2", "5.0", "9.5", "400", slope="0-8"),  # S2 / S2
        _area(4, "Basah", "6.2", "1.5", "9.5", "400", soil=WET),  # S3 / S2
        _area(5, "Curam", "6.2", "5.0", "9.5", "400", slope="41-60"),  # N / N
        _area(6, "Water", "-9999", "-9999", "-9999", "-9999", slope="", texture="No Data"),
    ]


def _criteria_models() -> list[JagungCriterion]:
    return [
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
        for r in seed_rows()
    ]


def _client(with_criteria: bool) -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    session_factory: Callable[[], Session] = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    Base.metadata.create_all(bind=engine)
    with session_factory() as db:
        db.add_all(_areas())
        if with_criteria:
            db.add_all(_criteria_models())
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    yield from _client(with_criteria=True)


@pytest.fixture()
def client_without_criteria() -> Generator[TestClient, None, None]:
    yield from _client(with_criteria=False)


# ---------------------------------------------------------------------------
# GET /api/v1/jagung/areas/{id}
# ---------------------------------------------------------------------------


def test_area_returns_actual_and_potential_class(client: TestClient) -> None:
    response = client.get("/api/v1/jagung/areas/2")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    body = response.json()
    assert body["area_id"] == 2
    assert body["status"] == "evaluated"
    assert body["rule_set_version"] == "jagung-juknis2011-v1"
    assert (body["actual_class"], body["potential_class"]) == ("S3", "S1")
    assert body["limiting_factors"] == ["p"]
    assert body["improvable_factors"] == ["p"]
    assert [p["parameter"] for p in body["parameters"]] == [
        "ph",
        "n",
        "p",
        "k",
        "lereng",
        "tekstur",
        "drainase",
    ]
    assert body["area_ha"] is None  # luas hanya tersedia di PostGIS


def test_area_with_wet_soil_improves_drainage_one_class(client: TestClient) -> None:
    body = client.get("/api/v1/jagung/areas/4").json()
    assert (body["actual_class"], body["potential_class"]) == ("S3", "S2")
    assert body["potential_limiting_factors"] == ["drainase"]


def test_area_without_data_is_insufficient(client: TestClient) -> None:
    body = client.get("/api/v1/jagung/areas/6").json()
    assert body["status"] == "insufficient_data"
    assert body["actual_class"] is None


def test_unknown_area_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/jagung/areas/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Area hara tidak ditemukan."


def test_missing_criteria_returns_500_with_reason(client_without_criteria: TestClient) -> None:
    response = client_without_criteria.get("/api/v1/jagung/areas/1")
    assert response.status_code == 500
    assert "Kriteria jagung tidak valid" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/v1/jagung/ranking
# ---------------------------------------------------------------------------


def test_ranking_default_excludes_n_and_no_data(client: TestClient) -> None:
    response = client.get("/api/v1/jagung/ranking")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"
    body = response.json()
    assert body["min_class"] == "S3"
    assert body["total_candidates"] == 4
    assert [item["area_id"] for item in body["items"]] == [1, 2, 3, 4]
    assert [item["rank"] for item in body["items"]] == [1, 2, 3, 4]
    assert all(item["cluster_id"] is None for item in body["items"])


def test_ranking_filter_and_limit(client: TestClient) -> None:
    body = client.get("/api/v1/jagung/ranking", params={"min_class": "S1", "limit": 1}).json()
    assert body["total_candidates"] == 2
    assert [item["area_id"] for item in body["items"]] == [1]
    everything = client.get("/api/v1/jagung/ranking", params={"min_class": "N"}).json()
    assert everything["total_candidates"] == 5


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"min_class": "S4"}])
def test_ranking_rejects_invalid_query(client: TestClient, params: dict) -> None:
    assert client.get("/api/v1/jagung/ranking", params=params).status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/jagung/suitability
# ---------------------------------------------------------------------------


def test_suitability_layer_covers_every_polygon(client: TestClient) -> None:
    response = client.get("/api/v1/jagung/suitability")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert body["rule_set_version"] == "jagung-juknis2011-v1"
    properties = {f["properties"]["id"]: f["properties"] for f in body["features"]}
    assert sorted(properties) == [1, 2, 3, 4, 5, 6]
    assert properties[5]["potential_class"] == "N"
    assert properties[6]["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Endpoint lama tidak berubah
# ---------------------------------------------------------------------------


def test_existing_diagnosis_still_returns_old_crop_list(client: TestClient) -> None:
    body = client.get("/api/v1/hara/areas/1/diagnosis").json()
    crops = {item["crop"] for item in body["crop_suitabilities"]}
    assert {"jagung", "kacang_tanah", "kakao"} <= crops
