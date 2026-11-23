"""Uji integrasi endpoint jagung terhadap PostgreSQL + PostGIS sungguhan.

Dilewati kecuali variabel lingkungan JAGUNG_POSTGIS_URL diisi, misalnya:

    $env:JAGUNG_POSTGIS_URL = "postgresql+psycopg://safrons:safrons@localhost:5432/safrons"
    pytest -q tests/test_jagung_postgis.py

Uji ini hanya membaca data. Angka sebaran kelas mengandaikan data hara_bogor
masih sama dengan seed CAPS-32 (migrasi 002) dan kriteria jagung-juknis2011-v1.
"""

import os
from collections import Counter
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import get_db
from app.main import app

POSTGIS_URL = os.environ.get("JAGUNG_POSTGIS_URL")

pytestmark = pytest.mark.skipif(
    not POSTGIS_URL, reason="JAGUNG_POSTGIS_URL tidak diisi; uji PostGIS dilewati."
)


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(str(POSTGIS_URL))
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
        engine.dispose()


def test_suitability_layer_has_geometry_area_and_expected_distribution(
    client: TestClient,
) -> None:
    body = client.get("/api/v1/jagung/suitability").json()
    features = body["features"]
    assert len(features) == 191
    assert all(f["geometry"] and f["geometry"]["type"] for f in features)

    evaluated = [f["properties"] for f in features if f["properties"]["status"] == "evaluated"]
    assert len(evaluated) == 186
    assert all(p["area_ha"] and p["area_ha"] > 0 for p in evaluated)
    assert Counter(p["actual_class"] for p in evaluated) == {"S2": 3, "S3": 49, "N": 134}
    assert Counter(p["potential_class"] for p in evaluated) == {"S2": 41, "S3": 12, "N": 133}


def test_unitname_is_read_for_tidal_swamp_drainage(client: TestClient) -> None:
    features = client.get("/api/v1/jagung/suitability").json()["features"]
    found = False
    for feature in features:
        area_id = feature["properties"]["id"]
        detail = client.get(f"/api/v1/jagung/areas/{area_id}").json()
        drainase = next(p for p in detail["parameters"] if p["parameter"] == "drainase")
        if drainase["raw_value"] and "Tidal Swamps" in drainase["raw_value"]:
            assert drainase["category"] == "sangat_terhambat"
            found = True
    assert found, "Poligon Tidal Swamps tidak ditemukan; kolom unitname tidak terbaca."


def test_ranking_is_ordered_and_uses_area(client: TestClient) -> None:
    body = client.get("/api/v1/jagung/ranking", params={"limit": 200}).json()
    assert body["total_candidates"] == 53
    items = body["items"]
    order = {"S1": 0, "S2": 1, "S3": 2, "N": 3}
    keys = [(order[i["potential_class"]], order[i["actual_class"]]) for i in items]
    assert keys == sorted(keys)
    assert all(i["area_ha"] and i["area_ha"] > 0 for i in items)
