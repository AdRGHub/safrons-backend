"""Uji unit evaluasi kesesuaian jagung (Tahap 2 SAFRONS Jagung)."""

import json
from collections import Counter
from collections.abc import Generator
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from jagung_fixtures import seed_rows
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.jagung_criteria import JagungCriterion
from app.services.jagung_suitability import (
    AreaValues,
    CriteriaError,
    CriterionRule,
    drainage_proxy,
    evaluate_area,
    load_criteria,
    rank_areas,
    texture_top,
    validate_criteria,
)

CRITERIA = seed_rows()
VERSION = "jagung-juknis2011-v1"
GEOJSON = (
    Path(__file__).resolve().parents[1] / "reference" / "data-hara-bogor" / "hara-bogor.geojson"
)

# Lahan datar ideal: semua parameter S1.
IDEAL = AreaValues(
    area_id=1,
    name="Lahan Uji",
    ph=6.2,
    n=5.0,
    p=9.5,
    k=400.0,
    slope="<2",
    texture_of="mod. fine/fine; mod. fine/fine",
    unitname="Plains",
    soil_great="Dystropepts; Tropudults",
)


def classes(values: AreaValues) -> tuple[str | None, str | None]:
    result = evaluate_area(values, CRITERIA)
    return result.actual_class, result.potential_class


def parameter(values: AreaValues, key: str) -> tuple[str | None, str | None]:
    result = evaluate_area(values, CRITERIA)
    item = next(p for p in result.parameters if p.parameter == key)
    return item.actual_class, item.potential_class


# ---------------------------------------------------------------------------
# Seed kriteria
# ---------------------------------------------------------------------------


def test_seed_has_one_version_and_all_parameters() -> None:
    assert len(CRITERIA) == 34
    assert validate_criteria(CRITERIA) == VERSION
    assert all(rule.source.startswith("Ritung dkk. 2011") for rule in CRITERIA)


def test_validate_criteria_rejects_empty_mixed_and_incomplete() -> None:
    with pytest.raises(CriteriaError):
        validate_criteria([])
    with pytest.raises(CriteriaError):
        validate_criteria([*CRITERIA, replace(CRITERIA[0], rule_set_version="lain")])
    with pytest.raises(CriteriaError):
        validate_criteria([r for r in CRITERIA if r.parameter != "drainase"])


# ---------------------------------------------------------------------------
# Normalisasi tekstur dan proksi drainase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texture_of", "expected"),
    [
        ("mod. fine/fine; mod. fine/mod. fine; medium/rock", "agak_halus"),
        ("fine/fine; mod.coarse/medium; mod. fine/fine", "halus"),
        ("mod. coarse/mod. coarse; fine/fine", "agak_kasar"),
        ("mod.coarse/medium; mod. fine/mod.fine", "agak_kasar"),
        ("medium mod. fine; mod. coarse/medium; mod. fine/fine", "sedang"),
        ("medium/mod. fine; fine/fine", "sedang"),
        ("Fine/Fine", "halus"),
        ("coarse/mod.coarse", "kasar"),
        ("No Data", None),
        ("", None),
        (None, None),
        ("peat/fine", None),
    ],
)
def test_texture_top(texture_of: str | None, expected: str | None) -> None:
    assert texture_top(texture_of) == expected


@pytest.mark.parametrize(
    ("unitname", "soil_great", "expected"),
    [
        ("Hills", "Dystropepts; Tropudults; Troporthents", "baik"),
        ("Alluvial Valleys", "Tropaquepts; Tropofluvents; Eutropepts", "terhambat"),
        ("Alluvial Plains", "Fluvaquents;Tropaquents; Tropohemists", "terhambat"),
        ("Beaches", "Tropopsamments; Tropaquents", "cepat"),
        ("Tidal Swamps", "Hydraquents; Sulfaquents", "sangat_terhambat"),
        ("Tidal Swamps", None, "sangat_terhambat"),
        ("Plains", "Paleudults; Tropaquepts; Dystropepts", "baik"),
        ("Fans and Lahars", "No Data", None),
        ("Plains", None, None),
    ],
)
def test_drainage_proxy(unitname: str, soil_great: str | None, expected: str | None) -> None:
    assert drainage_proxy(unitname, soil_great) == expected


# ---------------------------------------------------------------------------
# Kelas per parameter pada batas ambang
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5.8, "S1"),
        (7.8, "S1"),
        (5.79, "S2"),
        (5.5, "S2"),
        (7.81, "S2"),
        (8.2, "S2"),
        (5.49, "S3"),
        (4.4, "S3"),
        (8.21, "S3"),
    ],
)
def test_ph_boundaries(value: float, expected: str) -> None:
    assert parameter(replace(IDEAL, ph=value), "ph")[0] == expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        # Nilai tepat di batas masuk status bawah, sama dengan diagnosis hara lama.
        ("n", 2.450581, "S2"),
        ("n", 2.4506, "S1"),
        ("n", 1.989691, "S3"),
        ("n", 1.99, "S2"),
        ("p", 8.022388, "S2"),
        ("p", 8.0224, "S1"),
        ("p", 7.790811, "S3"),
        ("p", 7.8, "S2"),
        ("k", 331.24494, "S2"),
        ("k", 331.25, "S1"),
        ("k", 197.49262, "S3"),
        ("k", 200.0, "S2"),
    ],
)
def test_nutrient_boundaries(key: str, value: float, expected: str) -> None:
    assert parameter(replace(IDEAL, **{key: value}), key) == (expected, "S1")


@pytest.mark.parametrize(
    ("slope", "expected"),
    [
        ("<2", "S1"),
        ("0-8", "S2"),
        ("2-8", "S2"),
        ("9-15", "S3"),
        ("16-25", "N"),
        ("26-40", "N"),
        ("41-60", "N"),
        (">60", "N"),
    ],
)
def test_slope_mapping_is_permanent(slope: str, expected: str) -> None:
    assert parameter(replace(IDEAL, slope=slope), "lereng") == (expected, expected)


def test_drainage_potential_only_improves_excess_water() -> None:
    terhambat = replace(IDEAL, unitname="Alluvial Valleys", soil_great="Tropaquepts")
    cepat = replace(IDEAL, unitname="Beaches", soil_great="Tropopsamments; Tropaquents")
    swamp = replace(IDEAL, unitname="Tidal Swamps", soil_great="Hydraquents; Sulfaquents")
    assert parameter(terhambat, "drainase") == ("S3", "S2")
    assert parameter(cepat, "drainase") == ("N", "N")
    assert parameter(swamp, "drainase") == ("N", "S3")


def test_alkaline_ph_is_not_improved_by_liming() -> None:
    assert parameter(replace(IDEAL, ph=8.5), "ph") == ("S3", "S3")
    assert parameter(replace(IDEAL, ph=5.0), "ph") == ("S3", "S1")


# ---------------------------------------------------------------------------
# Kelas aktual dan potensial
# ---------------------------------------------------------------------------


def test_ideal_land_is_s1_without_limiting_factors() -> None:
    result = evaluate_area(IDEAL, CRITERIA)
    assert (result.status, result.actual_class, result.potential_class) == ("evaluated", "S1", "S1")
    assert result.limiting_factors == []
    assert result.improvable_factors == []
    assert result.rule_set_version == VERSION


def test_steep_slope_is_always_potential_n() -> None:
    result = evaluate_area(replace(IDEAL, slope="41-60", p=6.0, ph=5.0), CRITERIA)
    assert (result.actual_class, result.potential_class) == ("N", "N")
    assert result.limiting_factors == ["lereng"]
    assert result.potential_limiting_factors == ["lereng"]
    assert result.improvable_factors == ["ph", "p"]


def test_flat_land_with_low_p_improves_to_s1() -> None:
    result = evaluate_area(replace(IDEAL, p=6.0), CRITERIA)
    assert (result.actual_class, result.potential_class) == ("S3", "S1")
    assert result.limiting_factors == ["p"]
    assert result.improvable_factors == ["p"]


def test_wet_flat_land_improves_one_class() -> None:
    values = replace(IDEAL, n=1.5, unitname="Alluvial Valleys", soil_great="Tropaquepts")
    result = evaluate_area(values, CRITERIA)
    assert (result.actual_class, result.potential_class) == ("S3", "S2")
    assert result.limiting_factors == ["n", "drainase"]
    assert result.improvable_factors == ["n", "drainase"]
    assert result.potential_limiting_factors == ["drainase"]
    assert result.summary == (
        "Kelas aktual S3 (sesuai marginal), dibatasi oleh Nitrogen (N), Drainase (perkiraan). "
        "Kelas potensial S2 (cukup sesuai) setelah perbaikan; pembatas yang tersisa: "
        "Drainase (perkiraan)."
    )
    assert "permanen" not in result.summary


def test_coarse_texture_limits_potential() -> None:
    result = evaluate_area(replace(IDEAL, texture_of="mod. coarse/mod. coarse"), CRITERIA)
    assert (result.actual_class, result.potential_class) == ("S3", "S3")
    assert result.improvable_factors == []


# ---------------------------------------------------------------------------
# Data tidak cukup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "missing"),
    [
        ({"ph": -9999.0}, ["ph"]),
        ({"n": None}, ["n"]),
        ({"slope": None}, ["lereng"]),
        ({"slope": "abc"}, ["lereng"]),
        ({"texture_of": "No Data", "soil_great": "No Data"}, ["tekstur", "drainase"]),
    ],
)
def test_missing_or_unknown_data_is_insufficient(changes: dict, missing: list[str]) -> None:
    result = evaluate_area(replace(IDEAL, **changes), CRITERIA)
    assert result.status == "insufficient_data"
    assert (result.actual_class, result.potential_class) == (None, None)
    assert result.missing_parameters == missing


def test_water_polygon_is_insufficient() -> None:
    result = evaluate_area(replace(IDEAL, name="Water"), CRITERIA)
    assert result.status == "insufficient_data"


# ---------------------------------------------------------------------------
# Peringkat
# ---------------------------------------------------------------------------


def _evaluations() -> list:
    areas = [
        replace(IDEAL, area_id=1),  # S1 / S1
        replace(IDEAL, area_id=2, p=6.0),  # S3 / S1, 1 faktor
        replace(IDEAL, area_id=3, p=6.0, n=1.5),  # S3 / S1, 2 faktor
        replace(IDEAL, area_id=4, slope="0-8"),  # S2 / S2
        replace(IDEAL, area_id=5, slope="9-15"),  # S3 / S3
        replace(IDEAL, area_id=6, slope="41-60"),  # N / N
        replace(IDEAL, area_id=7, ph=-9999.0),  # tidak cukup data
        replace(IDEAL, area_id=8, p=6.0),  # S3 / S1, 1 faktor, luas lebih besar
    ]
    return [evaluate_area(a, CRITERIA) for a in areas]


def test_rank_order_and_tie_breakers() -> None:
    ranking = rank_areas(_evaluations(), area_ha={2: 10.0, 8: 25.0, 3: 99.0})
    assert [item.area_id for item in ranking] == [1, 8, 2, 3, 4, 5, 6]
    assert [item.rank for item in ranking] == list(range(1, 8))


def test_rank_filters_min_class_and_limit() -> None:
    ranking = rank_areas(_evaluations(), min_class="S2")
    assert {item.potential_class for item in ranking} <= {"S1", "S2"}
    assert 6 not in [item.area_id for item in ranking]
    assert len(rank_areas(_evaluations(), limit=2)) == 2


def test_rank_excludes_insufficient_and_ignores_cluster_for_order() -> None:
    plain = rank_areas(_evaluations())
    clustered = rank_areas(_evaluations(), cluster_ids={1: 2, 8: 0})
    assert 7 not in [item.area_id for item in plain]
    assert [i.area_id for i in plain] == [i.area_id for i in clustered]
    assert clustered[0].cluster_id == 2


# ---------------------------------------------------------------------------
# Pemuat kriteria dari basis data (SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _model(
    rule: CriterionRule, *, active: bool = True, version: str | None = None
) -> JagungCriterion:
    return JagungCriterion(
        rule_set_version=version or rule.rule_set_version,
        parameter=rule.parameter,
        suitability_class=rule.suitability_class,
        rule_key=rule.rule_key,
        value_type=rule.value_type,
        min_value=None if rule.min_value is None else Decimal(str(rule.min_value)),
        max_value=None if rule.max_value is None else Decimal(str(rule.max_value)),
        lower_inclusive=rule.lower_inclusive,
        upper_inclusive=rule.upper_inclusive,
        category=rule.category,
        improvable=rule.improvable,
        improvement_step=rule.improvement_step,
        source=rule.source,
        is_active=active,
    )


def test_load_criteria_reads_only_active_rows(db: Session) -> None:
    db.add_all(_model(rule) for rule in CRITERIA)
    db.add_all(_model(rule, active=False, version="versi-lama") for rule in CRITERIA)
    db.commit()
    loaded = load_criteria(db)
    assert len(loaded) == 34
    assert evaluate_area(replace(IDEAL, p=6.0), loaded).actual_class == "S3"


def test_load_criteria_fails_loudly_without_rows(db: Session) -> None:
    with pytest.raises(CriteriaError):
        load_criteria(db)


# ---------------------------------------------------------------------------
# Regresi pada atribut data asli hara-bogor (191 poligon)
# ---------------------------------------------------------------------------


def test_real_dataset_distribution() -> None:
    features = json.loads(GEOJSON.read_text(encoding="utf-8"))["features"]
    results = []
    for index, feature in enumerate(features, start=1):
        p = feature["properties"]
        results.append(
            evaluate_area(
                AreaValues(
                    area_id=index,
                    name=p["NAME"],
                    ph=p["pH_rata2"],
                    n=p["N_rata2"],
                    p=p["P_rata2"],
                    k=p["K_rata2"],
                    slope=p["SLOPE__"],
                    texture_of=p["TEXTURE_OF"],
                    unitname=p["UNITNAME"],
                    soil_great=p["SOIL_GREAT"],
                ),
                CRITERIA,
            )
        )
    evaluated = [r for r in results if r.status == "evaluated"]
    assert len(results) - len(evaluated) == 5
    assert Counter(r.actual_class for r in evaluated) == {"S2": 3, "S3": 49, "N": 134}
    assert Counter(r.potential_class for r in evaluated) == {"S2": 41, "S3": 12, "N": 133}
    assert len(rank_areas(evaluated, min_class="S3")) == 53
