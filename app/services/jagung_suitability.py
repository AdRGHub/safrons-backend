"""Evaluasi kesesuaian lahan jagung (SAFRONS Jagung).

Logika ini sengaja dibuat sebagai fungsi Python murni: ``evaluate_area`` dan
``rank_areas`` tidak menyentuh basis data, sehingga dapat diuji tanpa PostGIS.
Ambang kelas tidak ditulis di sini; semuanya dibaca dari tabel
``jagung_criteria`` (migrasi 008) lewat ``load_criteria``.

Aturan:
* Kelas per parameter: S1/S2/S3/N menurut baris kriteria yang cocok.
* Kelas aktual: kelas terburuk antarparameter (hukum minimum Liebig).
* Kelas potensial: tiap parameter dinaikkan menurut ``improvement_step`` baris
  kriteria yang cocok (``ke_S1`` atau ``naik_satu_kelas``); parameter permanen
  tetap. Kelas potensial = kelas terburuk setelah perbaikan.
* Poligon tanpa data (nama Water/No Data, nilai -9999, atau kategori yang tidak
  dikenali) berstatus ``insufficient_data`` dan tidak masuk peringkat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.hara import HaraArea
from app.models.jagung_criteria import JagungCriterion
from app.schemas.jagung import (
    ImprovementStep,
    JagungSuitabilityRead,
    ParameterClass,
    ParameterKey,
    RankingItem,
    SuitabilityClass,
)

SUITABILITY_ORDER: tuple[SuitabilityClass, ...] = ("S1", "S2", "S3", "N")
NUMERIC_PARAMETERS: tuple[ParameterKey, ...] = ("ph", "n", "p", "k")
CATEGORY_PARAMETERS: tuple[ParameterKey, ...] = ("lereng", "tekstur", "drainase")
PARAMETER_ORDER: tuple[ParameterKey, ...] = NUMERIC_PARAMETERS + CATEGORY_PARAMETERS

PARAMETER_LABELS: dict[ParameterKey, str] = {
    "ph": "pH tanah",
    "n": "Nitrogen (N)",
    "p": "Fosfor (P)",
    "k": "Kalium (K)",
    "lereng": "Kemiringan lereng",
    "tekstur": "Tekstur tanah lapisan atas",
    "drainase": "Drainase (perkiraan)",
}

CLASS_LABELS: dict[SuitabilityClass, str] = {
    "S1": "sangat sesuai",
    "S2": "cukup sesuai",
    "S3": "sesuai marginal",
    "N": "tidak sesuai",
}

NO_DATA_VALUE = -9999
NO_DATA_NAMES = {"water", "no data"}

# Kelas tekstur USDA berbahasa Inggris di TEXTURE_OF -> kategori di jagung_criteria.
TEXTURE_CATEGORIES: dict[str, str] = {
    "mod. coarse": "agak_kasar",
    "mod. fine": "agak_halus",
    "coarse": "kasar",
    "medium": "sedang",
    "fine": "halus",
}


class CriteriaError(RuntimeError):
    """Kriteria jagung di basis data tidak ada atau tidak lengkap."""


@dataclass(frozen=True)
class CriterionRule:
    """Salinan murni satu baris ``jagung_criteria``."""

    rule_set_version: str
    parameter: ParameterKey
    suitability_class: SuitabilityClass
    rule_key: str
    value_type: str
    min_value: float | None
    max_value: float | None
    lower_inclusive: bool
    upper_inclusive: bool
    category: str | None
    improvable: bool
    improvement_step: ImprovementStep | None
    source: str

    def matches_value(self, value: float) -> bool:
        if self.min_value is not None:
            if self.lower_inclusive and value < self.min_value:
                return False
            if not self.lower_inclusive and value <= self.min_value:
                return False
        if self.max_value is not None:
            if self.upper_inclusive and value > self.max_value:
                return False
            if not self.upper_inclusive and value >= self.max_value:
                return False
        return True


@dataclass(frozen=True)
class AreaValues:
    """Atribut satu poligon ``hara_bogor`` yang dibutuhkan evaluasi jagung."""

    area_id: int
    name: str | None
    ph: float | None
    n: float | None
    p: float | None
    k: float | None
    slope: str | None
    texture_of: str | None
    unitname: str | None
    soil_great: str | None


@dataclass(frozen=True)
class AreaRecord:
    """Atribut poligon beserta luas (ha) dan geometri GeoJSON bila tersedia."""

    values: AreaValues
    area_ha: float | None = None
    geometry: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Normalisasi atribut kategoris
# ---------------------------------------------------------------------------


def texture_top(texture_of: str | None) -> str | None:
    """Kategori tekstur lapisan atas dari komponen tanah pertama di TEXTURE_OF.

    Contoh: "mod. fine/fine; mod. fine/fine" -> "agak_halus".
    Mengembalikan None bila kosong, "No Data", atau tidak dikenali.
    """
    if texture_of is None:
        return None
    first = texture_of.split(";")[0].strip().lower()
    first = re.sub(r"mod\.\s*", "mod. ", first)
    first = re.sub(r"\s+", " ", first)
    if not first or first == "no data":
        return None
    top = first.split("/")[0].strip()
    for key in sorted(TEXTURE_CATEGORIES, key=len, reverse=True):
        if top == key or top.startswith(f"{key} "):
            return TEXTURE_CATEGORIES[key]
    return None


def drainage_proxy(unitname: str | None, soil_great: str | None) -> str | None:
    """Perkiraan kelas drainase dari landform (UNITNAME) dan great group tanah.

    Dataset tidak memuat kolom drainase, sehingga dipakai proksi:
    * Tidal Swamps, Hydraquents, Sulfaquents -> sangat_terhambat
    * great group pertama mengandung "psamm" (tanah berpasir) -> cepat
    * great group pertama mengandung "aqu" (tanah jenuh air) -> terhambat
    * lainnya -> baik
    Mengembalikan None bila data tanah tidak tersedia.
    """
    unit = (unitname or "").strip().lower()
    first = (soil_great or "").split(";")[0].strip().lower()
    if unit == "tidal swamps" or first.startswith(("hydraqu", "sulfaqu")):
        return "sangat_terhambat"
    if not first or first == "no data":
        return None
    if "psamm" in first:
        return "cepat"
    if "aqu" in first:
        return "terhambat"
    return "baik"


def slope_category(slope: str | None) -> str | None:
    if slope is None:
        return None
    cleaned = slope.strip()
    if not cleaned or cleaned.lower() == "no data":
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Kriteria
# ---------------------------------------------------------------------------


def _to_float(value: Decimal | float | None) -> float | None:
    return None if value is None else float(value)


def criterion_from_model(row: JagungCriterion) -> CriterionRule:
    return CriterionRule(
        rule_set_version=row.rule_set_version,
        parameter=cast(ParameterKey, row.parameter),
        suitability_class=cast(SuitabilityClass, row.suitability_class),
        rule_key=row.rule_key,
        value_type=row.value_type,
        min_value=_to_float(row.min_value),
        max_value=_to_float(row.max_value),
        lower_inclusive=row.lower_inclusive,
        upper_inclusive=row.upper_inclusive,
        category=row.category,
        improvable=row.improvable,
        improvement_step=cast(ImprovementStep | None, row.improvement_step),
        source=row.source,
    )


def validate_criteria(criteria: Sequence[CriterionRule]) -> str:
    """Pastikan kriteria berasal dari satu versi dan mencakup semua parameter.

    Mengembalikan ``rule_set_version``; melempar ``CriteriaError`` bila tidak valid.
    """
    if not criteria:
        raise CriteriaError("Kriteria jagung aktif tidak ditemukan di tabel jagung_criteria.")
    versions = {rule.rule_set_version for rule in criteria}
    if len(versions) != 1:
        raise CriteriaError(
            "Lebih dari satu versi kriteria jagung aktif: " + ", ".join(sorted(versions))
        )
    missing = [p for p in PARAMETER_ORDER if not any(r.parameter == p for r in criteria)]
    if missing:
        raise CriteriaError("Kriteria jagung belum mencakup parameter: " + ", ".join(missing))
    return versions.pop()


def load_criteria(db: Session) -> list[CriterionRule]:
    """Muat kriteria aktif dari basis data. Error query tidak ditelan."""
    rows = db.scalars(
        select(JagungCriterion)
        .where(JagungCriterion.is_active.is_(True))
        .order_by(JagungCriterion.parameter, JagungCriterion.suitability_class)
    ).all()
    criteria = [criterion_from_model(row) for row in rows]
    validate_criteria(criteria)
    return criteria


# ---------------------------------------------------------------------------
# Klasifikasi
# ---------------------------------------------------------------------------


def class_rank(value: SuitabilityClass) -> int:
    return SUITABILITY_ORDER.index(value)


def worst_class(values: Iterable[SuitabilityClass]) -> SuitabilityClass:
    return max(values, key=class_rank)


def step_up(value: SuitabilityClass) -> SuitabilityClass:
    return SUITABILITY_ORDER[max(0, class_rank(value) - 1)]


def classify_numeric(value: float, rules: Sequence[CriterionRule]) -> CriterionRule | None:
    """Baris dengan kelas terbaik yang cocok; None berarti tidak ada yang cocok (N)."""
    matches = [rule for rule in rules if rule.matches_value(value)]
    if not matches:
        return None
    return min(matches, key=lambda rule: class_rank(rule.suitability_class))


def classify_category(category: str, rules: Sequence[CriterionRule]) -> CriterionRule | None:
    """Baris yang kategorinya sama persis; None berarti kategori tidak dikenali."""
    matches = [rule for rule in rules if rule.category == category]
    if not matches:
        return None
    return min(matches, key=lambda rule: class_rank(rule.suitability_class))


def potential_of(actual: SuitabilityClass, rule: CriterionRule | None) -> SuitabilityClass:
    if rule is None or not rule.improvable:
        return actual
    if rule.improvement_step == "ke_S1":
        return "S1"
    if rule.improvement_step == "naik_satu_kelas":
        return step_up(actual)
    return actual


def _numeric_value(values: AreaValues, parameter: ParameterKey) -> float | None:
    raw = cast(float | None, getattr(values, parameter))
    if raw is None or raw <= NO_DATA_VALUE:
        return None
    return float(raw)


def _category_value(values: AreaValues, parameter: ParameterKey) -> tuple[str | None, str | None]:
    """(nilai mentah, kategori ternormalisasi) untuk parameter kategoris."""
    if parameter == "lereng":
        return values.slope, slope_category(values.slope)
    if parameter == "tekstur":
        return values.texture_of, texture_top(values.texture_of)
    raw = "; ".join(part for part in (values.unitname, values.soil_great) if part) or None
    return raw, drainage_proxy(values.unitname, values.soil_great)


def _parameter_class(
    values: AreaValues, parameter: ParameterKey, rules: Sequence[CriterionRule]
) -> tuple[ParameterClass, bool]:
    """Kelas satu parameter dan penanda apakah datanya tersedia."""
    label = PARAMETER_LABELS[parameter]
    if parameter in NUMERIC_PARAMETERS:
        number = _numeric_value(values, parameter)
        if number is None:
            return _missing(parameter, label, None, None), False
        rule = classify_numeric(number, rules)
        actual: SuitabilityClass = rule.suitability_class if rule else "N"
        return _evaluated(parameter, label, number, None, actual, rule), True

    raw, category = _category_value(values, parameter)
    if category is None:
        return _missing(parameter, label, raw, None), False
    rule = classify_category(category, rules)
    if rule is None:
        return _missing(parameter, label, raw, category), False
    return _evaluated(parameter, label, raw, category, rule.suitability_class, rule), True


def _missing(
    parameter: ParameterKey, label: str, raw: float | str | None, category: str | None
) -> ParameterClass:
    return ParameterClass(
        parameter=parameter,
        label=label,
        raw_value=raw,
        category=category,
        actual_class=None,
        potential_class=None,
        rule_key=None,
        improvable=False,
        improvement_step=None,
        source=None,
    )


def _evaluated(
    parameter: ParameterKey,
    label: str,
    raw: float | str | None,
    category: str | None,
    actual: SuitabilityClass,
    rule: CriterionRule | None,
) -> ParameterClass:
    return ParameterClass(
        parameter=parameter,
        label=label,
        raw_value=raw,
        category=category,
        actual_class=actual,
        potential_class=potential_of(actual, rule),
        rule_key=rule.rule_key if rule else None,
        improvable=bool(rule and rule.improvable),
        improvement_step=rule.improvement_step if rule else None,
        source=rule.source if rule else None,
    )


def evaluate_area(values: AreaValues, criteria: Sequence[CriterionRule]) -> JagungSuitabilityRead:
    """Kelas aktual, kelas potensial, dan faktor pembatas jagung untuk satu poligon."""
    version = validate_criteria(criteria)
    by_parameter = {p: [r for r in criteria if r.parameter == p] for p in PARAMETER_ORDER}

    parameters: list[ParameterClass] = []
    missing: list[ParameterKey] = []
    for parameter in PARAMETER_ORDER:
        result, available = _parameter_class(values, parameter, by_parameter[parameter])
        parameters.append(result)
        if not available:
            missing.append(parameter)

    no_data_name = (values.name or "").strip().lower() in NO_DATA_NAMES
    if missing or no_data_name:
        return JagungSuitabilityRead(
            area_id=values.area_id,
            name=values.name,
            status="insufficient_data",
            rule_set_version=version,
            actual_class=None,
            potential_class=None,
            limiting_factors=[],
            improvable_factors=[],
            potential_limiting_factors=[],
            missing_parameters=missing or list(PARAMETER_ORDER),
            parameters=parameters,
            summary="Data lahan tidak cukup untuk menilai kesesuaian jagung.",
        )

    actual_by = {p.parameter: cast(SuitabilityClass, p.actual_class) for p in parameters}
    potential_by = {p.parameter: cast(SuitabilityClass, p.potential_class) for p in parameters}
    actual = worst_class(actual_by.values())
    potential = worst_class(potential_by.values())

    limiting = [p for p in PARAMETER_ORDER if actual != "S1" and actual_by[p] == actual]
    improvable = [
        p for p in PARAMETER_ORDER if class_rank(potential_by[p]) < class_rank(actual_by[p])
    ]
    potential_limiting = [
        p for p in PARAMETER_ORDER if potential != "S1" and potential_by[p] == potential
    ]

    return JagungSuitabilityRead(
        area_id=values.area_id,
        name=values.name,
        status="evaluated",
        rule_set_version=version,
        actual_class=actual,
        potential_class=potential,
        limiting_factors=limiting,
        improvable_factors=improvable,
        potential_limiting_factors=potential_limiting,
        missing_parameters=[],
        parameters=parameters,
        summary=_summary(actual, potential, limiting, potential_limiting),
    )


def _summary(
    actual: SuitabilityClass,
    potential: SuitabilityClass,
    limiting: Sequence[ParameterKey],
    potential_limiting: Sequence[ParameterKey],
) -> str:
    text = f"Kelas aktual {actual} ({CLASS_LABELS[actual]})"
    if limiting:
        text += ", dibatasi oleh " + ", ".join(PARAMETER_LABELS[p] for p in limiting)
    text += f". Kelas potensial {potential} ({CLASS_LABELS[potential]})"
    if potential_limiting:
        # Pembatas pada kelas potensial belum tentu permanen: drainase yang
        # dinaikkan satu kelas bisa tetap menjadi pembatas.
        text += " setelah perbaikan; pembatas yang tersisa: " + ", ".join(
            PARAMETER_LABELS[p] for p in potential_limiting
        )
    return text + "."


# ---------------------------------------------------------------------------
# Peringkat
# ---------------------------------------------------------------------------


def rank_areas(
    evaluations: Iterable[JagungSuitabilityRead],
    *,
    min_class: SuitabilityClass = "N",
    limit: int | None = None,
    area_ha: Mapping[int, float | None] | None = None,
    cluster_ids: Mapping[int, int | None] | None = None,
) -> list[RankingItem]:
    """Urutkan poligon terevaluasi untuk rekomendasi lahan jagung.

    Urutan: kelas potensial, kelas aktual, jumlah faktor yang perlu diperbaiki
    (sedikit lebih baik), luas (besar lebih baik; luas kosong di akhir), id.
    Hanya poligon dengan kelas potensial setara atau lebih baik dari ``min_class``.
    ``cluster_ids`` hanya ditempelkan ke hasil dan tidak memengaruhi urutan.
    """
    areas = area_ha or {}
    clusters = cluster_ids or {}
    eligible = [
        e
        for e in evaluations
        if e.status == "evaluated"
        and e.potential_class is not None
        and e.actual_class is not None
        and class_rank(e.potential_class) <= class_rank(min_class)
    ]

    def sort_key(e: JagungSuitabilityRead) -> tuple[int, int, int, bool, float, int]:
        size = areas.get(e.area_id)
        return (
            class_rank(cast(SuitabilityClass, e.potential_class)),
            class_rank(cast(SuitabilityClass, e.actual_class)),
            len(e.improvable_factors),
            size is None,
            -(size or 0.0),
            e.area_id,
        )

    ordered = sorted(eligible, key=sort_key)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        RankingItem(
            rank=index,
            area_id=e.area_id,
            name=e.name,
            actual_class=cast(SuitabilityClass, e.actual_class),
            potential_class=cast(SuitabilityClass, e.potential_class),
            limiting_factors=e.limiting_factors,
            improvable_factors=e.improvable_factors,
            cluster_id=clusters.get(e.area_id),
            area_ha=areas.get(e.area_id),
        )
        for index, e in enumerate(ordered, start=1)
    ]


# ---------------------------------------------------------------------------
# Pemuat data poligon
# ---------------------------------------------------------------------------

# Kolom unitname dan soil_great tidak dipetakan di model HaraArea (models/hara.py
# sengaja tidak diubah), sehingga di PostgreSQL dibaca lewat SQL mentah.
AREA_COLUMNS = """
    id,
    name,
    ph_rata2,
    n_rata2,
    p_rata2,
    k_rata2,
    slope__,
    texture_of,
    unitname,
    soil_great,
    ST_Area(geom::geography) / 10000.0 AS area_ha
"""
GEOMETRY_COLUMN = "ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, 0.0001))::json AS geometry"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_area_records(
    db: Session, *, area_id: int | None = None, with_geometry: bool = False
) -> list[AreaRecord]:
    """Muat poligon hara_bogor untuk evaluasi jagung, urut menurut id.

    Di PostgreSQL/PostGIS: termasuk unitname, luas (ha, dihitung pada geografi)
    dan, bila diminta, geometri GeoJSON yang disederhanakan seperti /hara/areas.
    Di dialek lain (SQLite pada uji): hanya kolom model HaraArea; unitname,
    luas, dan geometri bernilai None.
    """
    if db.get_bind().dialect.name != "postgresql":
        statement = select(HaraArea).order_by(HaraArea.id)
        if area_id is not None:
            statement = statement.where(HaraArea.id == area_id)
        return [
            AreaRecord(
                values=AreaValues(
                    area_id=area.id,
                    name=area.name,
                    ph=_number(area.ph_rata2),
                    n=_number(area.n_rata2),
                    p=_number(area.p_rata2),
                    k=_number(area.k_rata2),
                    slope=area.slope__,
                    texture_of=area.texture_of,
                    unitname=None,
                    soil_great=area.soil_great,
                )
            )
            for area in db.scalars(statement).all()
        ]

    columns = AREA_COLUMNS + (f",\n    {GEOMETRY_COLUMN}" if with_geometry else "")
    where = "WHERE id = :area_id" if area_id is not None else ""
    rows = db.execute(
        text(f"SELECT {columns} FROM hara_bogor {where} ORDER BY id"),
        {"area_id": area_id} if area_id is not None else {},
    ).mappings()
    return [
        AreaRecord(
            values=AreaValues(
                area_id=row["id"],
                name=row["name"],
                ph=_number(row["ph_rata2"]),
                n=_number(row["n_rata2"]),
                p=_number(row["p_rata2"]),
                k=_number(row["k_rata2"]),
                slope=row["slope__"],
                texture_of=row["texture_of"],
                unitname=row["unitname"],
                soil_great=row["soil_great"],
            ),
            area_ha=_number(row["area_ha"]),
            geometry=row["geometry"] if with_geometry else None,
        )
        for row in rows
    ]
