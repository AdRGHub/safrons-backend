"""Bantuan uji SAFRONS Jagung: kriteria dibaca langsung dari migrasi 008.

Dengan membaca berkas SQL, uji selalu memakai angka yang sama dengan seed di
basis data, sehingga tidak ada salinan ambang kedua yang bisa tidak sinkron.
"""

import csv
from pathlib import Path
from typing import cast

from app.schemas.jagung import ImprovementStep, ParameterKey, SuitabilityClass
from app.services.jagung_suitability import CriterionRule

MIGRATION_008 = (
    Path(__file__).resolve().parents[1] / "database" / "migrations" / "008_jagung_criteria.sql"
)


def _literal(token: str) -> str | float | bool | None:
    token = token.strip()
    if token == "NULL":
        return None
    if token == "TRUE":
        return True
    if token == "FALSE":
        return False
    if token.startswith("'"):
        return token[1:-1].replace("''", "'")
    return float(token)


def seed_rows() -> list[CriterionRule]:
    """Baris seed pada blok INSERT di 008_jagung_criteria.sql."""
    rules: list[CriterionRule] = []
    for line in MIGRATION_008.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("('jagung-"):
            continue
        inner = stripped.rstrip(",")[1:-1]
        tokens = next(csv.reader([inner], quotechar="'", skipinitialspace=True))
        # csv membuang tanda kutip; kembalikan agar _literal bisa membedakan teks.
        raw = [f"'{t}'" if not _is_bare(t) else t for t in tokens]
        values = [_literal(t) for t in raw]
        (
            version,
            parameter,
            suitability_class,
            rule_key,
            value_type,
            min_value,
            max_value,
            lower_inclusive,
            upper_inclusive,
            category,
            improvable,
            improvement_step,
            source,
        ) = values
        rules.append(
            CriterionRule(
                rule_set_version=cast(str, version),
                parameter=cast(ParameterKey, parameter),
                suitability_class=cast(SuitabilityClass, suitability_class),
                rule_key=cast(str, rule_key),
                value_type=cast(str, value_type),
                min_value=cast(float | None, min_value),
                max_value=cast(float | None, max_value),
                lower_inclusive=cast(bool, lower_inclusive),
                upper_inclusive=cast(bool, upper_inclusive),
                category=cast(str | None, category),
                improvable=cast(bool, improvable),
                improvement_step=cast(ImprovementStep | None, improvement_step),
                source=cast(str, source),
            )
        )
    return rules


def _is_bare(token: str) -> bool:
    if token in {"NULL", "TRUE", "FALSE"}:
        return True
    try:
        float(token)
    except ValueError:
        return False
    return True
