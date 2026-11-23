"""Schema respons evaluasi kesesuaian jagung (SAFRONS Jagung)."""

from typing import Literal

from pydantic import BaseModel

SuitabilityClass = Literal["S1", "S2", "S3", "N"]
EvaluationStatus = Literal["evaluated", "insufficient_data"]
ParameterKey = Literal["ph", "n", "p", "k", "lereng", "tekstur", "drainase"]
ImprovementStep = Literal["ke_S1", "naik_satu_kelas"]


class ParameterClass(BaseModel):
    """Hasil penilaian satu parameter lahan terhadap kriteria jagung."""

    parameter: ParameterKey
    label: str
    raw_value: float | str | None
    category: str | None
    actual_class: SuitabilityClass | None
    potential_class: SuitabilityClass | None
    rule_key: str | None
    improvable: bool
    improvement_step: ImprovementStep | None
    source: str | None


class JagungSuitabilityRead(BaseModel):
    """Kelas aktual dan potensial jagung untuk satu poligon."""

    area_id: int
    name: str | None
    status: EvaluationStatus
    rule_set_version: str
    actual_class: SuitabilityClass | None
    potential_class: SuitabilityClass | None
    limiting_factors: list[ParameterKey]
    improvable_factors: list[ParameterKey]
    potential_limiting_factors: list[ParameterKey]
    missing_parameters: list[ParameterKey]
    parameters: list[ParameterClass]
    summary: str


class RankingItem(BaseModel):
    """Satu baris peringkat rekomendasi lahan jagung."""

    rank: int
    area_id: int
    name: str | None
    actual_class: SuitabilityClass
    potential_class: SuitabilityClass
    limiting_factors: list[ParameterKey]
    improvable_factors: list[ParameterKey]
    cluster_id: int | None = None
    area_ha: float | None = None
