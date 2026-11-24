"""Endpoint kesesuaian lahan jagung (SAFRONS Jagung).

Terpisah dari /api/v1/hara agar respons dan cache 24 jam pada /hara/areas tidak
berubah. Kelas dihitung ulang dari data terkini pada setiap permintaan; tidak
ada pelatihan model di sini.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.jagung import (
    JagungRankingRead,
    JagungSuitabilityCollection,
    JagungSuitabilityFeature,
    JagungSuitabilityProperties,
    JagungSuitabilityRead,
    SuitabilityClass,
)
from app.services.jagung_suitability import (
    CriteriaError,
    CriterionRule,
    evaluate_area,
    load_area_records,
    load_criteria,
    rank_areas,
    validate_criteria,
)
from app.services.nutrient_clusters import active_cluster_ids, nutrient_profile

router = APIRouter(prefix="/api/v1/jagung", tags=["jagung"])

# Cache pendek: expert dapat mengubah data hara, dan kelas harus segera ikut berubah.
SHORT_CACHE = "public, max-age=300"


def _criteria(db: Session) -> list[CriterionRule]:
    try:
        return load_criteria(db)
    except CriteriaError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kriteria jagung tidak valid: {exc}",
        ) from exc


@router.get("/areas/{area_id}", response_model=JagungSuitabilityRead)
def get_jagung_area(
    area_id: int,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> JagungSuitabilityRead:
    criteria = _criteria(db)
    records = load_area_records(db, area_id=area_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Area hara tidak ditemukan."
        )
    response.headers["Cache-Control"] = "no-cache"
    record = records[0]
    result = evaluate_area(record.values, criteria)
    return result.model_copy(
        update={"area_ha": record.area_ha, "nutrient_profile": nutrient_profile(db, area_id)}
    )


@router.get("/ranking", response_model=JagungRankingRead)
def get_jagung_ranking(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    min_class: Annotated[
        SuitabilityClass,
        Query(description="Kelas potensial terendah yang ikut ditampilkan."),
    ] = "S3",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JagungRankingRead:
    criteria = _criteria(db)
    records = load_area_records(db)
    evaluations = [evaluate_area(record.values, criteria) for record in records]
    candidates = rank_areas(
        evaluations,
        min_class=min_class,
        area_ha={record.values.area_id: record.area_ha for record in records},
        cluster_ids=active_cluster_ids(db),
    )
    response.headers["Cache-Control"] = SHORT_CACHE
    return JagungRankingRead(
        rule_set_version=validate_criteria(criteria),
        min_class=min_class,
        limit=limit,
        total_candidates=len(candidates),
        items=candidates[:limit],
    )


@router.get("/suitability", response_model=JagungSuitabilityCollection)
def get_jagung_suitability_layer(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> JagungSuitabilityCollection:
    criteria = _criteria(db)
    cluster_ids = active_cluster_ids(db)
    features: list[JagungSuitabilityFeature] = []
    for record in load_area_records(db, with_geometry=True):
        result = evaluate_area(record.values, criteria)
        features.append(
            JagungSuitabilityFeature(
                geometry=record.geometry,
                properties=JagungSuitabilityProperties(
                    id=result.area_id,
                    name=result.name,
                    status=result.status,
                    actual_class=result.actual_class,
                    potential_class=result.potential_class,
                    limiting_factors=result.limiting_factors,
                    potential_limiting_factors=result.potential_limiting_factors,
                    area_ha=record.area_ha,
                    cluster_id=cluster_ids.get(result.area_id),
                ),
            )
        )
    response.headers["Cache-Control"] = SHORT_CACHE
    return JagungSuitabilityCollection(
        rule_set_version=validate_criteria(criteria), features=features
    )
