"""Endpoint publik tipe profil hara (klaster k-means)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.cluster import ClustersRead
from app.services.nutrient_clusters import list_clusters

router = APIRouter(prefix="/api/v1/clusters", tags=["clusters"])


@router.get("", response_model=ClustersRead)
def get_clusters(response: Response, db: Annotated[Session, Depends(get_db)]) -> ClustersRead:
    response.headers["Cache-Control"] = "no-cache"
    return list_clusters(db)
