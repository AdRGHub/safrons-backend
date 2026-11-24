"""Endpoint expert untuk paket pemupukan per tipe profil hara."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_expert
from app.db.session import get_db
from app.models.user import User
from app.schemas.cluster import PackageRead, PackageUpdate
from app.services.nutrient_clusters import update_package

router = APIRouter(prefix="/api/v1/expert/clusters", tags=["expert"])


@router.patch("/{cluster_id}/package", response_model=PackageRead)
def update_cluster_package(
    cluster_id: int,
    payload: PackageUpdate,
    expert_user: Annotated[User, Depends(require_expert)],
    db: Annotated[Session, Depends(get_db)],
) -> PackageRead:
    package = update_package(db, cluster_id, payload, expert_user.id)
    if package is None:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Klaster tidak ditemukan pada model aktif.",
        )
    db.commit()
    return package
