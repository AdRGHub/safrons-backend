"""Endpoint admin untuk pelatihan ulang dan riwayat model klaster."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.cluster import ModelRunRead, RetrainRequest
from app.services.nutrient_clusters import TrainingError, list_runs, retrain

router = APIRouter(prefix="/api/v1/admin/clusters", tags=["admin"])


@router.post("/retrain", response_model=ModelRunRead, status_code=status.HTTP_201_CREATED)
def retrain_clusters(
    admin_user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    payload: RetrainRequest | None = None,
) -> ModelRunRead:
    request = payload or RetrainRequest()
    try:
        run = retrain(
            db,
            triggered_by=admin_user.id,
            k=request.k,
            n_init=request.n_init,
            random_state=request.random_state,
        )
    except TrainingError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return next(r for r in list_runs(db) if r.model_version == run.model_version)


@router.get("/runs", response_model=list[ModelRunRead])
def get_cluster_runs(
    _admin_user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[ModelRunRead]:
    return list_runs(db)
