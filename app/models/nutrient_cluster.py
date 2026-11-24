"""Model tipe profil hara hasil k-means (lihat migrasi 009_nutrient_clusters.sql)."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Impor agar tabel users dan hara_bogor terdaftar di Base.metadata sebelum
# ForeignKey di bawah di-resolve (pola yang sama dengan hara_area_change.py).
from app.db.base import Base
from app.models.hara import HaraArea  # noqa: F401
from app.models.user import User  # noqa: F401


def _now() -> datetime:
    return datetime.now(UTC)


class MlModelRun(Base):
    __tablename__ = "ml_model_runs"
    __table_args__ = (
        CheckConstraint("k >= 2", name="ml_model_runs_k_check"),
        CheckConstraint("n_init >= 1", name="ml_model_runs_n_init_check"),
        Index(
            "ml_model_runs_one_active_idx",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False, default="kmeans")
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    n_init: Mapped[int] = mapped_column(Integer, nullable=False)
    random_state: Mapped[int] = mapped_column(Integer, nullable=False)
    features: Mapped[str] = mapped_column(Text, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    n_unique: Mapped[int] = mapped_column(Integer, nullable=False)
    silhouette: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    ari_mean: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    ari_min: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    inertia: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=dict
    )
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    triggered_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


class NutrientCluster(Base):
    __tablename__ = "nutrient_clusters"
    __table_args__ = (CheckConstraint("cluster_id >= 0", name="nutrient_clusters_id_check"),)

    model_version: Mapped[str] = mapped_column(
        ForeignKey("ml_model_runs.model_version", ondelete="CASCADE"), primary_key=True
    )
    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(96), nullable=False)
    n_areas: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_ph: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    mean_n: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    mean_p: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    mean_k: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    median_ph: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    median_n: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    median_p: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    median_k: Mapped[Decimal] = mapped_column(Numeric, nullable=False)


class HaraClusterAssignment(Base):
    __tablename__ = "hara_cluster_assignment"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_version", "cluster_id"],
            ["nutrient_clusters.model_version", "nutrient_clusters.cluster_id"],
            ondelete="CASCADE",
        ),
        Index("hara_cluster_assignment_version_idx", "model_version", "cluster_id"),
    )

    hara_area_id: Mapped[int] = mapped_column(
        ForeignKey("hara_bogor.id", ondelete="CASCADE"), primary_key=True
    )
    model_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


class ClusterFertilizerPackage(Base):
    __tablename__ = "cluster_fertilizer_packages"
    __table_args__ = (
        UniqueConstraint(
            "model_version", "cluster_id", name="cluster_fertilizer_packages_cluster_uq"
        ),
        ForeignKeyConstraint(
            ["model_version", "cluster_id"],
            ["nutrient_clusters.model_version", "nutrient_clusters.cluster_id"],
            ondelete="CASCADE",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_draft: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
