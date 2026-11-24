"""Schema tipe profil hara (klaster k-means) dan paket pemupukan."""

from datetime import datetime

from pydantic import BaseModel, Field


class PackageRead(BaseModel):
    cluster_id: int
    title: str
    content: str
    is_draft: bool
    updated_by: int | None
    updated_at: datetime


class PackageUpdate(BaseModel):
    title: str = Field(min_length=3, max_length=128)
    content: str = Field(min_length=10, max_length=5000)
    is_draft: bool = False


class NutrientProfileRead(BaseModel):
    """Tipe profil hara satu poligon. Bukan tingkat kesuburan atau kelas kesesuaian."""

    model_version: str
    cluster_id: int
    profile_name: str
    package: PackageRead | None


class ClusterRead(BaseModel):
    cluster_id: int
    profile_name: str
    n_areas: int
    mean_ph: float
    mean_n: float
    mean_p: float
    mean_k: float
    median_ph: float
    median_n: float
    median_p: float
    median_k: float
    package: PackageRead | None


class ClustersRead(BaseModel):
    model_version: str | None
    silhouette: float | None
    clusters: list[ClusterRead]
    note: str = (
        "Klaster menggambarkan pola hara (tipe profil) untuk memilih paket pemupukan, "
        "bukan tingkat kesuburan, dan tidak memengaruhi kelas kesesuaian jagung."
    )


class RetrainRequest(BaseModel):
    k: int = Field(default=3, ge=2, le=8)
    n_init: int = Field(default=50, ge=1, le=200)
    random_state: int = Field(default=42, ge=0)


class ModelRunRead(BaseModel):
    model_version: str
    algorithm: str
    k: int
    n_init: int
    random_state: int
    n_samples: int
    n_unique: int
    silhouette: float | None
    ari_mean: float | None
    ari_min: float | None
    cluster_sizes: list[int]
    artifact_path: str
    is_active: bool
    triggered_by: int | None
    trained_at: datetime
