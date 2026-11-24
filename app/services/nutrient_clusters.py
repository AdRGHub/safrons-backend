"""Tipe profil hara dengan k-means (SAFRONS Jagung).

Peran k-means dibatasi (Analisis Kelayakan K-Means + Rule-Based):
* Klaster = TIPE PROFIL HARA untuk memilih paket pemupukan, bukan tingkat
  kesuburan. Nama profil diturunkan dari median klaster relatif terhadap median
  dataset, bukan dari nomor klaster.
* ``cluster_id`` tidak dipakai dalam kelas kesesuaian maupun peringkat jagung.

Modul ini memuat logika pelatihan (dipakai oleh ``ml/train_clusters.py`` dan,
mulai Tahap 5, oleh endpoint retrain admin). Pelatihan tidak pernah dijalankan
saat melayani permintaan publik.
"""

from __future__ import annotations

import json
import os
import platform
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.hara import HaraArea
from app.models.nutrient_cluster import (
    ClusterFertilizerPackage,
    HaraClusterAssignment,
    MlModelRun,
    NutrientCluster,
)
from app.schemas.cluster import (
    ClusterRead,
    ClustersRead,
    ModelRunRead,
    NutrientProfileRead,
    PackageRead,
    PackageUpdate,
)
from app.services.hara_diagnosis import NUTRIENT_THRESHOLDS, classify_by_thresholds

DEFAULT_K = 3
DEFAULT_N_INIT = 50
DEFAULT_RANDOM_STATE = 42
ARI_SEEDS = tuple(range(10))
FEATURE_DESCRIPTION = "pH, log1p(N), log1p(P), log1p(K); StandardScaler"

NO_DATA_VALUE = -9999
NO_DATA_NAMES = {"water", "no data"}

# Batas rasio median klaster terhadap median dataset untuk penamaan profil.
LOW_RATIO = 0.8
HIGH_RATIO = 1.25

NUTRIENT_LABELS = {"n": "N", "p": "P", "k": "K"}
FERTILIZERS = {"n": "Urea", "p": "SP-36", "k": "KCl"}


class TrainingError(RuntimeError):
    """Data atau parameter tidak memungkinkan pelatihan."""


@dataclass(frozen=True)
class NutrientSample:
    area_id: int
    ph: float
    n: float
    p: float
    k: float


@dataclass
class ClusterProfile:
    cluster_id: int
    profile_name: str
    n_areas: int
    mean: dict[str, float]
    median: dict[str, float]
    levels: dict[str, str]
    acidity: str


@dataclass
class TrainingResult:
    model_version: str
    k: int
    n_init: int
    random_state: int
    scaler: StandardScaler
    kmeans: KMeans
    labels: dict[int, int]
    profiles: list[ClusterProfile]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def silhouette(self) -> float:
        return float(self.metrics["silhouette"])

    @property
    def ari_mean(self) -> float:
        return float(self.metrics["ari_mean"])

    @property
    def ari_min(self) -> float:
        return float(self.metrics["ari_min"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _valid(value: Any) -> bool:
    return value is not None and float(value) > NO_DATA_VALUE


def load_samples(db: Session) -> list[NutrientSample]:
    """Poligon dengan pH, N, P, K lengkap; Water/No Data dan -9999 dibuang."""
    samples: list[NutrientSample] = []
    for area in db.scalars(select(HaraArea).order_by(HaraArea.id)).all():
        if (area.name or "").strip().lower() in NO_DATA_NAMES:
            continue
        values = (area.ph_rata2, area.n_rata2, area.p_rata2, area.k_rata2)
        if not all(_valid(v) for v in values):
            continue
        ph, n, p, k = (float(v) for v in values)  # type: ignore[arg-type]
        samples.append(NutrientSample(area_id=area.id, ph=ph, n=n, p=p, k=k))
    return samples


def feature_matrix(samples: Sequence[NutrientSample]) -> np.ndarray:
    """Kolom: pH, log1p(N), log1p(P), log1p(K) (sebelum standardisasi)."""
    return np.array(
        [[s.ph, np.log1p(s.n), np.log1p(s.p), np.log1p(s.k)] for s in samples], dtype=float
    )


# ---------------------------------------------------------------------------
# Profil
# ---------------------------------------------------------------------------


def acidity_label(ph: float) -> str:
    if ph < 5.0:
        return "sangat masam"
    if ph < 5.5:
        return "masam"
    if ph < 6.6:
        return "agak masam"
    return "netral"


def nutrient_level(cluster_median: float, dataset_median: float) -> str:
    ratio = cluster_median / dataset_median if dataset_median else 1.0
    if ratio < LOW_RATIO:
        return "rendah"
    if ratio > HIGH_RATIO:
        return "tinggi"
    return "sedang"


def profile_name(levels: dict[str, str], acidity: str) -> str:
    parts = [
        f"{NUTRIENT_LABELS[key]} {level}" for key, level in levels.items() if level != "sedang"
    ]
    if not parts:
        parts = ["N, P, K sedang"]
    return ", ".join(parts) + f", {acidity}"


def draft_package(profile: ClusterProfile) -> tuple[str, str]:
    """Judul dan isi draf paket pemupukan dari profil klaster.

    Hanya arah tindakan, tanpa dosis; isi wajib divalidasi expert (is_draft).
    """
    lines = [
        "DRAF otomatis dari profil klaster; wajib divalidasi expert sebelum dipakai.",
        "",
    ]
    for key in ("n", "p", "k"):
        level = profile.levels[key]
        fertilizer = FERTILIZERS[key]
        label = NUTRIENT_LABELS[key]
        if level == "rendah":
            lines.append(f"- {label} relatif rendah: prioritaskan {fertilizer}.")
        elif level == "tinggi":
            lines.append(f"- {label} relatif tinggi: kurangi atau tunda {fertilizer}.")
        else:
            lines.append(f"- {label} sedang: {fertilizer} dosis pemeliharaan.")
    if profile.acidity in {"sangat masam", "masam"}:
        lines.append(f"- Tanah {profile.acidity}: pengapuran (dolomit) sebelum tanam.")
    elif profile.acidity == "agak masam":
        lines.append("- Tanah agak masam: pantau pH; pengapuran ringan bila pH di bawah 5,8.")
    lines.append("")
    lines.append(
        "Tipe profil ini menggambarkan pola hara, bukan tingkat kesuburan atau "
        "kesesuaian lahan untuk jagung."
    )
    return f"Paket pemupukan: {profile.profile_name}", "\n".join(lines)


def _profiles(
    samples: Sequence[NutrientSample], labels: np.ndarray, k: int
) -> list[ClusterProfile]:
    raw = np.array([[s.ph, s.n, s.p, s.k] for s in samples], dtype=float)
    dataset_median = np.median(raw, axis=0)
    profiles: list[ClusterProfile] = []
    for cluster_id in range(k):
        members = raw[labels == cluster_id]
        mean = members.mean(axis=0)
        median = np.median(members, axis=0)
        levels = {
            key: nutrient_level(float(median[i]), float(dataset_median[i]))
            for i, key in ((1, "n"), (2, "p"), (3, "k"))
        }
        acidity = acidity_label(float(median[0]))
        profiles.append(
            ClusterProfile(
                cluster_id=cluster_id,
                profile_name=profile_name(levels, acidity),
                n_areas=len(members),
                mean=dict(zip(("ph", "n", "p", "k"), map(float, mean), strict=True)),
                median=dict(zip(("ph", "n", "p", "k"), map(float, median), strict=True)),
                levels=levels,
                acidity=acidity,
            )
        )
    return profiles


def _crosstab(samples: Sequence[NutrientSample], labels: np.ndarray, k: int) -> dict[str, Any]:
    """Sebaran status hara rule-based SAFRONS (kuantil) di tiap klaster."""
    table: dict[str, Any] = {}
    for key, column in (("n", "n_rata2"), ("p", "p_rata2"), ("k", "k_rata2")):
        thresholds = NUTRIENT_THRESHOLDS[column]
        per_cluster: dict[str, dict[str, int]] = {}
        for cluster_id in range(k):
            statuses = Counter(
                classify_by_thresholds(getattr(s, key), thresholds)
                for s, label in zip(samples, labels, strict=True)
                if label == cluster_id
            )
            per_cluster[str(cluster_id)] = dict(sorted(statuses.items()))
        table[key] = per_cluster
    return table


# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------


def _canonical_order(kmeans: KMeans, raw_ph: np.ndarray) -> np.ndarray:
    """Urutan klaster: anggota terbanyak dulu, lalu median pH terendah.

    Nomor klaster dari scikit-learn bergantung pada inisialisasi; urutan ini
    membuat cluster_id stabil dan mudah dirujuk di laporan.
    """
    labels = kmeans.labels_
    keys = []
    for cluster_id in range(kmeans.n_clusters):
        mask = labels == cluster_id
        keys.append((-int(mask.sum()), float(np.median(raw_ph[mask])), cluster_id))
    return np.array([key[2] for key in sorted(keys)])


def train_clusters(
    samples: Sequence[NutrientSample],
    *,
    model_version: str,
    k: int = DEFAULT_K,
    n_init: int = DEFAULT_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainingResult:
    """Latih k-means pada pH dan log N, P, K yang distandardisasi."""
    if k < 2:
        raise TrainingError("k minimal 2.")
    if n_init < 1:
        raise TrainingError("n_init minimal 1.")
    features = feature_matrix(samples)
    n_unique = len({tuple(row) for row in features})
    if n_unique < k:
        raise TrainingError(
            f"Hanya ada {n_unique} kombinasi hara unik; tidak cukup untuk {k} klaster."
        )

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(scaled)

    order = _canonical_order(kmeans, features[:, 0])
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    kmeans.cluster_centers_ = kmeans.cluster_centers_[order]
    kmeans.labels_ = remap[kmeans.labels_]
    labels = kmeans.labels_

    seed_labels = [
        KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(scaled)
        for seed in ARI_SEEDS
    ]
    aris = [adjusted_rand_score(labels, other) for other in seed_labels]

    profiles = _profiles(samples, labels, k)
    metrics: dict[str, Any] = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "ari_mean": float(np.mean(aris)),
        "ari_min": float(np.min(aris)),
        "ari_seeds": list(ARI_SEEDS),
        "inertia": float(kmeans.inertia_),
        "n_samples": len(samples),
        "n_unique": n_unique,
        "cluster_sizes": [p.n_areas for p in profiles],
        "crosstab_rule_based_status": _crosstab(samples, labels, k),
        "profile_rule": (
            f"rendah bila median klaster < {LOW_RATIO} x median dataset, "
            f"tinggi bila > {HIGH_RATIO} x; keasaman dari median pH"
        ),
    }
    return TrainingResult(
        model_version=model_version,
        k=k,
        n_init=n_init,
        random_state=random_state,
        scaler=scaler,
        kmeans=kmeans,
        labels={s.area_id: int(label) for s, label in zip(samples, labels, strict=True)},
        profiles=profiles,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Artefak dan basis data
# ---------------------------------------------------------------------------


def model_parameters(result: TrainingResult) -> dict[str, Any]:
    """Parameter yang cukup untuk prediksi tanpa memuat pickle scikit-learn."""
    return {
        "features": ["ph", "log1p_n", "log1p_p", "log1p_k"],
        "scaler_mean": result.scaler.mean_.tolist(),
        "scaler_scale": result.scaler.scale_.tolist(),
        "centroids": result.kmeans.cluster_centers_.tolist(),
    }


def save_artifacts(result: TrainingResult, directory: Path) -> Path:
    """Tulis scaler.joblib, kmeans.joblib, dan meta.json ke ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.scaler, directory / "scaler.joblib")
    joblib.dump(result.kmeans, directory / "kmeans.joblib")
    meta = {
        "model_version": result.model_version,
        "algorithm": "kmeans",
        "k": result.k,
        "n_init": result.n_init,
        "random_state": result.random_state,
        "feature_description": FEATURE_DESCRIPTION,
        "parameters": model_parameters(result),
        "metrics": result.metrics,
        "profiles": [
            {
                "cluster_id": p.cluster_id,
                "profile_name": p.profile_name,
                "n_areas": p.n_areas,
                "mean": p.mean,
                "median": p.median,
                "levels": p.levels,
                "acidity": p.acidity,
            }
            for p in result.profiles
        ],
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return directory


def persist_training(
    db: Session,
    result: TrainingResult,
    *,
    artifact_path: str,
    triggered_by: int | None = None,
    activate: bool = True,
) -> MlModelRun:
    """Simpan versi model, profil, assignment, dan draf paket dalam satu transaksi.

    Tidak melakukan commit; pemanggil yang menentukan. Bila ``activate``, versi
    lain dinonaktifkan lebih dulu agar indeks unik "satu versi aktif" terpenuhi.
    """
    existing = db.scalar(select(MlModelRun).where(MlModelRun.model_version == result.model_version))
    if existing is not None:
        raise TrainingError(f"Versi model {result.model_version} sudah ada.")

    if activate:
        db.execute(update(MlModelRun).where(MlModelRun.is_active.is_(True)).values(is_active=False))
        db.flush()

    run = MlModelRun(
        model_version=result.model_version,
        algorithm="kmeans",
        k=result.k,
        n_init=result.n_init,
        random_state=result.random_state,
        features=FEATURE_DESCRIPTION,
        n_samples=int(result.metrics["n_samples"]),
        n_unique=int(result.metrics["n_unique"]),
        silhouette=result.silhouette,
        ari_mean=result.ari_mean,
        ari_min=result.ari_min,
        inertia=float(result.metrics["inertia"]),
        metrics={**result.metrics, "parameters": model_parameters(result)},
        artifact_path=artifact_path,
        is_active=activate,
        triggered_by=triggered_by,
    )
    db.add(run)
    db.flush()

    for profile in result.profiles:
        db.add(
            NutrientCluster(
                model_version=result.model_version,
                cluster_id=profile.cluster_id,
                profile_name=profile.profile_name,
                n_areas=profile.n_areas,
                mean_ph=profile.mean["ph"],
                mean_n=profile.mean["n"],
                mean_p=profile.mean["p"],
                mean_k=profile.mean["k"],
                median_ph=profile.median["ph"],
                median_n=profile.median["n"],
                median_p=profile.median["p"],
                median_k=profile.median["k"],
            )
        )
    db.flush()

    for profile in result.profiles:
        title, content = draft_package(profile)
        db.add(
            ClusterFertilizerPackage(
                model_version=result.model_version,
                cluster_id=profile.cluster_id,
                title=title,
                content=content,
                is_draft=True,
            )
        )
    db.add_all(
        HaraClusterAssignment(
            hara_area_id=area_id, model_version=result.model_version, cluster_id=cluster_id
        )
        for area_id, cluster_id in result.labels.items()
    )
    db.flush()
    return run


# ---------------------------------------------------------------------------
# Runtime: prediksi, profil per poligon, paket, retrain
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_KEYS = ("scaler_mean", "scaler_scale", "centroids")

# Cache parameter model per versi (keputusan 2b): dimuat saat pertama kali dipakai,
# dikosongkan setelah retrain.
_PARAMETER_CACHE: dict[str, dict[str, Any]] = {}


class ClusterModelError(RuntimeError):
    """Model klaster aktif tidak dapat dipakai (parameter atau artefak tidak ada)."""


def artifact_root() -> Path:
    """Folder dasar untuk artifact_path di ml_model_runs (bawaan: akar repo).

    Dapat diganti lewat CLUSTER_ARTIFACT_ROOT (variabel lingkungan atau .env), mis.
    volume persisten di deployment.
    """
    configured = os.environ.get("CLUSTER_ARTIFACT_ROOT") or settings.cluster_artifact_root
    return Path(configured) if configured else REPO_ROOT


def clear_model_cache() -> None:
    _PARAMETER_CACHE.clear()


def active_run(db: Session) -> MlModelRun | None:
    return db.scalar(select(MlModelRun).where(MlModelRun.is_active.is_(True)))


def run_parameters(run: MlModelRun) -> dict[str, Any]:
    """Parameter scaler dan centroid versi model.

    Diutamakan dari ml_model_runs.metrics["parameters"] (tidak bergantung pada
    disk). Versi lama tanpa kolom itu dibaca dari meta.json di artifact_path.
    """
    cached = _PARAMETER_CACHE.get(run.model_version)
    if cached is not None:
        return cached
    parameters = (run.metrics or {}).get("parameters")
    if parameters is None:
        meta_path = artifact_root() / run.artifact_path / "meta.json"
        if not meta_path.exists():
            raise ClusterModelError(
                f"Parameter model {run.model_version} tidak ada di basis data dan "
                f"{meta_path} tidak ditemukan."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_version") != run.model_version:
            raise ClusterModelError(
                f"meta.json di {meta_path} milik versi {meta.get('model_version')}, "
                f"bukan {run.model_version}."
            )
        parameters = meta["parameters"]
    missing = [key for key in PARAMETER_KEYS if key not in parameters]
    if missing:
        raise ClusterModelError(
            f"Parameter model {run.model_version} tidak lengkap: {', '.join(missing)}."
        )
    _PARAMETER_CACHE[run.model_version] = parameters
    return parameters


def predict_cluster(parameters: dict[str, Any], ph: float, n: float, p: float, k: float) -> int:
    """Centroid terdekat pada ruang fitur terstandardisasi (setara KMeans.predict)."""
    features = np.array([ph, np.log1p(n), np.log1p(p), np.log1p(k)], dtype=float)
    scaled = (features - np.array(parameters["scaler_mean"])) / np.array(parameters["scaler_scale"])
    centroids = np.array(parameters["centroids"], dtype=float)
    return int(((centroids - scaled) ** 2).sum(axis=1).argmin())


def reassign_cluster(db: Session, area_id: int) -> int | None:
    """Perbarui cluster_id satu poligon untuk versi model aktif (F-15).

    Dipanggil di dalam transaksi edit/buat area oleh expert, sebelum commit.
    Mengembalikan cluster_id baru, atau None bila belum ada model aktif atau data
    hara poligon tidak lengkap (assignment lama untuk versi aktif dihapus).
    """
    run = active_run(db)
    if run is None:
        return None
    db.flush()
    area = db.execute(
        select(HaraArea).where(HaraArea.id == area_id).execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if area is None:
        return None
    db.execute(
        delete(HaraClusterAssignment).where(
            HaraClusterAssignment.hara_area_id == area_id,
            HaraClusterAssignment.model_version == run.model_version,
        )
    )
    values = (area.ph_rata2, area.n_rata2, area.p_rata2, area.k_rata2)
    no_data_name = (area.name or "").strip().lower() in NO_DATA_NAMES
    if no_data_name or not all(_valid(v) for v in values):
        db.flush()
        return None
    ph, n, p, k = (float(v) for v in values)  # type: ignore[arg-type]
    cluster_id = predict_cluster(run_parameters(run), ph, n, p, k)
    db.add(
        HaraClusterAssignment(
            hara_area_id=area_id, model_version=run.model_version, cluster_id=cluster_id
        )
    )
    db.flush()
    return cluster_id


def _package_read(package: ClusterFertilizerPackage | None) -> PackageRead | None:
    if package is None:
        return None
    return PackageRead(
        cluster_id=package.cluster_id,
        title=package.title,
        content=package.content,
        is_draft=package.is_draft,
        updated_by=package.updated_by,
        updated_at=package.updated_at,
    )


def _package(db: Session, model_version: str, cluster_id: int) -> ClusterFertilizerPackage | None:
    return db.scalar(
        select(ClusterFertilizerPackage).where(
            ClusterFertilizerPackage.model_version == model_version,
            ClusterFertilizerPackage.cluster_id == cluster_id,
        )
    )


def nutrient_profile(db: Session, area_id: int) -> NutrientProfileRead | None:
    """Tipe profil hara dan paket pemupukan satu poligon (versi model aktif)."""
    run = active_run(db)
    if run is None:
        return None
    assignment = db.get(HaraClusterAssignment, (area_id, run.model_version))
    if assignment is None:
        return None
    cluster = db.get(NutrientCluster, (run.model_version, assignment.cluster_id))
    if cluster is None:
        return None
    return NutrientProfileRead(
        model_version=run.model_version,
        cluster_id=cluster.cluster_id,
        profile_name=cluster.profile_name,
        package=_package_read(_package(db, run.model_version, cluster.cluster_id)),
    )


def active_cluster_ids(db: Session) -> dict[int, int]:
    """area_id -> cluster_id untuk versi aktif; kosong bila belum ada model."""
    run = active_run(db)
    if run is None:
        return {}
    rows = db.execute(
        select(HaraClusterAssignment.hara_area_id, HaraClusterAssignment.cluster_id).where(
            HaraClusterAssignment.model_version == run.model_version
        )
    ).all()
    return {area_id: cluster_id for area_id, cluster_id in rows}


def list_clusters(db: Session) -> ClustersRead:
    run = active_run(db)
    if run is None:
        return ClustersRead(model_version=None, silhouette=None, clusters=[])
    clusters = db.scalars(
        select(NutrientCluster)
        .where(NutrientCluster.model_version == run.model_version)
        .order_by(NutrientCluster.cluster_id)
    ).all()
    return ClustersRead(
        model_version=run.model_version,
        silhouette=None if run.silhouette is None else float(run.silhouette),
        clusters=[
            ClusterRead(
                cluster_id=c.cluster_id,
                profile_name=c.profile_name,
                n_areas=c.n_areas,
                mean_ph=float(c.mean_ph),
                mean_n=float(c.mean_n),
                mean_p=float(c.mean_p),
                mean_k=float(c.mean_k),
                median_ph=float(c.median_ph),
                median_n=float(c.median_n),
                median_p=float(c.median_p),
                median_k=float(c.median_k),
                package=_package_read(_package(db, run.model_version, c.cluster_id)),
            )
            for c in clusters
        ],
    )


def update_package(
    db: Session, cluster_id: int, payload: PackageUpdate, user_id: int
) -> PackageRead | None:
    """Ubah paket pemupukan klaster pada versi aktif. None bila tidak ditemukan."""
    run = active_run(db)
    if run is None:
        return None
    package = _package(db, run.model_version, cluster_id)
    if package is None:
        return None
    package.title = payload.title.strip()
    package.content = payload.content.strip()
    package.is_draft = payload.is_draft
    package.updated_by = user_id
    package.updated_at = datetime.now(UTC)
    db.flush()
    return _package_read(package)


def retrain(
    db: Session,
    *,
    triggered_by: int | None,
    k: int = DEFAULT_K,
    n_init: int = DEFAULT_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MlModelRun:
    """Latih ulang penuh dari data terkini, simpan artefak, aktifkan versi baru (F-16).

    Paket yang sudah divalidasi expert (is_draft = FALSE) disalin ke klaster versi
    baru yang nama profilnya sama; klaster lain mendapat draf baru. Tidak commit.
    """
    previous = active_run(db)
    validated: dict[str, ClusterFertilizerPackage] = {}
    if previous is not None:
        for cluster in db.scalars(
            select(NutrientCluster).where(NutrientCluster.model_version == previous.model_version)
        ).all():
            package = _package(db, previous.model_version, cluster.cluster_id)
            if package is not None and not package.is_draft:
                validated[cluster.profile_name] = package

    version = f"kmeans-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    result = train_clusters(
        load_samples(db), model_version=version, k=k, n_init=n_init, random_state=random_state
    )
    relative = Path("ml") / "artifacts" / version
    target = artifact_root() / relative
    if target.exists():
        raise TrainingError(f"Folder artefak {relative.as_posix()} sudah ada.")
    run = persist_training(
        db, result, artifact_path=relative.as_posix(), triggered_by=triggered_by, activate=True
    )
    save_artifacts(result, target)
    for profile in result.profiles:
        source = validated.get(profile.profile_name)
        if source is None:
            continue
        package = _package(db, version, profile.cluster_id)
        if package is not None:
            package.title = source.title
            package.content = source.content
            package.is_draft = False
            package.updated_by = source.updated_by
    db.flush()
    clear_model_cache()
    return run


def list_runs(db: Session) -> list[ModelRunRead]:
    runs = db.scalars(
        select(MlModelRun).order_by(MlModelRun.trained_at.desc(), MlModelRun.id.desc())
    ).all()
    return [
        ModelRunRead(
            model_version=r.model_version,
            algorithm=r.algorithm,
            k=r.k,
            n_init=r.n_init,
            random_state=r.random_state,
            n_samples=r.n_samples,
            n_unique=r.n_unique,
            silhouette=None if r.silhouette is None else float(r.silhouette),
            ari_mean=None if r.ari_mean is None else float(r.ari_mean),
            ari_min=None if r.ari_min is None else float(r.ari_min),
            cluster_sizes=list((r.metrics or {}).get("cluster_sizes", [])),
            artifact_path=r.artifact_path,
            is_active=r.is_active,
            triggered_by=r.triggered_by,
            trained_at=r.trained_at,
        )
        for r in runs
    ]
