"""Latih k-means tipe profil hara dari data PostGIS (SAFRONS Jagung).

Pembungkus tipis untuk app.services.nutrient_clusters. Jalankan dari akar repo:

    python ml/train_clusters.py --version kmeans-v1
    python ml/train_clusters.py --version kmeans-v1 --dry-run
    python ml/train_clusters.py --version kmeans-v1 --db-only

--db-only dipakai bila artefak versi itu sudah ada di repo (mis. hasil commit
kelompok) tetapi basis data lokal belum berisi barisnya: model dilatih ulang
dengan parameter dari meta.json, dicocokkan dengan artefak, lalu hanya basis
data yang ditulis.

Hasil:
* ml/artifacts/<versi>/{scaler.joblib, kmeans.joblib, meta.json}
* salinan versi aktif di ml/artifacts/current/
* baris ml_model_runs, nutrient_clusters, hara_cluster_assignment, dan draf
  cluster_fertilizer_packages di basis data (kecuali --dry-run)

Basis data harus sudah menjalankan migrasi 009_nutrient_clusters.sql.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.nutrient_clusters import (
    DEFAULT_K,
    DEFAULT_N_INIT,
    DEFAULT_RANDOM_STATE,
    TrainingError,
    TrainingResult,
    load_samples,
    model_parameters,
    persist_training,
    save_artifacts,
    train_clusters,
)

ARTIFACT_ROOT = REPO_ROOT / "ml" / "artifacts"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih k-means tipe profil hara.")
    parser.add_argument("--version", required=True, help="Nama versi model, mis. kmeans-v1.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--database-url",
        default=settings.database_url,
        help="Bawaan: DATABASE_URL dari .env.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Latih dan tampilkan metrik saja; tidak menulis artefak maupun basis data.",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help=(
            "Artefak versi ini sudah ada: latih ulang dengan parameter meta.json, pastikan "
            "hasilnya sama, lalu hanya tulis basis data."
        ),
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Simpan versi tanpa menjadikannya aktif (ml/artifacts/current tidak diganti).",
    )
    return parser.parse_args(argv)


def same_model(trained: dict, stored: dict, tolerance: float = 1e-6) -> bool:
    """Bandingkan parameter scaler dan centroid dua model."""
    for key in ("scaler_mean", "scaler_scale", "centroids"):
        left = [float(x) for x in _flatten(trained[key])]
        right = [float(x) for x in _flatten(stored[key])]
        if len(left) != len(right):
            return False
        if any(abs(a - b) > tolerance for a, b in zip(left, right, strict=True)):
            return False
    return True


def _flatten(values: list) -> list:
    flat: list = []
    for value in values:
        flat.extend(_flatten(value) if isinstance(value, list) else [value])
    return flat


def print_summary(result: TrainingResult) -> None:
    m = result.metrics
    print(f"Versi model      : {result.model_version}")
    print(
        f"Parameter        : k={result.k}, n_init={result.n_init}, "
        f"random_state={result.random_state}"
    )
    print(f"Sampel           : {m['n_samples']} poligon ({m['n_unique']} kombinasi hara unik)")
    print(f"Silhouette (log) : {m['silhouette']:.3f}")
    print(f"ARI antar seed   : rata-rata {m['ari_mean']:.3f}, minimum {m['ari_min']:.3f}")
    print("Profil klaster   :")
    for p in result.profiles:
        median = ", ".join(f"{key} {value:.2f}" for key, value in p.median.items())
        print(f"  {p.cluster_id}: {p.profile_name} ({p.n_areas} poligon; median {median})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    version_dir = ARTIFACT_ROOT / args.version
    meta: dict | None = None
    if args.db_only:
        meta_path = version_dir / "meta.json"
        if not meta_path.exists():
            print(f"Gagal: {meta_path} tidak ditemukan untuk --db-only.", file=sys.stderr)
            return 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        args.k, args.n_init, args.random_state = meta["k"], meta["n_init"], meta["random_state"]
    elif not args.dry_run and version_dir.exists():
        print(
            f"Gagal: folder artefak {version_dir} sudah ada. "
            "Pakai --db-only untuk mengisi basis data dari versi ini.",
            file=sys.stderr,
        )
        return 1

    engine = create_engine(args.database_url)
    with Session(engine) as db:
        samples = load_samples(db)
        result = train_clusters(
            samples,
            model_version=args.version,
            k=args.k,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        print_summary(result)
        if args.dry_run:
            print("Dry run: tidak ada artefak atau baris basis data yang ditulis.")
            return 0

        if meta is not None:
            if not same_model(model_parameters(result), meta["parameters"]):
                print(
                    "Gagal: hasil pelatihan ulang berbeda dengan artefak di repo. "
                    "Data hara lokal kemungkinan sudah diubah; latih versi baru.",
                    file=sys.stderr,
                )
                return 1
            print("Pelatihan ulang identik dengan artefak di repo.")
        else:
            save_artifacts(result, version_dir)
        relative = version_dir.relative_to(REPO_ROOT).as_posix()
        try:
            persist_training(db, result, artifact_path=relative, activate=not args.no_activate)
        except TrainingError as exc:
            db.rollback()
            print(f"Gagal: {exc}", file=sys.stderr)
            return 1
        db.commit()

    if not args.no_activate and meta is None:
        current = ARTIFACT_ROOT / "current"
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(version_dir, current)
        print(f"Artefak aktif    : {current.relative_to(REPO_ROOT).as_posix()}")
    print(f"Artefak versi    : {relative}")
    print(
        "Basis data       : ml_model_runs, nutrient_clusters, hara_cluster_assignment, "
        "cluster_fertilizer_packages diperbarui."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
