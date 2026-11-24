-- 009_nutrient_clusters.sql
-- Tabel tipe profil hara hasil k-means (SAFRONS Jagung, proyek PPKBP).
--
-- Peran k-means dibatasi (Analisis Kelayakan K-Means + Rule-Based): klaster
-- dibaca sebagai TIPE PROFIL HARA untuk memilih paket pemupukan, bukan tingkat
-- kesuburan, dan TIDAK dipakai dalam kelas kesesuaian maupun peringkat jagung.
--
-- Pelatihan berjalan offline (ml/train_clusters.py) atau lewat endpoint admin;
-- API publik hanya membaca hasilnya. Tabel hara_bogor tidak diubah; cluster_id
-- disimpan di hara_cluster_assignment.
--
-- Tabel:
--   ml_model_runs              satu baris per versi model (parameter, metrik, artefak)
--   nutrient_clusters          profil rata-rata tiap klaster per versi
--   hara_cluster_assignment    cluster_id tiap poligon per versi
--   cluster_fertilizer_packages paket anjuran pemupukan tiap klaster, dapat diedit expert
--
-- Hanya satu versi boleh aktif (indeks unik parsial pada ml_model_runs).
--
-- Idempoten: aman dijalankan berulang, baik oleh initdb Docker (psql) maupun
-- oleh runner migrasi di lifespan aplikasi (SQLAlchemy text()). Berkas ini tidak
-- memakai meta-command psql dan tidak memakai pola titik dua yang diikuti nama.

CREATE TABLE IF NOT EXISTS ml_model_runs (
    id             SERIAL       PRIMARY KEY,
    model_version  VARCHAR(32)  NOT NULL UNIQUE,
    algorithm      VARCHAR(32)  NOT NULL DEFAULT 'kmeans',
    k              INTEGER      NOT NULL CHECK (k >= 2),
    n_init         INTEGER      NOT NULL CHECK (n_init >= 1),
    random_state   INTEGER      NOT NULL,
    features       TEXT         NOT NULL,
    n_samples      INTEGER      NOT NULL,
    n_unique       INTEGER      NOT NULL,
    silhouette     NUMERIC,
    ari_mean       NUMERIC,
    ari_min        NUMERIC,
    inertia        NUMERIC,
    metrics        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    artifact_path  TEXT         NOT NULL,
    is_active      BOOLEAN      NOT NULL DEFAULT FALSE,
    triggered_by   INTEGER      REFERENCES users (id) ON DELETE SET NULL,
    trained_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ml_model_runs_one_active_idx
    ON ml_model_runs (is_active)
    WHERE is_active;

CREATE TABLE IF NOT EXISTS nutrient_clusters (
    model_version  VARCHAR(32)  NOT NULL
        REFERENCES ml_model_runs (model_version) ON DELETE CASCADE,
    cluster_id     INTEGER      NOT NULL CHECK (cluster_id >= 0),
    profile_name   VARCHAR(96)  NOT NULL,
    n_areas        INTEGER      NOT NULL,
    mean_ph        NUMERIC      NOT NULL,
    mean_n         NUMERIC      NOT NULL,
    mean_p         NUMERIC      NOT NULL,
    mean_k         NUMERIC      NOT NULL,
    median_ph      NUMERIC      NOT NULL,
    median_n       NUMERIC      NOT NULL,
    median_p       NUMERIC      NOT NULL,
    median_k       NUMERIC      NOT NULL,
    PRIMARY KEY (model_version, cluster_id)
);

CREATE TABLE IF NOT EXISTS hara_cluster_assignment (
    hara_area_id   INTEGER      NOT NULL REFERENCES hara_bogor (id) ON DELETE CASCADE,
    model_version  VARCHAR(32)  NOT NULL,
    cluster_id     INTEGER      NOT NULL,
    assigned_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (hara_area_id, model_version),
    FOREIGN KEY (model_version, cluster_id)
        REFERENCES nutrient_clusters (model_version, cluster_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS hara_cluster_assignment_version_idx
    ON hara_cluster_assignment (model_version, cluster_id);

CREATE TABLE IF NOT EXISTS cluster_fertilizer_packages (
    id             SERIAL       PRIMARY KEY,
    model_version  VARCHAR(32)  NOT NULL,
    cluster_id     INTEGER      NOT NULL,
    title          VARCHAR(128) NOT NULL,
    content        TEXT         NOT NULL,
    is_draft       BOOLEAN      NOT NULL DEFAULT TRUE,
    updated_by     INTEGER      REFERENCES users (id) ON DELETE SET NULL,
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT cluster_fertilizer_packages_cluster_uq UNIQUE (model_version, cluster_id),
    FOREIGN KEY (model_version, cluster_id)
        REFERENCES nutrient_clusters (model_version, cluster_id) ON DELETE CASCADE
);

COMMENT ON TABLE ml_model_runs IS
    'Riwayat pelatihan model k-means tipe profil hara: parameter, metrik, lokasi artefak. Satu versi aktif.';
COMMENT ON TABLE nutrient_clusters IS
    'Profil rata-rata dan median pH, N, P, K tiap klaster. profile_name deskriptif, bukan tingkat kesuburan.';
COMMENT ON TABLE hara_cluster_assignment IS
    'cluster_id tiap poligon hara_bogor per versi model. Tidak dipakai untuk kelas kesesuaian jagung.';
COMMENT ON TABLE cluster_fertilizer_packages IS
    'Paket anjuran pemupukan per klaster. is_draft = TRUE sampai divalidasi expert.';
