-- 008_jagung_criteria.sql
-- Tabel kriteria kesesuaian lahan khusus jagung (SAFRONS Jagung, proyek PPKBP).
--
-- Tabel ini adalah SATU-SATUNYA sumber ambang jagung untuk service
-- app/services/jagung_suitability.py. Tabel lama crop_suitability_criteria dan
-- view hara_crop_suitability (007) tidak diubah dan tetap melayani field
-- crop_suitabilities pada endpoint diagnosis lama.
--
-- Rujukan kriteria:
--   Ritung, S., K. Nugroho, A. Mulyani, dan E. Suryani. 2011. Petunjuk Teknis
--   Evaluasi Lahan untuk Komoditas Pertanian (Edisi Revisi). BBSDLP, Badan
--   Litbang Pertanian, Bogor. Lampiran 1, tabel Jagung (Zea mays).
--
-- Bentuk data: satu baris = satu rentang numerik ATAU satu kategori untuk satu
-- parameter pada satu kelas. Kelas yang punya dua rentang (pH S2 dan S3 di sisi
-- masam dan basa) ditulis sebagai dua baris.
--
-- Aturan pencocokan (dijalankan di service Python, bukan di SQL):
--   * numerik: nilai cocok bila memenuhi batas bawah dan batas atas baris itu;
--     NULL berarti tanpa batas; lower_inclusive/upper_inclusive menentukan
--     apakah batas ikut. Bila lebih dari satu kelas cocok, kelas terbaik dipakai.
--     Nilai yang tidak cocok dengan baris mana pun dianggap N.
--   * kategori: nilai yang sudah dinormalisasi service dicocokkan sama persis
--     dengan kolom category. Kategori yang tidak dikenali = data tidak cukup.
--
-- Kelas potensial (asumsi yang disetujui kelompok, 23 Nov 2026, mengikuti
-- contoh penilaian aktual dan potensial di Juknis):
--   * improvement_step = 'ke_S1': faktor naik ke S1 setelah perbaikan
--     (pH masam lewat pengapuran; N, P, K lewat pemupukan).
--   * improvement_step = 'naik_satu_kelas': faktor naik satu kelas
--     (drainase kelebihan air lewat saluran drainase).
--   * improvable = FALSE: faktor permanen (lereng, tekstur, drainase cepat,
--     pH basa).
--
-- Pemetaan data:
--   * N, P, K: satuan dataset tidak diketahui (Laporan CAPS-32 4.11), sehingga
--     status Juknis (sangat rendah, rendah, sedang, tinggi) diterjemahkan memakai
--     batas NUTRIENT_THRESHOLDS SAFRONS (kuantil 20/40/60/80 dataset), sama
--     dengan diagnosis hara lama. Batas diberi lower_inclusive = FALSE agar sama
--     dengan classify_by_thresholds (nilai tepat di batas masuk status bawah).
--   * Lereng: SLOPE__ berupa rentang kelas; tiap rentang dipetakan ke kelas
--     terburuk yang disentuhnya (0-8 dan 2-8 melewati batas 3% sehingga S2;
--     16-25 melewati batas 15% sehingga N).
--   * Tekstur dan drainase: kategori hasil normalisasi service.
--
-- Mengganti kriteria: tambahkan baris dengan rule_set_version baru, lalu set
-- is_active = FALSE pada versi lama. Baris lama tidak dihapus agar hasil lama
-- tetap dapat ditelusuri. Service menolak bila ada lebih dari satu versi aktif.
--
-- Idempoten: aman dijalankan berulang, baik oleh initdb Docker (psql) maupun
-- oleh runner migrasi di lifespan aplikasi (SQLAlchemy text()). Karena itu
-- berkas ini tidak memakai meta-command psql dan tidak memakai pola titik dua
-- yang diikuti nama.

-- =====================================================================
-- 1. Tabel
-- =====================================================================

CREATE TABLE IF NOT EXISTS jagung_criteria (
    id                SERIAL       PRIMARY KEY,
    rule_set_version  VARCHAR(32)  NOT NULL,
    parameter         VARCHAR(16)  NOT NULL,
    suitability_class VARCHAR(2)   NOT NULL,
    rule_key          VARCHAR(32)  NOT NULL,
    value_type        VARCHAR(8)   NOT NULL,
    min_value         NUMERIC,
    max_value         NUMERIC,
    lower_inclusive   BOOLEAN      NOT NULL DEFAULT TRUE,
    upper_inclusive   BOOLEAN      NOT NULL DEFAULT TRUE,
    category          VARCHAR(32),
    improvable        BOOLEAN      NOT NULL,
    improvement_step  VARCHAR(16),
    source            TEXT         NOT NULL,
    is_active         BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT jagung_criteria_rule_uq
        UNIQUE (rule_set_version, parameter, suitability_class, rule_key),
    CONSTRAINT jagung_criteria_parameter_ck
        CHECK (parameter IN ('ph', 'n', 'p', 'k', 'lereng', 'tekstur', 'drainase')),
    CONSTRAINT jagung_criteria_class_ck
        CHECK (suitability_class IN ('S1', 'S2', 'S3', 'N')),
    CONSTRAINT jagung_criteria_value_type_ck
        CHECK (
            (value_type = 'numeric' AND category IS NULL
                AND (min_value IS NOT NULL OR max_value IS NOT NULL))
            OR (value_type = 'category' AND category IS NOT NULL
                AND min_value IS NULL AND max_value IS NULL)
        ),
    CONSTRAINT jagung_criteria_improvement_ck
        CHECK (
            (improvable AND improvement_step IS NOT NULL
                AND improvement_step IN ('ke_S1', 'naik_satu_kelas'))
            OR (NOT improvable AND improvement_step IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS jagung_criteria_active_idx
    ON jagung_criteria (rule_set_version, parameter)
    WHERE is_active;

COMMENT ON TABLE jagung_criteria IS
    'Kriteria kesesuaian lahan jagung per parameter per kelas (S1/S2/S3/N). Satu-satunya sumber ambang jagung.';
COMMENT ON COLUMN jagung_criteria.rule_key IS
    'Penanda unik rentang/kategori dalam satu kelas, mis. masam, basa, rendah, 41-60, agak_halus.';
COMMENT ON COLUMN jagung_criteria.improvable IS
    'TRUE bila faktor dapat diperbaiki pada kelas potensial; FALSE untuk faktor permanen.';
COMMENT ON COLUMN jagung_criteria.improvement_step IS
    'ke_S1 = naik ke S1 setelah perbaikan; naik_satu_kelas = naik satu kelas; NULL bila permanen.';
COMMENT ON COLUMN jagung_criteria.source IS
    'Rujukan ambang dan catatan pemetaan data.';

-- =====================================================================
-- 2. Seed versi jagung-juknis2011-v1 (34 baris)
-- =====================================================================

INSERT INTO jagung_criteria
    (rule_set_version, parameter, suitability_class, rule_key, value_type,
     min_value, max_value, lower_inclusive, upper_inclusive, category,
     improvable, improvement_step, source)
VALUES
    ('jagung-juknis2011-v1', 'ph', 'S1', 'optimal', 'numeric', 5.8, 7.8, TRUE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'ph', 'S2', 'masam', 'numeric', 5.5, 5.8, TRUE, FALSE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'ph', 'S2', 'basa', 'numeric', 7.8, 8.2, FALSE, TRUE, NULL, FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'ph', 'S3', 'masam', 'numeric', NULL, 5.5, TRUE, FALSE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'ph', 'S3', 'basa', 'numeric', 8.2, NULL, FALSE, TRUE, NULL, FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'n', 'S1', 'sedang_ke_atas', 'numeric', 2.450581, NULL, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'n', 'S2', 'rendah', 'numeric', 1.989691, 2.450581, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'n', 'S3', 'sangat_rendah', 'numeric', NULL, 1.989691, TRUE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'p', 'S1', 'tinggi_ke_atas', 'numeric', 8.022388, NULL, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'p', 'S2', 'sedang', 'numeric', 7.790811, 8.022388, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'p', 'S3', 'rendah_ke_bawah', 'numeric', NULL, 7.790811, TRUE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'k', 'S1', 'tinggi_ke_atas', 'numeric', 331.24494, NULL, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'k', 'S2', 'sedang', 'numeric', 197.49262, 331.24494, FALSE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'k', 'S3', 'rendah_ke_bawah', 'numeric', NULL, 197.49262, TRUE, TRUE, NULL, TRUE, 'ke_S1', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; status hara via NUTRIENT_THRESHOLDS SAFRONS (kuantil dataset)'),
    ('jagung-juknis2011-v1', 'lereng', 'S1', '<2', 'category', NULL, NULL, TRUE, TRUE, '<2', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'S2', '0-8', 'category', NULL, NULL, TRUE, TRUE, '0-8', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'S2', '2-8', 'category', NULL, NULL, TRUE, TRUE, '2-8', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'S3', '9-15', 'category', NULL, NULL, TRUE, TRUE, '9-15', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'N', '16-25', 'category', NULL, NULL, TRUE, TRUE, '16-25', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'N', '26-40', 'category', NULL, NULL, TRUE, TRUE, '26-40', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'N', '41-60', 'category', NULL, NULL, TRUE, TRUE, '41-60', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'lereng', 'N', '>60', 'category', NULL, NULL, TRUE, TRUE, '>60', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; pemetaan kelas SLOPE__ ke kelas terburuk dalam rentang'),
    ('jagung-juknis2011-v1', 'tekstur', 'S1', 'halus', 'category', NULL, NULL, TRUE, TRUE, 'halus', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'tekstur', 'S1', 'agak_halus', 'category', NULL, NULL, TRUE, TRUE, 'agak_halus', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'tekstur', 'S1', 'sedang', 'category', NULL, NULL, TRUE, TRUE, 'sedang', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'tekstur', 'S3', 'agak_kasar', 'category', NULL, NULL, TRUE, TRUE, 'agak_kasar', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'tekstur', 'N', 'kasar', 'category', NULL, NULL, TRUE, TRUE, 'kasar', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung'),
    ('jagung-juknis2011-v1', 'drainase', 'S1', 'baik', 'category', NULL, NULL, TRUE, TRUE, 'baik', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'S1', 'sedang', 'category', NULL, NULL, TRUE, TRUE, 'sedang', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'S2', 'agak_cepat', 'category', NULL, NULL, TRUE, TRUE, 'agak_cepat', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'S2', 'agak_terhambat', 'category', NULL, NULL, TRUE, TRUE, 'agak_terhambat', TRUE, 'naik_satu_kelas', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'S3', 'terhambat', 'category', NULL, NULL, TRUE, TRUE, 'terhambat', TRUE, 'naik_satu_kelas', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'N', 'sangat_terhambat', 'category', NULL, NULL, TRUE, TRUE, 'sangat_terhambat', TRUE, 'naik_satu_kelas', 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME'),
    ('jagung-juknis2011-v1', 'drainase', 'N', 'cepat', 'category', NULL, NULL, TRUE, TRUE, 'cepat', FALSE, NULL, 'Ritung dkk. 2011, Juknis Evaluasi Lahan (Edisi Revisi), Lampiran 1 Jagung; proksi drainase dari SOIL_GREAT dan UNITNAME')
ON CONFLICT ON CONSTRAINT jagung_criteria_rule_uq DO NOTHING;
