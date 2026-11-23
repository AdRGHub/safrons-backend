"""SQLAlchemy model modules."""

# Model baru SAFRONS Jagung diimpor di sini agar terdaftar di Base.metadata
# setiap kali paket app.models dimuat (termasuk pada Base.metadata.create_all di uji).
from app.models.jagung_criteria import JagungCriterion  # noqa: F401
