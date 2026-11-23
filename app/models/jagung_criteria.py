from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class JagungCriterion(Base):
    """Satu baris kriteria kesesuaian jagung (lihat migrasi 008_jagung_criteria.sql).

    Satu baris = satu rentang numerik atau satu kategori untuk satu parameter
    pada satu kelas. Tabel ini adalah satu-satunya sumber ambang jagung.
    """

    __tablename__ = "jagung_criteria"
    __table_args__ = (
        UniqueConstraint(
            "rule_set_version",
            "parameter",
            "suitability_class",
            "rule_key",
            name="jagung_criteria_rule_uq",
        ),
        CheckConstraint(
            "parameter IN ('ph', 'n', 'p', 'k', 'lereng', 'tekstur', 'drainase')",
            name="jagung_criteria_parameter_ck",
        ),
        CheckConstraint(
            "suitability_class IN ('S1', 'S2', 'S3', 'N')",
            name="jagung_criteria_class_ck",
        ),
        CheckConstraint(
            "(value_type = 'numeric' AND category IS NULL"
            " AND (min_value IS NOT NULL OR max_value IS NOT NULL))"
            " OR (value_type = 'category' AND category IS NOT NULL"
            " AND min_value IS NULL AND max_value IS NULL)",
            name="jagung_criteria_value_type_ck",
        ),
        CheckConstraint(
            "(improvable AND improvement_step IS NOT NULL"
            " AND improvement_step IN ('ke_S1', 'naik_satu_kelas'))"
            " OR (NOT improvable AND improvement_step IS NULL)",
            name="jagung_criteria_improvement_ck",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    parameter: Mapped[str] = mapped_column(String(16), nullable=False)
    suitability_class: Mapped[str] = mapped_column(String(2), nullable=False)
    rule_key: Mapped[str] = mapped_column(String(32), nullable=False)
    value_type: Mapped[str] = mapped_column(String(8), nullable=False)
    min_value: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    max_value: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    lower_inclusive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    upper_inclusive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    improvable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    improvement_step: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
