"""Add source fingerprints for idempotent memory updates."""

from alembic import op
import sqlalchemy as sa


revision = "0002_fact_fingerprints"
down_revision = "0001_structured_memory"
branch_labels = None
depends_on = None


TABLES = ("requirements", "tasks", "decisions", "risks")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("source_fingerprint", sa.String(length=64), nullable=True))
        op.create_unique_constraint(
            f"uq_{table}_source_fingerprint",
            table,
            ["source_activity_id", "source_fingerprint"],
        )


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_constraint(f"uq_{table}_source_fingerprint", table, type_="unique")
        op.drop_column(table, "source_fingerprint")
