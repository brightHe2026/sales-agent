"""Add exact source quotes for grounded derived facts."""

from alembic import op
import sqlalchemy as sa


revision = "0003_fact_source_quotes"
down_revision = "0002_fact_fingerprints"
branch_labels = None
depends_on = None


TABLES = ("requirements", "tasks", "decisions", "risks")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("source_quote", sa.Text(), nullable=True))


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_column(table, "source_quote")
