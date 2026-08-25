"""Create structured project memory tables."""

from alembic import op
import sqlalchemy as sa

revision = "0001_structured_memory"
down_revision = None
branch_labels = None
depends_on = None

project_stage = sa.Enum("PROJECT_INTAKE", "REQUIREMENT_CONFIRMATION", "SOLUTION_DESIGN", "CUSTOMER_VISIT_MATERIALS", "DEMO_AND_POC", "BUDGET_APPROVAL", "BIDDING_CONTROL", "BID_RESPONSE", "CONTRACT_HANDOVER", "PROJECT_RETROSPECTIVE", name="project_stage")
project_status = sa.Enum("ACTIVE", "ON_HOLD", "WON", "LOST", "CANCELLED", "COMPLETED", name="project_status")
activity_type = sa.Enum("CUSTOMER_VISIT", "PHONE_CALL", "WECHAT_COMMUNICATION", "MEETING", "INTERNAL_DISCUSSION", "EMAIL", "DOCUMENT_REVIEW", "POC_ACTIVITY", "BID_ACTIVITY", "MANUAL_NOTE", "OTHER", name="activity_type")
source_type = sa.Enum("MANUAL", "OBSIDIAN", "EMAIL", "CALENDAR", "MEETING_TRANSCRIPT", "DOCUMENT", "WECHAT", "API", "OTHER", name="source_type")
extraction_status = sa.Enum("PENDING", "PROCESSED", "PARTIAL", "FAILED", "REVIEW_REQUIRED", name="extraction_status")
priority = sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="priority")
requirement_status = sa.Enum("OPEN", "CONFIRMED", "CHANGED", "SATISFIED", "REJECTED", "UNKNOWN", name="requirement_status")
owner_type = sa.Enum("SELF", "CUSTOMER", "TEAM", "SALES", "PARTNER", "OTHER", "UNKNOWN", name="owner_type")
task_status = sa.Enum("OPEN", "IN_PROGRESS", "BLOCKED", "DONE", "CANCELLED", "UNKNOWN", name="task_status")
risk_severity = sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN", name="risk_severity")
risk_status = sa.Enum("OPEN", "MONITORING", "MITIGATED", "RESOLVED", "ACCEPTED", "UNKNOWN", name="risk_status")


def audit_columns():
    return [sa.Column("id", sa.Uuid(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False), sa.PrimaryKeyConstraint("id")]


def upgrade() -> None:
    op.create_table("customers", sa.Column("name", sa.String(255), nullable=False), sa.Column("normalized_name", sa.String(255)), sa.Column("description", sa.Text()), *audit_columns())
    op.create_table("projects", sa.Column("customer_id", sa.Uuid(), nullable=False), sa.Column("name", sa.String(255), nullable=False), sa.Column("description", sa.Text()), sa.Column("stage", project_stage, nullable=False), sa.Column("status", project_status, nullable=False), sa.Column("expected_timeline", sa.Text()), sa.Column("next_action", sa.Text()), sa.ForeignKeyConstraint(["customer_id"], ["customers.id"]), *audit_columns())
    op.create_table("activities", sa.Column("customer_id", sa.Uuid()), sa.Column("project_id", sa.Uuid()), sa.Column("activity_type", activity_type, nullable=False), sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False), sa.Column("raw_content", sa.Text(), nullable=False), sa.Column("summary", sa.Text()), sa.Column("source_type", source_type, nullable=False), sa.Column("source_ref", sa.Text()), sa.Column("participants", sa.JSON()), sa.Column("extraction_status", extraction_status, nullable=False), sa.Column("extraction_version", sa.String(100)), sa.Column("extraction_confidence", sa.Float()), sa.CheckConstraint("extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)", name="ck_activities_extraction_confidence"), sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="SET NULL"), sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="SET NULL"), *audit_columns())
    _create_facts()


def _fact_columns(table: str):
    return [sa.Column("project_id", sa.Uuid(), nullable=False), sa.Column("source_activity_id", sa.Uuid()), sa.Column("title", sa.String(255), nullable=False), sa.Column("confidence", sa.Float()), sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name=f"ck_{table}_confidence"), sa.ForeignKeyConstraint(["project_id"], ["projects.id"]), sa.ForeignKeyConstraint(["source_activity_id"], ["activities.id"], ondelete="SET NULL")]


def _create_facts() -> None:
    op.create_table("requirements", *_fact_columns("requirements"), sa.Column("description", sa.Text(), nullable=False), sa.Column("requirement_type", sa.String(100)), sa.Column("priority", priority), sa.Column("status", requirement_status, nullable=False), *audit_columns())
    op.create_table("tasks", *_fact_columns("tasks"), sa.Column("description", sa.Text()), sa.Column("owner_type", owner_type, nullable=False), sa.Column("owner_name", sa.String(255)), sa.Column("due_at", sa.DateTime(timezone=True)), sa.Column("priority", priority), sa.Column("status", task_status, nullable=False), *audit_columns())
    op.create_table("decisions", *_fact_columns("decisions"), sa.Column("description", sa.Text(), nullable=False), sa.Column("decision_maker", sa.String(255)), sa.Column("decided_at", sa.DateTime(timezone=True)), *audit_columns())
    op.create_table("risks", *_fact_columns("risks"), sa.Column("description", sa.Text(), nullable=False), sa.Column("severity", risk_severity, nullable=False), sa.Column("status", risk_status, nullable=False), sa.Column("mitigation", sa.Text()), *audit_columns())


def downgrade() -> None:
    for table in ("risks", "decisions", "tasks", "requirements", "activities", "projects", "customers"):
        op.drop_table(table)
    bind = op.get_bind()
    for enum_type in (risk_status, risk_severity, task_status, owner_type, requirement_status, priority, extraction_status, source_type, activity_type, project_status, project_stage):
        enum_type.drop(bind, checkfirst=True)
