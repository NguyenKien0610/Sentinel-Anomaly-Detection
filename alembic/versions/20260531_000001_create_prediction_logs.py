"""create prediction_logs table

Revision ID: 20260531_000001
Revises: 
Create Date: 2026-05-31 00:00:01
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260531_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("prediction_logs"):
        op.create_table(
            "prediction_logs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("feature_15", sa.Float(), nullable=False),
            sa.Column("feature_16", sa.Float(), nullable=False),
            sa.Column("feature_19", sa.Float(), nullable=False),
            sa.Column("feature_20", sa.Float(), nullable=False),
            sa.Column("feature_9", sa.Float(), nullable=False),
            sa.Column("prediction", sa.String(length=20), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    indexes = {index["name"] for index in inspector.get_indexes("prediction_logs")}
    index_name = op.f("ix_prediction_logs_id")
    if index_name not in indexes:
        op.create_index(index_name, "prediction_logs", ["id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    index_name = op.f("ix_prediction_logs_id")

    if inspector.has_table("prediction_logs"):
        indexes = {index["name"] for index in inspector.get_indexes("prediction_logs")}
        if index_name in indexes:
            op.drop_index(index_name, table_name="prediction_logs")
        op.drop_table("prediction_logs")
