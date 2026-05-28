from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    feature_15: Mapped[float] = mapped_column(Float, nullable=False)
    feature_16: Mapped[float] = mapped_column(Float, nullable=False)
    feature_19: Mapped[float] = mapped_column(Float, nullable=False)
    feature_20: Mapped[float] = mapped_column(Float, nullable=False)
    feature_9: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
