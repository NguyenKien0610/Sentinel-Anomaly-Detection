from pydantic import BaseModel

from src.core.config import get_settings


class ServerMetrics(BaseModel):
    feature_15: float
    feature_16: float
    feature_19: float
    feature_20: float
    feature_9: float

    def to_feature_payload(self) -> dict[str, float]:
        payload = self.model_dump()
        return {
            feature_name: float(payload[feature_name])
            for feature_name in get_settings().feature_columns
        }
