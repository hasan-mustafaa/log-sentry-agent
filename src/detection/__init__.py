"""
Detection package.

Converts raw log and metrics streams into structured anomaly signals through
a four-stage pipeline:

  1. LogParser        — mines structured templates from raw log lines (Drain)
  2. FeatureExtractor — builds time-windowed multivariate feature vectors
  3. StatisticalDetector — Z-score / moving-average single-metric detection
  4. MLDetector       — Isolation Forest multivariate detection

The two detector outputs are combined by FeatureExtractor into an ensemble
score using configurable weights (config.yaml → detection.ensemble_weights).

Typical usage:

    from src.detection import LogParser, FeatureExtractor
    from src.detection import StatisticalDetector, MLAnomalyDetector

    parser    = LogParser(config["detection"])
    extractor = FeatureExtractor(config["detection"])
    stat_det  = StatisticalDetector(config["detection"])
    ml_det    = MLAnomalyDetector(config["detection"]["isolation_forest"])
"""

from src.detection.log_parser import LogParser
from src.detection.feature_extractor import FeatureExtractor
from src.detection.statistical_detector import StatisticalDetector
from src.detection.ml_detector import MLAnomalyDetector

__all__ = ["LogParser", "FeatureExtractor", "StatisticalDetector", "MLAnomalyDetector"]
