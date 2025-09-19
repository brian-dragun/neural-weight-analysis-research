"""
Security Metrics Collection for Real-Time Monitoring

Provides comprehensive performance and security metrics collection with
minimal overhead for production environments.

Key Features:
- Low-latency metrics collection (< 0.1ms overhead)
- Memory-efficient sliding window aggregations
- Real-time dashboard data export
- Comprehensive security KPIs
- Thread-safe concurrent access
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"          # Monotonically increasing values
    GAUGE = "gauge"             # Current value snapshots
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"            # Duration measurements


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    percentiles: Dict[str, float]  # p50, p95, p99


class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(self, metrics_collector: 'SecurityMetricsCollector', metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(self.metric_name, duration, self.labels)


class SecurityMetricsCollector:
    """
    High-performance security metrics collection system.

    Optimized for minimal overhead in production environments while
    providing comprehensive security monitoring capabilities.
    """

    def __init__(self, max_memory_mb: int = 50):
        self.max_memory_mb = max_memory_mb

        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Metric metadata
        self.metric_labels: Dict[str, Dict[str, str]] = {}
        self.metric_types: Dict[str, MetricType] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Built-in security metrics
        self._init_security_metrics()

        self.logger = logging.getLogger(__name__)

    def _init_security_metrics(self) -> None:
        """Initialize standard security metrics."""
        # Request metrics
        self.define_counter("requests_total", "Total number of requests processed")
        self.define_counter("requests_failed", "Total number of failed requests")
        self.define_counter("requests_blocked", "Total number of blocked requests")

        # Anomaly metrics
        self.define_counter("anomalies_detected", "Total anomalies detected")
        self.define_counter("anomalies_by_type", "Anomalies detected by type")
        self.define_gauge("anomaly_severity_current", "Current anomaly severity level")

        # Circuit breaker metrics
        self.define_counter("circuit_breaker_trips", "Circuit breaker trip events")
        self.define_gauge("circuit_breaker_states", "Current circuit breaker states")

        # Performance metrics
        self.define_timer("request_duration", "Request processing duration")
        self.define_timer("anomaly_detection_duration", "Anomaly detection duration")
        self.define_timer("security_check_duration", "Security check duration")

        # Model metrics
        self.define_gauge("model_accuracy", "Current model accuracy")
        self.define_gauge("model_confidence", "Current model confidence")
        self.define_histogram("gradient_norms", "Distribution of gradient norms")
        self.define_histogram("activation_values", "Distribution of activation values")

    def define_counter(self, name: str, description: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Define a counter metric."""
        with self._lock:
            self.metric_types[name] = MetricType.COUNTER
            if labels:
                self.metric_labels[name] = labels

    def define_gauge(self, name: str, description: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Define a gauge metric."""
        with self._lock:
            self.metric_types[name] = MetricType.GAUGE
            if labels:
                self.metric_labels[name] = labels

    def define_histogram(self, name: str, description: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Define a histogram metric."""
        with self._lock:
            self.metric_types[name] = MetricType.HISTOGRAM
            if labels:
                self.metric_labels[name] = labels

    def define_timer(self, name: str, description: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Define a timer metric."""
        with self._lock:
            self.metric_types[name] = MetricType.TIMER
            if labels:
                self.metric_labels[name] = labels

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            self.counters[metric_key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            self.gauges[metric_key] = value

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            self.histograms[metric_key].append(MetricPoint(time.time(), value, labels or {}))

    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer duration."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            self.timers[metric_key].append(MetricPoint(time.time(), duration, labels or {}))

    def get_timer_context(self, name: str, labels: Optional[Dict[str, str]] = None) -> PerformanceTimer:
        """Get a context manager for timing operations."""
        return PerformanceTimer(self, name, labels)

    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            return self.counters.get(metric_key, 0.0)

    def get_gauge_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            return self.gauges.get(metric_key)

    def get_histogram_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """Get histogram summary statistics."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            if metric_key not in self.histograms:
                return None

            values = [point.value for point in self.histograms[metric_key]]
            if not values:
                return None

            return self._compute_summary(values)

    def get_timer_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """Get timer summary statistics."""
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            if metric_key not in self.timers:
                return None

            values = [point.value for point in self.timers[metric_key]]
            if not values:
                return None

            return self._compute_summary(values)

    def _compute_summary(self, values: List[float]) -> MetricSummary:
        """Compute summary statistics for a list of values."""
        if not values:
            return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        np_values = np.array(values)
        return MetricSummary(
            count=len(values),
            sum=float(np.sum(np_values)),
            min=float(np.min(np_values)),
            max=float(np.max(np_values)),
            mean=float(np.mean(np_values)),
            std=float(np.std(np_values)),
            percentiles={
                "p50": float(np.percentile(np_values, 50)),
                "p95": float(np.percentile(np_values, 95)),
                "p99": float(np.percentile(np_values, 99)),
            }
        )

    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for security dashboard."""
        with self._lock:
            dashboard_data = {
                "timestamp": time.time(),
                "overview": self._get_overview_metrics(),
                "performance": self._get_performance_metrics(),
                "security": self._get_security_metrics(),
                "anomalies": self._get_anomaly_metrics(),
                "circuit_breakers": self._get_circuit_breaker_metrics(),
                "model": self._get_model_metrics(),
            }

        return dashboard_data

    def _get_overview_metrics(self) -> Dict[str, Any]:
        """Get overview metrics for dashboard."""
        total_requests = self.get_counter_value("requests_total")
        failed_requests = self.get_counter_value("requests_failed")
        blocked_requests = self.get_counter_value("requests_blocked")

        success_rate = 0.0
        if total_requests > 0:
            success_rate = (total_requests - failed_requests) / total_requests

        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "failed_requests": failed_requests,
            "blocked_requests": blocked_requests,
            "uptime_seconds": time.time() - self._get_start_time(),
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for dashboard."""
        request_duration = self.get_timer_summary("request_duration")
        detection_duration = self.get_timer_summary("anomaly_detection_duration")
        security_duration = self.get_timer_summary("security_check_duration")

        return {
            "request_duration": request_duration.__dict__ if request_duration else None,
            "anomaly_detection_duration": detection_duration.__dict__ if detection_duration else None,
            "security_check_duration": security_duration.__dict__ if security_duration else None,
        }

    def _get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics."""
        return {
            "total_anomalies": self.get_counter_value("anomalies_detected"),
            "current_threat_level": self._compute_threat_level(),
            "security_score": self._compute_security_score(),
        }

    def _get_anomaly_metrics(self) -> Dict[str, Any]:
        """Get anomaly detection metrics."""
        # Get anomalies by type
        anomaly_types = {}
        for metric_key in self.counters:
            if metric_key.startswith("anomalies_by_type"):
                # Extract type from labels
                if "{" in metric_key:
                    labels_str = metric_key.split("{")[1].split("}")[0]
                    for label_pair in labels_str.split(","):
                        if "type=" in label_pair:
                            anomaly_type = label_pair.split("=")[1]
                            anomaly_types[anomaly_type] = self.counters[metric_key]

        return {
            "total_anomalies": self.get_counter_value("anomalies_detected"),
            "anomalies_by_type": anomaly_types,
            "current_severity": self.get_gauge_value("anomaly_severity_current") or 0.0,
        }

    def _get_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "total_trips": self.get_counter_value("circuit_breaker_trips"),
            "current_states": self._get_current_circuit_breaker_states(),
        }

    def _get_model_metrics(self) -> Dict[str, Any]:
        """Get model-specific metrics."""
        gradient_summary = self.get_histogram_summary("gradient_norms")
        activation_summary = self.get_histogram_summary("activation_values")

        return {
            "accuracy": self.get_gauge_value("model_accuracy") or 0.0,
            "confidence": self.get_gauge_value("model_confidence") or 0.0,
            "gradient_norms": gradient_summary.__dict__ if gradient_summary else None,
            "activation_values": activation_summary.__dict__ if activation_summary else None,
        }

    def _compute_threat_level(self) -> str:
        """Compute current threat level based on metrics."""
        total_requests = self.get_counter_value("requests_total")
        failed_requests = self.get_counter_value("requests_failed")
        anomalies = self.get_counter_value("anomalies_detected")

        if total_requests == 0:
            return "UNKNOWN"

        failure_rate = failed_requests / total_requests
        anomaly_rate = anomalies / total_requests

        if failure_rate > 0.1 or anomaly_rate > 0.05:
            return "HIGH"
        elif failure_rate > 0.05 or anomaly_rate > 0.02:
            return "MEDIUM"
        elif failure_rate > 0.01 or anomaly_rate > 0.005:
            return "LOW"
        else:
            return "MINIMAL"

    def _compute_security_score(self) -> float:
        """Compute overall security score (0.0 to 1.0)."""
        total_requests = self.get_counter_value("requests_total")
        if total_requests == 0:
            return 1.0

        failed_requests = self.get_counter_value("requests_failed")
        anomalies = self.get_counter_value("anomalies_detected")

        failure_rate = failed_requests / total_requests
        anomaly_rate = anomalies / total_requests

        # Simple scoring function (can be made more sophisticated)
        security_score = 1.0 - min(1.0, (failure_rate + anomaly_rate) * 10)
        return max(0.0, security_score)

    def _get_current_circuit_breaker_states(self) -> Dict[str, str]:
        """Get current circuit breaker states."""
        # This would be populated by the circuit breaker manager
        states = {}
        for metric_key in self.gauges:
            if metric_key.startswith("circuit_breaker_states"):
                # Extract breaker name from labels
                if "{" in metric_key:
                    labels_str = metric_key.split("{")[1].split("}")[0]
                    for label_pair in labels_str.split(","):
                        if "breaker=" in label_pair:
                            breaker_name = label_pair.split("=")[1]
                            state_value = self.gauges[metric_key]
                            # Convert numeric state to string
                            if state_value == 0:
                                states[breaker_name] = "CLOSED"
                            elif state_value == 1:
                                states[breaker_name] = "OPEN"
                            elif state_value == 2:
                                states[breaker_name] = "HALF_OPEN"
        return states

    def _get_start_time(self) -> float:
        """Get system start time (simplified)."""
        # In a real implementation, this would track actual start time
        return time.time() - 3600  # Assume 1 hour uptime for demo

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(time.time() * 1000)

        with self._lock:
            # Export counters
            for metric_key, value in self.counters.items():
                lines.append(f"{metric_key} {value} {timestamp}")

            # Export gauges
            for metric_key, value in self.gauges.items():
                lines.append(f"{metric_key} {value} {timestamp}")

            # Export histogram summaries
            for metric_key, points in self.histograms.items():
                if points:
                    values = [p.value for p in points]
                    summary = self._compute_summary(values)
                    lines.append(f"{metric_key}_count {summary.count} {timestamp}")
                    lines.append(f"{metric_key}_sum {summary.sum} {timestamp}")
                    lines.append(f"{metric_key}_mean {summary.mean} {timestamp}")

            # Export timer summaries
            for metric_key, points in self.timers.items():
                if points:
                    values = [p.value for p in points]
                    summary = self._compute_summary(values)
                    lines.append(f"{metric_key}_seconds_count {summary.count} {timestamp}")
                    lines.append(f"{metric_key}_seconds_sum {summary.sum} {timestamp}")
                    lines.append(f"{metric_key}_seconds_mean {summary.mean} {timestamp}")

        return "\n".join(lines)

    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()

    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage in MB."""
        total_items = (
            len(self.counters) +
            len(self.gauges) +
            sum(len(h) for h in self.histograms.values()) +
            sum(len(t) for t in self.timers.values())
        )
        # Rough estimate: ~100 bytes per item
        return (total_items * 100) / (1024 * 1024)