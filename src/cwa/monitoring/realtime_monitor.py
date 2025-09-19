"""
Real-Time Security Monitor

Main orchestrator for real-time security monitoring with millisecond-latency
anomaly detection and automatic circuit breaker protection.

Key Features:
- Sub-millisecond monitoring overhead
- Automatic threat detection and response
- Circuit breaker integration for fail-safe operation
- Comprehensive metrics collection
- Production-ready scalability
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import queue
import torch
import torch.nn as nn

from .circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitBreakerConfig, CircuitBreakerState
from .anomaly_detector import AnomalyDetector, DetectionConfig, AnomalyAlert, AnomalyType
from .security_metrics import SecurityMetricsCollector


@dataclass
class MonitoringConfig:
    """Configuration for real-time security monitoring."""
    # Performance settings
    max_latency_ms: float = 1.0          # Maximum allowed monitoring latency
    batch_size: int = 32                 # Batch size for processing
    worker_threads: int = 4              # Number of worker threads
    queue_size: int = 1000              # Maximum queue size

    # Detection settings
    enable_anomaly_detection: bool = True
    enable_circuit_breakers: bool = True
    enable_metrics_collection: bool = True

    # Alert settings
    alert_threshold: float = 0.7         # Minimum severity for alerts
    max_alerts_per_minute: int = 100     # Rate limiting for alerts

    # Circuit breaker settings
    default_circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Monitoring intervals
    health_check_interval_seconds: float = 30.0
    metrics_export_interval_seconds: float = 60.0


@dataclass
class MonitoringRequest:
    """Request for monitoring a model inference."""
    request_id: str
    model: nn.Module
    inputs: torch.Tensor
    outputs: Optional[torch.Tensor] = None
    gradients: Optional[Dict[str, torch.Tensor]] = None
    activations: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MonitoringResponse:
    """Response from security monitoring."""
    request_id: str
    is_safe: bool
    alerts: List[AnomalyAlert]
    latency_ms: float
    circuit_breaker_states: Dict[str, str]
    security_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealtimeSecurityMonitor:
    """
    Production-ready real-time security monitoring system.

    Provides comprehensive security monitoring with:
    - Millisecond-latency anomaly detection
    - Automatic circuit breaker protection
    - Real-time metrics collection
    - Thread-safe concurrent operation
    - Configurable alerting and response
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        on_alert: Optional[Callable[[AnomalyAlert], None]] = None,
        on_circuit_breaker_trip: Optional[Callable[[str, CircuitBreakerState], None]] = None
    ):
        """
        Initialize real-time security monitor.

        Args:
            config: Monitoring configuration
            on_alert: Callback for anomaly alerts
            on_circuit_breaker_trip: Callback for circuit breaker events
        """
        self.config = config or MonitoringConfig()
        self.on_alert = on_alert
        self.on_circuit_breaker_trip = on_circuit_breaker_trip

        # Initialize components
        self.anomaly_detector = AnomalyDetector(DetectionConfig()) if self.config.enable_anomaly_detection else None
        self.circuit_breaker_manager = CircuitBreakerManager() if self.config.enable_circuit_breakers else None
        self.metrics_collector = SecurityMetricsCollector() if self.config.enable_metrics_collection else None

        # Threading and queues
        self.monitoring_queue = queue.Queue(maxsize=self.config.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)
        self.is_running = False

        # Alert rate limiting
        self.alert_history = []
        self.alert_lock = threading.Lock()

        # Health monitoring
        self.last_health_check = time.time()
        self.health_status = {"status": "starting", "last_check": self.last_health_check}

        # Logging
        self.logger = logging.getLogger(__name__)

        # Start background tasks
        self._start_background_tasks()

        self.logger.info("Real-time security monitor initialized")

    def start(self) -> None:
        """Start the monitoring system."""
        if self.is_running:
            self.logger.warning("Monitor is already running")
            return

        self.is_running = True

        # Initialize circuit breakers
        if self.circuit_breaker_manager:
            self._initialize_circuit_breakers()

        # Start worker threads
        for i in range(self.config.worker_threads):
            self.executor.submit(self._worker_loop)

        self.logger.info("Real-time security monitor started")

    def stop(self) -> None:
        """Stop the monitoring system."""
        self.is_running = False

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.logger.info("Real-time security monitor stopped")

    def monitor_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MonitoringResponse:
        """
        Monitor a single model inference for security threats.

        Args:
            model: The neural network model
            inputs: Input tensors
            outputs: Output tensors (optional)
            gradients: Gradient tensors (optional)
            activations: Activation tensors (optional)
            request_id: Unique request identifier
            metadata: Additional metadata

        Returns:
            MonitoringResponse with security assessment
        """
        start_time = time.time()
        request_id = request_id or f"req_{int(time.time()*1000000)}"

        # Record request metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter("requests_total")

        try:
            # Create monitoring request
            monitoring_request = MonitoringRequest(
                request_id=request_id,
                model=model,
                inputs=inputs,
                outputs=outputs,
                gradients=gradients,
                activations=activations,
                metadata=metadata or {},
                timestamp=start_time
            )

            # Perform security checks
            response = self._perform_security_checks(monitoring_request)

            # Record successful request
            if self.metrics_collector:
                self.metrics_collector.increment_counter("requests_total")
                if response.is_safe:
                    latency = time.time() - start_time
                    self.metrics_collector.record_timer("request_duration", latency)

            return response

        except Exception as e:
            # Record failed request
            if self.metrics_collector:
                self.metrics_collector.increment_counter("requests_failed")

            self.logger.error(f"Monitoring failed for request {request_id}: {e}")

            return MonitoringResponse(
                request_id=request_id,
                is_safe=False,  # Fail safe
                alerts=[],
                latency_ms=(time.time() - start_time) * 1000,
                circuit_breaker_states={},
                security_score=0.0,
                metadata={"error": str(e)}
            )

    def _perform_security_checks(self, request: MonitoringRequest) -> MonitoringResponse:
        """Perform comprehensive security checks on a monitoring request."""
        start_time = time.time()
        alerts = []

        # Check circuit breakers first
        if self.circuit_breaker_manager:
            breaker_states = self._check_circuit_breakers()
            if any(state == "OPEN" for state in breaker_states.values()):
                if self.metrics_collector:
                    self.metrics_collector.increment_counter("requests_blocked")

                return MonitoringResponse(
                    request_id=request.request_id,
                    is_safe=False,
                    alerts=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    circuit_breaker_states=breaker_states,
                    security_score=0.0,
                    metadata={"blocked_by": "circuit_breaker"}
                )

        # Perform anomaly detection
        if self.anomaly_detector:
            with self.metrics_collector.get_timer_context("anomaly_detection_duration") if self.metrics_collector else nullcontext():
                detection_alerts = self.anomaly_detector.comprehensive_detection(
                    model=request.model,
                    inputs=request.inputs,
                    gradients=request.gradients,
                    activations=request.activations
                )
                alerts.extend(detection_alerts)

        # Process alerts
        high_severity_alerts = [alert for alert in alerts if alert.severity >= self.config.alert_threshold]

        # Rate limit alerts
        if high_severity_alerts:
            filtered_alerts = self._rate_limit_alerts(high_severity_alerts)
            for alert in filtered_alerts:
                if self.on_alert:
                    try:
                        self.on_alert(alert)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed: {e}")

                # Record alert metrics
                if self.metrics_collector:
                    self.metrics_collector.increment_counter("anomalies_detected")
                    self.metrics_collector.increment_counter(
                        "anomalies_by_type",
                        labels={"type": alert.anomaly_type.value}
                    )
                    self.metrics_collector.set_gauge("anomaly_severity_current", alert.severity)

        # Determine if request is safe
        is_safe = len(high_severity_alerts) == 0

        # Compute security score
        security_score = self._compute_security_score(alerts)

        # Update model metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge("model_confidence", security_score)

        latency_ms = (time.time() - start_time) * 1000

        # Check latency SLA
        if latency_ms > self.config.max_latency_ms:
            self.logger.warning(f"Monitoring latency exceeded SLA: {latency_ms:.2f}ms > {self.config.max_latency_ms}ms")

        return MonitoringResponse(
            request_id=request.request_id,
            is_safe=is_safe,
            alerts=alerts,
            latency_ms=latency_ms,
            circuit_breaker_states=self._check_circuit_breakers(),
            security_score=security_score,
            metadata={
                "detection_count": len(alerts),
                "high_severity_count": len(high_severity_alerts)
            }
        )

    def _check_circuit_breakers(self) -> Dict[str, str]:
        """Check current state of all circuit breakers."""
        if not self.circuit_breaker_manager:
            return {}

        states = {}
        for name, breaker in self.circuit_breaker_manager.get_all_breakers().items():
            states[name] = breaker.state.value

            # Update metrics
            if self.metrics_collector:
                state_value = {"closed": 0, "open": 1, "half_open": 2}[breaker.state.value]
                self.metrics_collector.set_gauge(
                    "circuit_breaker_states",
                    state_value,
                    labels={"breaker": name}
                )

        return states

    def _rate_limit_alerts(self, alerts: List[AnomalyAlert]) -> List[AnomalyAlert]:
        """Apply rate limiting to alerts."""
        with self.alert_lock:
            current_time = time.time()

            # Clean old alerts (older than 1 minute)
            self.alert_history = [
                timestamp for timestamp in self.alert_history
                if current_time - timestamp < 60.0
            ]

            # Check rate limit
            if len(self.alert_history) >= self.config.max_alerts_per_minute:
                self.logger.warning("Alert rate limit exceeded, dropping alerts")
                return []

            # Add new alerts to history
            filtered_alerts = alerts[:self.config.max_alerts_per_minute - len(self.alert_history)]
            for _ in filtered_alerts:
                self.alert_history.append(current_time)

            return filtered_alerts

    def _compute_security_score(self, alerts: List[AnomalyAlert]) -> float:
        """Compute overall security score based on alerts."""
        if not alerts:
            return 1.0

        # Weight alerts by severity
        total_severity = sum(alert.severity for alert in alerts)
        max_possible_severity = len(alerts) * 1.0

        # Invert to get security score (higher is better)
        security_score = 1.0 - min(1.0, total_severity / max_possible_severity)
        return max(0.0, security_score)

    def _initialize_circuit_breakers(self) -> None:
        """Initialize standard circuit breakers."""
        if not self.circuit_breaker_manager:
            return

        # Main inference circuit breaker
        self.circuit_breaker_manager.create_breaker(
            "inference",
            self.config.default_circuit_breaker_config,
            self._on_circuit_breaker_state_change
        )

        # Anomaly detection circuit breaker
        self.circuit_breaker_manager.create_breaker(
            "anomaly_detection",
            self.config.default_circuit_breaker_config,
            self._on_circuit_breaker_state_change
        )

        self.logger.info("Circuit breakers initialized")

    def _on_circuit_breaker_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState) -> None:
        """Handle circuit breaker state changes."""
        if self.metrics_collector:
            self.metrics_collector.increment_counter("circuit_breaker_trips")

        if self.on_circuit_breaker_trip:
            try:
                self.on_circuit_breaker_trip("circuit_breaker", new_state)
            except Exception as e:
                self.logger.error(f"Circuit breaker callback failed: {e}")

        self.logger.warning(f"Circuit breaker state change: {old_state.value} -> {new_state.value}")

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        def health_check_loop():
            while True:
                try:
                    self._perform_health_check()
                    time.sleep(self.config.health_check_interval_seconds)
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")

        def metrics_export_loop():
            while True:
                try:
                    if self.metrics_collector:
                        dashboard_data = self.metrics_collector.get_security_dashboard_data()
                        self.logger.debug(f"Security metrics: {dashboard_data['overview']}")
                    time.sleep(self.config.metrics_export_interval_seconds)
                except Exception as e:
                    self.logger.error(f"Metrics export failed: {e}")

        # Start background threads
        threading.Thread(target=health_check_loop, daemon=True).start()
        threading.Thread(target=metrics_export_loop, daemon=True).start()

    def _perform_health_check(self) -> None:
        """Perform system health check."""
        current_time = time.time()
        health_status = {
            "status": "healthy" if self.is_running else "stopped",
            "last_check": current_time,
            "components": {}
        }

        # Check anomaly detector
        if self.anomaly_detector:
            try:
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()
                health_status["components"]["anomaly_detector"] = {
                    "status": "healthy",
                    "total_alerts": anomaly_summary["total_alerts"]
                }
            except Exception as e:
                health_status["components"]["anomaly_detector"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        # Check circuit breaker manager
        if self.circuit_breaker_manager:
            try:
                system_health = self.circuit_breaker_manager.get_system_health()
                health_status["components"]["circuit_breakers"] = {
                    "status": "healthy",
                    "total_breakers": system_health["total_breakers"],
                    "open_breakers": system_health["summary"]["open"]
                }
            except Exception as e:
                health_status["components"]["circuit_breakers"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        # Check metrics collector
        if self.metrics_collector:
            try:
                memory_usage = self.metrics_collector.get_memory_usage_mb()
                health_status["components"]["metrics_collector"] = {
                    "status": "healthy" if memory_usage < 100 else "warning",
                    "memory_usage_mb": memory_usage
                }
            except Exception as e:
                health_status["components"]["metrics_collector"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        self.health_status = health_status
        self.last_health_check = current_time

    def _worker_loop(self) -> None:
        """Worker thread loop for processing monitoring requests."""
        while self.is_running:
            try:
                # Get request from queue (with timeout)
                request = self.monitoring_queue.get(timeout=1.0)

                # Process request
                response = self._perform_security_checks(request)

                # Mark task as done
                self.monitoring_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker thread error: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the monitoring system."""
        return self.health_status.copy()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.metrics_collector:
            return {"error": "Metrics collection disabled"}

        return self.metrics_collector.get_security_dashboard_data()

    def reset_baseline(self, model: nn.Module) -> None:
        """Reset baseline for anomaly detection."""
        if self.anomaly_detector:
            self.anomaly_detector.reset_baseline(model)

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if not self.metrics_collector:
            return ""

        if format == "prometheus":
            return self.metrics_collector.export_prometheus_metrics()
        else:
            raise ValueError(f"Unsupported metrics format: {format}")


# Context manager for null operations
class nullcontext:
    """Null context manager for Python < 3.7 compatibility."""
    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        return False