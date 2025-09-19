"""
Circuit Breaker Implementation for Real-Time Security Monitoring

Implements circuit breaker pattern for automatic protection against detected attacks.
Provides fail-safe operation with automatic recovery and configurable thresholds.

Key Features:
- Configurable failure thresholds and timeout periods
- Automatic state transitions (CLOSED → OPEN → HALF_OPEN)
- Exponential backoff for recovery attempts
- Thread-safe operation for production environments
- Comprehensive metrics and logging
"""

import logging
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class CircuitBreakerState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, blocking requests
    HALF_OPEN = "half_open"  # Testing if service is recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Number of failures before opening
    timeout_seconds: float = 60.0       # Time to wait before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    monitoring_window_seconds: float = 300.0  # Rolling window for failure tracking
    max_backoff_seconds: float = 300.0  # Maximum backoff time
    backoff_multiplier: float = 2.0     # Exponential backoff multiplier


@dataclass
class CircuitBreakerMetrics:
    """Metrics collected by the circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    state_transitions: Dict[str, int] = field(default_factory=dict)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_backoff_seconds: float = 0.0


class CircuitBreaker:
    """
    Production-ready circuit breaker for real-time security monitoring.

    Automatically protects against cascading failures by:
    - Monitoring request success/failure rates
    - Opening circuit when failure threshold is exceeded
    - Automatically attempting recovery with exponential backoff
    - Providing fail-fast behavior during outages

    Thread-safe and optimized for low-latency operation.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration parameters
            on_state_change: Optional callback for state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # Thread-safe state management
        self._lock = threading.RLock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None

        # Rolling window for failure tracking
        self._recent_failures = deque()

        # Metrics collection
        self.metrics = CircuitBreakerMetrics()

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing fast)."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result if successful

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self.metrics.total_requests += 1

            # Check if we should block the request
            if self._should_block_request():
                self.metrics.blocked_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is {self._state.value}"
                )

        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise

    def _should_block_request(self) -> bool:
        """Check if request should be blocked based on current state."""
        current_time = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            return False

        elif self._state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if (self._last_failure_time and
                current_time - self._last_failure_time >= self._get_current_timeout()):
                self._transition_to_half_open()
                return False
            return True

        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Allow limited requests in half-open state
            return False

        return True

    def _record_success(self, execution_time: float) -> None:
        """Record successful request."""
        with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = time.time()
            self._last_success_time = self.metrics.last_success_time

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitBreakerState.CLOSED:
                # Clean up old failures in rolling window
                self._clean_old_failures()

        self.logger.debug(f"Success recorded (execution_time={execution_time:.3f}s)")

    def _record_failure(self, exception: Exception, execution_time: float) -> None:
        """Record failed request."""
        with self._lock:
            self.metrics.failed_requests += 1
            current_time = time.time()
            self.metrics.last_failure_time = current_time
            self._last_failure_time = current_time

            # Add to recent failures
            self._recent_failures.append(current_time)
            self._clean_old_failures()

            if self._state == CircuitBreakerState.CLOSED:
                if len(self._recent_failures) >= self.config.failure_threshold:
                    self._transition_to_open()

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()

        self.logger.warning(
            f"Failure recorded: {exception} (execution_time={execution_time:.3f}s)"
        )

    def _clean_old_failures(self) -> None:
        """Remove failures outside the monitoring window."""
        current_time = time.time()
        window_start = current_time - self.config.monitoring_window_seconds

        while self._recent_failures and self._recent_failures[0] < window_start:
            self._recent_failures.popleft()

    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.OPEN
        self._failure_count = len(self._recent_failures)
        self._success_count = 0

        # Update backoff time
        self.metrics.current_backoff_seconds = min(
            self.config.timeout_seconds * (self.config.backoff_multiplier ** self._failure_count),
            self.config.max_backoff_seconds
        )

        self._record_state_transition(old_state, self._state)
        self.logger.warning(
            f"Circuit breaker '{self.name}' transitioned to OPEN "
            f"(failures={self._failure_count}, backoff={self.metrics.current_backoff_seconds:.1f}s)"
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.HALF_OPEN
        self._success_count = 0

        self._record_state_transition(old_state, self._state)
        self.logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        old_state = self._state
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self.metrics.current_backoff_seconds = 0.0

        # Clear recent failures
        self._recent_failures.clear()

        self._record_state_transition(old_state, self._state)
        self.logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")

    def _record_state_transition(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState) -> None:
        """Record state transition in metrics."""
        transition_key = f"{old_state.value}_to_{new_state.value}"
        self.metrics.state_transitions[transition_key] = (
            self.metrics.state_transitions.get(transition_key, 0) + 1
        )

        # Call state change callback if provided
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change callback failed: {e}")

    def _get_current_timeout(self) -> float:
        """Get current timeout value with exponential backoff."""
        return max(self.config.timeout_seconds, self.metrics.current_backoff_seconds)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._recent_failures.clear()
            self.metrics.current_backoff_seconds = 0.0

            if old_state != CircuitBreakerState.CLOSED:
                self._record_state_transition(old_state, self._state)

        self.logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the circuit breaker."""
        with self._lock:
            failure_rate = 0.0
            if self.metrics.total_requests > 0:
                failure_rate = self.metrics.failed_requests / self.metrics.total_requests

            return {
                "name": self.name,
                "state": self._state.value,
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "blocked_requests": self.metrics.blocked_requests,
                    "failure_rate": failure_rate,
                    "recent_failures": len(self._recent_failures),
                    "current_backoff_seconds": self.metrics.current_backoff_seconds,
                },
                "state_transitions": self.metrics.state_transitions.copy(),
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                    "success_threshold": self.config.success_threshold,
                    "monitoring_window_seconds": self.config.monitoring_window_seconds,
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized management and monitoring of circuit breakers
    across different components of the security monitoring system.
    """

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def create_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        with self._lock:
            if name in self.breakers:
                raise ValueError(f"Circuit breaker '{name}' already exists")

            breaker = CircuitBreaker(name, config, on_state_change)
            self.breakers[name] = breaker
            self.logger.info(f"Created circuit breaker '{name}'")
            return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self.breakers.get(name)

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        with self._lock:
            return self.breakers.copy()

    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        with self._lock:
            health_status = {
                "total_breakers": len(self.breakers),
                "breakers": {},
                "summary": {
                    "closed": 0,
                    "open": 0,
                    "half_open": 0,
                    "total_requests": 0,
                    "total_failures": 0,
                    "overall_failure_rate": 0.0
                }
            }

            total_requests = 0
            total_failures = 0

            for name, breaker in self.breakers.items():
                breaker_health = breaker.get_health_status()
                health_status["breakers"][name] = breaker_health

                # Update summary
                state = breaker_health["state"]
                health_status["summary"][state] += 1

                metrics = breaker_health["metrics"]
                total_requests += metrics["total_requests"]
                total_failures += metrics["failed_requests"]

            # Calculate overall failure rate
            if total_requests > 0:
                health_status["summary"]["overall_failure_rate"] = total_failures / total_requests

            health_status["summary"]["total_requests"] = total_requests
            health_status["summary"]["total_failures"] = total_failures

            return health_status

    def reset_all_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()
        self.logger.info("All circuit breakers reset")

    def remove_breaker(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self.breakers:
                del self.breakers[name]
                self.logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False