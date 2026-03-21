"""
Retry logic with backoff and circuit breaker pattern.

This module provides configurable retry policies with various backoff
strategies and a circuit breaker for preventing cascade failures.

Example:
    >>> from petals.client.feedback.retry_policy import RetryPolicy, CircuitBreaker
    >>>
    >>> # Simple retry with exponential backoff
    >>> policy = RetryPolicy(max_attempts=3, base_delay=1.0)
    >>> result = await policy.execute(async_function, arg1, arg2)
    >>>
    >>> # Circuit breaker for preventing cascade failures
    >>> cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
    >>> result = await cb.call(async_function, arg1)
"""
import asyncio
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Tuple, Type, Dict

import logging

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when all retries are exhausted.

    Attributes:
        attempts: Number of retry attempts made.
        last_exception: The last exception that was raised.
    """
    def __init__(self, message: str, attempts: int = 0, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BackoffStrategy:
    """Backoff strategies for retries."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"


@dataclass
class RetryPolicy:
    """Configurable retry policy with backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        backoff_strategy: Strategy for computing delay.
        jitter: Whether to add randomness to delay.
        retryable_exceptions: Tuple of exception types to retry.
        on_retry: Optional callback on each retry.
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: str = BackoffStrategy.EXPONENTIAL_WITH_JITTER
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[int, Exception], None]] = None

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry.
        """
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay

        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL_WITH_JITTER:
            delay = self.base_delay * (2 ** attempt)
            # Add jitter
            if self.jitter > 0:
                jitter_range = delay * self.jitter
                delay += random.uniform(-jitter_range, jitter_range)

        else:
            delay = self.base_delay

        # Cap at max_delay
        return min(max(0, delay), self.max_delay)

    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable.

        Args:
            exception: The exception to check.

        Returns:
            True if the exception should trigger a retry.
        """
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        return False

    async def execute(
        self,
        coro_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute coroutine with retry logic.

        Args:
            coro_func: Coroutine function to execute.
            *args: Positional arguments to pass to coro_func.
            **kwargs: Keyword arguments to pass to coro_func.

        Returns:
            Result of successful execution.

        Raises:
            RetryError: If all retries are exhausted.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                result = await coro_func(*args, **kwargs)
                if attempt > 0:
                    logger.debug(f"Success after {attempt + 1} attempts")
                return result

            except Exception as e:
                last_exception = e

                if not self.is_retryable(e):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise

                if attempt == self.max_attempts - 1:
                    logger.warning(f"All {self.max_attempts} attempts exhausted")
                    raise RetryError(
                        f"Failed after {self.max_attempts} attempts: {e}",
                        attempts=self.max_attempts,
                        last_exception=e
                    )

                delay = self.compute_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if self.on_retry:
                    self.on_retry(attempt + 1, e)

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise RetryError(
            f"Failed after {self.max_attempts} attempts",
            attempts=self.max_attempts,
            last_exception=last_exception
        )

    def sync_execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute synchronous function with retry logic.

        This is a convenience method for non-async functions.

        Args:
            func: Synchronous function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of successful execution.

        Raises:
            RetryError: If all retries are exhausted.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.debug(f"Success after {attempt + 1} attempts")
                return result

            except Exception as e:
                last_exception = e

                if not self.is_retryable(e):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise

                if attempt == self.max_attempts - 1:
                    logger.warning(f"All {self.max_attempts} attempts exhausted")
                    raise RetryError(
                        f"Failed after {self.max_attempts} attempts: {e}",
                        attempts=self.max_attempts,
                        last_exception=e
                    )

                delay = self.compute_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if self.on_retry:
                    self.on_retry(attempt + 1, e)

                import time
                time.sleep(delay)

        raise RetryError(
            f"Failed after {self.max_attempts} attempts",
            attempts=self.max_attempts,
            last_exception=last_exception
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker pattern configuration.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before attempting recovery.
        half_open_max_calls: Max calls allowed in half-open state.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures.

    States:
    - CLOSED: Normal operation, requests pass through.
    - OPEN: Failing fast, requests are rejected immediately.
    - HALF_OPEN: Testing recovery, limited requests allowed.

    Example:
        >>> config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
        >>> breaker = CircuitBreaker(config)
        >>>
        >>> try:
        ...     result = await breaker.call(some_async_function, arg1, arg2)
        ... except CircuitOpenError:
        ...     print("Circuit is open, request rejected")
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration.
        """
        self.config = config or CircuitBreakerConfig()
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state.

        Returns:
            Current state string.
        """
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count.

        Returns:
            Number of consecutive failures.
        """
        return self._failure_count

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state.

        Returns:
            True if request should proceed.
        """
        if self._state == self.STATE_CLOSED:
            return True

        if self._state == self.STATE_OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time is not None:
                import time
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    return True
            return False

        if self._state == self.STATE_HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    def _record_success(self) -> None:
        """Record successful call and update state accordingly."""
        if self._state == self.STATE_HALF_OPEN:
            self._success_count += 1
            # If enough successes in half-open, close the circuit
            if self._success_count >= self.config.half_open_max_calls:
                logger.info("Circuit breaker closing after successful recovery")
                self._state = self.STATE_CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state == self.STATE_CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record failed call and update state accordingly."""
        self._failure_count += 1
        self._last_failure_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None

        import time
        self._last_failure_time = time.time()

        if self._state == self.STATE_HALF_OPEN:
            # Any failure in half-open immediately opens circuit
            logger.warning("Circuit breaker opening after failure in half-open state")
            self._state = self.STATE_OPEN
            self._success_count = 0

        elif self._state == self.STATE_CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker opening after {self._failure_count} failures"
                )
                self._state = self.STATE_OPEN

    async def call(self, coro_func: Callable, *args, **kwargs) -> Any:
        """Execute with circuit breaker protection.

        Args:
            coro_func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function call.

        Raises:
            CircuitOpenError: If circuit is open.
        """
        async with self._lock:
            if not self._should_allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker is {self._state}. "
                    f"Failure count: {self._failure_count}"
                )

            if self._state == self.STATE_HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await coro_func(*args, **kwargs)
            async with self._lock:
                self._record_success()
            return result

        except Exception as e:
            async with self._lock:
                self._record_failure()
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        logger.info("Circuit breaker reset to closed state")

    def get_stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with current statistics.
        """
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_calls": self._half_open_calls,
            "last_failure_time": self._last_failure_time,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different providers.

    This class maintains a collection of circuit breakers, one per provider,
    allowing independent failure tracking and recovery for each service.

    Example:
        >>> manager = CircuitBreakerManager()
        >>> config = CircuitBreakerConfig(failure_threshold=3)
        >>> breaker = manager.get_breaker("openai", config)
        >>>
        >>> # Get all breaker states
        >>> states = manager.get_all_states()
        >>> print(states)  # {"openai": "closed"}
    """

    def __init__(self):
        """Initialize the circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, CircuitBreakerConfig] = {}

    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider.

        Args:
            name: Provider name/identifier.
            config: Optional circuit breaker configuration. If provided and
                a breaker already exists, the config is ignored.

        Returns:
            CircuitBreaker instance for the provider.
        """
        if name not in self._breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self._configs[name] = config
            self._breakers[name] = CircuitBreaker(config)

        return self._breakers[name]

    def get_all_states(self) -> Dict[str, str]:
        """Get the state of all circuit breakers.

        Returns:
            Dictionary mapping provider names to their circuit states.
        """
        return {name: breaker.state for name, breaker in self._breakers.items()}

    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping provider names to their statistics.
        """
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.reset()

    def remove_breaker(self, name: str) -> None:
        """Remove a circuit breaker for a provider.

        Args:
            name: Provider name/identifier.
        """
        if name in self._breakers:
            del self._breakers[name]
        if name in self._configs:
            del self._configs[name]


@dataclass
class RetryWithCircuitBreaker:
    """Combined retry policy with circuit breaker protection.

    Combines retry logic with circuit breaker for robust error handling.

    Attributes:
        retry_policy: The retry policy to use.
        circuit_breaker: The circuit breaker to use.
    """
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def execute(self, coro_func: Callable, *args, **kwargs) -> Any:
        """Execute with retry and circuit breaker protection.

        Args:
            coro_func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function call.

        Raises:
            CircuitOpenError: If circuit is open.
            RetryError: If all retries are exhausted.
        """
        return await self.circuit_breaker.call(
            self.retry_policy.execute,
            coro_func,
            *args,
            **kwargs
        )
