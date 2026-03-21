"""
TaskPool - Async task pool with concurrency limiting and graceful shutdown.

This module provides an async task pool implementation using semaphores
for concurrency control, with support for task tracking, cancellation,
and graceful shutdown.
"""
import asyncio
import logging
from typing import Any, Callable, Coroutine, Set, TypeVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class TaskPool:
    """Async task pool with concurrency limiting and graceful shutdown.

    Features:
    - Semaphore-based concurrency control
    - Task tracking and cancellation
    - Graceful shutdown with timeout
    - Statistics (active, completed, failed)

    Example:
        >>> import asyncio
        >>> from petals.client.async import TaskPool
        >>>
        >>> async def my_task(i):
        ...     await asyncio.sleep(0.1)
        ...     return f"result_{i}"
        >>>
        >>> async def main():
        ...     pool = TaskPool(max_concurrency=5)
        ...     results = await asyncio.gather(
        ...         pool.run(my_task, i) for i in range(10)
        ...     )
        ...     print(f"Stats: {pool.stats}")
        ...     await pool.shutdown()
        >>>
        >>> asyncio.run(main())
    """

    max_concurrency: int = 10
    shutdown_timeout: float = 30.0

    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _active_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)
    _completed_count: int = field(default=0)
    _failed_count: int = field(default=0)
    _cancelled_count: int = field(default=0)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the semaphore and lock after dataclass creation."""
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._active_tasks = set()
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0

    async def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Submit a coroutine to the pool.

        The coroutine will be wrapped with automatic semaphore acquire/release
        and task tracking.

        Args:
            coro: The coroutine to execute.

        Returns:
            asyncio.Task that can be awaited or cancelled.

        Example:
            >>> task = pool.submit(my_coroutine())
            >>> result = await task
        """
        async def _wrapped_coro() -> T:
            async with self._semaphore:
                return await coro

        task: asyncio.Task[T] = asyncio.create_task(_wrapped_coro())

        async with self._lock:
            self._active_tasks.add(task)

        # Add done callback to track completion
        def _on_done(t: asyncio.Task) -> None:
            asyncio.create_task(self._task_done_callback(t))

        task.add_done_callback(_on_done)

        logger.debug(f"Task submitted, active: {len(self._active_tasks)}")
        return task

    async def _task_done_callback(self, task: asyncio.Task) -> None:
        """Handle task completion callback.

        Updates statistics and removes task from active set.

        Args:
            task: The completed task.
        """
        async with self._lock:
            self._active_tasks.discard(task)

            if task.cancelled():
                self._cancelled_count += 1
                logger.debug("Task cancelled")
            elif task.exception() is not None:
                self._failed_count += 1
                logger.debug(f"Task failed: {task.exception()}")
            else:
                self._completed_count += 1
                logger.debug("Task completed successfully")

    async def run(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Run a function in the pool with automatic semaphore acquire/release.

        This is a convenience method that submits and awaits in one call.

        Args:
            func: An async function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the coroutine.

        Example:
            >>> result = await pool.run(async_http_call, url, timeout=30)
        """
        task = await self.submit(func(*args, **kwargs))
        return await task

    async def shutdown(self, cancel_pending: bool = False) -> None:
        """Gracefully shutdown the pool.

        Args:
            cancel_pending: If True, cancel all pending tasks.
                          If False, wait for them to complete.

        Example:
            >>> await pool.shutdown()  # Wait for pending tasks
            >>> await pool.shutdown(cancel_pending=True)  # Cancel immediately
        """
        logger.info(f"Shutting down TaskPool (cancel_pending={cancel_pending})")

        if cancel_pending:
            # Cancel all active tasks
            async with self._lock:
                active_copy = list(self._active_tasks)
                for task in active_copy:
                    task.cancel()

            # Wait for cancelled tasks to finalize
            if active_copy:
                await asyncio.wait(active_copy, timeout=self.shutdown_timeout)
        else:
            # Wait for all tasks to complete
            if self._active_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.wait(
                            self._active_tasks,
                            return_when=asyncio.ALL_COMPLETED
                        ),
                        timeout=self.shutdown_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Shutdown timeout after {self.shutdown_timeout}s, "
                        f"cancelling {len(self._active_tasks)} remaining tasks"
                    )
                    for task in self._active_tasks:
                        task.cancel()

        # Final cleanup
        async with self._lock:
            self._active_tasks.clear()

        logger.info(
            f"TaskPool shutdown complete. "
            f"Stats: {self.stats}"
        )

    @property
    def stats(self) -> dict:
        """Return current pool statistics.

        Returns:
            Dictionary with active, completed, failed, and cancelled counts.

        Example:
            >>> print(pool.stats)
            {'active': 3, 'completed': 100, 'failed': 2, 'cancelled': 1}
        """
        return {
            "active": len(self._active_tasks),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "cancelled": self._cancelled_count,
            "max_concurrency": self.max_concurrency,
        }

    @property
    def num_active(self) -> int:
        """Return the number of currently active tasks.

        Returns:
            Number of active tasks.
        """
        return len(self._active_tasks)

    def is_idle(self) -> bool:
        """Check if the pool has no active tasks.

        Returns:
            True if no tasks are currently active.
        """
        return len(self._active_tasks) == 0

    async def wait_idle(self, timeout: float = None) -> bool:
        """Wait until all active tasks complete.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if pool became idle, False if timeout occurred.
        """
        if self.is_idle():
            return True

        async def _wait_for_idle() -> None:
            while not self.is_idle():
                await asyncio.sleep(0.01)

        try:
            if timeout is None:
                await _wait_for_idle()
            else:
                await asyncio.wait_for(_wait_for_idle(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
