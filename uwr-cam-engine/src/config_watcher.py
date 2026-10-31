"""Asynchronous configuration file watcher with debounced hot-reloading."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Union

from .config import EngineConfig, load_config

logger = logging.getLogger(__name__)

ConfigCallback = Callable[[EngineConfig, EngineConfig], Union[None, Awaitable[None]]]


class ConfigWatcher:
    """Monitors config.yaml for changes, reloads and validates dynamically."""

    def __init__(
        self,
        config_path: str | Path,
        current_config: EngineConfig,
        poll_interval_sec: float = 1.0,
        debounce_sec: float = 0.5,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.current_config = current_config
        self.poll_interval_sec = poll_interval_sec
        self.debounce_sec = debounce_sec
        self._callbacks: List[ConfigCallback] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_mtime: Optional[int] = self._get_mtime()

    def _get_mtime(self) -> Optional[int]:
        try:
            return self.config_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning(f"Failed to stat config file {self.config_path}: {e}")
            return None

    def add_callback(self, callback: ConfigCallback) -> None:
        """Register a callback invoked when a valid new configuration is detected."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ConfigCallback) -> None:
        """Unregister a configuration change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def start(self) -> None:
        """Start the background monitoring task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._watch_loop(), name="config-watcher")
        logger.info(f"Started config watcher for {self.config_path} (polling every {self.poll_interval_sec}s)")

    async def stop(self) -> None:
        """Stop the background monitoring task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Config watcher stopped")

    async def _watch_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_sec)
                current_mtime = self._get_mtime()

                if current_mtime is not None and current_mtime != self._last_mtime:
                    # Debounce: wait briefly for file writing to finish
                    await asyncio.sleep(self.debounce_sec)
                    current_mtime = self._get_mtime()
                    self._last_mtime = current_mtime
                    await self._handle_file_change()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in config watcher loop: {e}", exc_info=True)

    async def _handle_file_change(self) -> None:
        logger.info(f"Detected modification in {self.config_path.name}. Validating new configuration...")
        try:
            new_config = load_config(self.config_path)
        except Exception as e:
            logger.error(f"Hot-reload rejected: Configuration validation failed: {e}. Keeping current active configuration.")
            return

        if new_config == self.current_config:
            logger.debug("Config file updated but contents are identical. No action needed.")
            return

        old_config = self.current_config
        logger.info("New configuration validated successfully. Applying runtime updates...")

        # Dispatch callbacks
        for cb in self._callbacks:
            try:
                result = cb(old_config, new_config)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Hot-reload rejected: Error applying configuration: {e}. Keeping current active configuration.")
                return

        self.current_config = new_config
        logger.info("New configuration applied successfully.")
