"""Unit tests for asynchronous configuration watcher and hot-reloading."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from src.config import EngineConfig, load_config
from src.config_watcher import ConfigWatcher
from tests.test_config import sample_valid_config_dict


@pytest.mark.asyncio
async def test_watcher_detects_valid_update(tmp_path):
    config_file = tmp_path / "config.yaml"
    initial_data = sample_valid_config_dict()
    config_file.write_text(yaml.dump(initial_data), encoding="utf-8")

    cfg = load_config(config_file)
    watcher = ConfigWatcher(
        config_path=config_file,
        current_config=cfg,
        poll_interval_sec=0.05,
        debounce_sec=0.02,
    )

    reloaded_events = []

    def on_change(old, new):
        reloaded_events.append((old, new))

    watcher.add_callback(on_change)
    await watcher.start()

    try:
        # Update config dwell time
        updated_data = sample_valid_config_dict()
        updated_data["switching"]["min_dwell_time_sec"] = 4.5
        config_file.write_text(yaml.dump(updated_data), encoding="utf-8")

        # Wait for watcher to poll and process
        for _ in range(30):
            if reloaded_events:
                break
            await asyncio.sleep(0.05)

        assert len(reloaded_events) == 1
        old_cfg, new_cfg = reloaded_events[0]
        assert old_cfg.switching.min_dwell_time_sec == 2.0
        assert new_cfg.switching.min_dwell_time_sec == 4.5
        assert watcher.current_config.switching.min_dwell_time_sec == 4.5
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_ignores_invalid_update(tmp_path):
    config_file = tmp_path / "config.yaml"
    initial_data = sample_valid_config_dict()
    config_file.write_text(yaml.dump(initial_data), encoding="utf-8")

    cfg = load_config(config_file)
    watcher = ConfigWatcher(
        config_path=config_file,
        current_config=cfg,
        poll_interval_sec=0.05,
        debounce_sec=0.02,
    )

    reloaded_events = []
    watcher.add_callback(lambda o, n: reloaded_events.append(n))
    await watcher.start()

    try:
        # Write corrupted / invalid YAML (e.g. only 1 camera, violating min_length=2)
        invalid_data = sample_valid_config_dict()
        invalid_data["cameras"] = [invalid_data["cameras"][0]]
        config_file.write_text(yaml.dump(invalid_data), encoding="utf-8")

        await asyncio.sleep(0.2)

        # No callback should have fired, existing config preserved
        assert len(reloaded_events) == 0
        assert len(watcher.current_config.cameras) == 2
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_add_remove_callback(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(sample_valid_config_dict()), encoding="utf-8")

    cfg = load_config(config_file)
    watcher = ConfigWatcher(config_file, cfg)

    cb = lambda o, n: None
    watcher.add_callback(cb)
    assert cb in watcher._callbacks

    watcher.remove_callback(cb)
    assert cb not in watcher._callbacks
