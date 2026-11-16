from __future__ import annotations

import pytest

from operations.build_info import build_fingerprint
from operations.process_lock import trading_process_lock


def test_build_fingerprint_is_stable_and_tracks_source_changes(tmp_path):
    source = tmp_path / "main.py"
    source.write_text("value = 1\n", encoding="utf-8")
    frontend = tmp_path / "dashboard.jsx"
    frontend.write_text("export const value = 1;\n", encoding="utf-8")

    first = build_fingerprint(tmp_path)
    assert build_fingerprint(tmp_path) == first

    source.write_text("value = 2\n", encoding="utf-8")
    changed = build_fingerprint(tmp_path)
    assert changed["digest"] != first["digest"]
    assert changed["source_files"] == 2


def test_build_fingerprint_ignores_local_runtime_and_test_files(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text("assert True\n", encoding="utf-8")
    (tmp_path / ".runtime").mkdir()
    (tmp_path / ".runtime" / "runtime.py").write_text("value = 2\n", encoding="utf-8")

    fingerprint = build_fingerprint(tmp_path)
    assert fingerprint["source_files"] == 1


def test_trading_process_lock_is_exclusive_and_released(tmp_path):
    lock_path = tmp_path / "database" / "trading_service.lock"

    with trading_process_lock(lock_path):
        assert lock_path.read_text(encoding="ascii").strip().isdigit()
        with pytest.raises(RuntimeError, match="already holds the process lock"):
            with trading_process_lock(lock_path):
                pytest.fail("A second trading process lock must not be acquired")

    with trading_process_lock(lock_path):
        assert lock_path.exists()
