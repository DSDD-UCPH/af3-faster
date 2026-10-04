"""Minimal pytest stand-in so tests run without installing pytest."""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
import types
from pathlib import Path


class _Raises:
    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        self.value = ev
        return True


class _Monkey:
    def __init__(self):
        self._env = []

    def setenv(self, k, v):
        self._env.append((k, os.environ.get(k), k in os.environ))
        os.environ[k] = v

    def delenv(self, k, raising=True):
        if k in os.environ:
            self._env.append((k, os.environ[k], True))
            del os.environ[k]
        elif raising:
            raise KeyError(k)

    def undo(self):
        for k, old, had in reversed(self._env):
            if had:
                os.environ[k] = old
            else:
                os.environ.pop(k, None)


class _DummyPytest(types.ModuleType):
    def raises(self, exc):
        return _Raises(exc)


def _call(fn):
    import inspect
    kw = {}
    mp = None
    tmp = None
    sig = inspect.signature(fn)
    if "monkeypatch" in sig.parameters:
        mp = _Monkey()
        kw["monkeypatch"] = mp
    if "tmp_path" in sig.parameters:
        tmp = tempfile.TemporaryDirectory()
        kw["tmp_path"] = Path(tmp.name)
    try:
        fn(**kw)
    finally:
        if mp:
            mp.undo()
        if tmp:
            tmp.cleanup()


def main():
    here = Path(__file__).resolve().parent
    pkg = here.parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(pkg))
    sys.modules["pytest"] = _DummyPytest("pytest")
    import test_af3_faster as t

    failed = 0
    ran = 0
    for name in sorted(n for n in dir(t) if n.startswith("test_")):
        ran += 1
        try:
            _call(getattr(t, name))
            print("ok", name)
        except Exception:
            failed += 1
            print("FAIL", name)
            traceback.print_exc()
    print(f"ran {ran} failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
