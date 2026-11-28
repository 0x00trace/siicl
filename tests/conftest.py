"""Fixtures shared by the tests."""

import pytest

from siicl import dynarmic, kernel


@pytest.fixture(params=["unicorn", "dynarmic"])
def cpu_backend(request, monkeypatch):
    """Runs a test once on each CPU backend. A test module asks for it with
    `pytest.mark.usefixtures("cpu_backend")`; every process the test starts then uses that backend."""
    if request.param == "dynarmic":
        if not dynarmic.available():
            pytest.skip("libsiicl_dynarmic.so is not built; see native/CMakeLists.txt")
        monkeypatch.setattr(kernel, "Cpu", dynarmic.Cpu)
    return request.param
