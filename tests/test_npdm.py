import pytest

from siicl import npdm
from tests.builders import build_npdm


def test_parses_header_and_kernel_capabilities():
    n = npdm.parse(build_npdm(svcs=(0x01, 0x21, 0x2F, 0x80, 0x88)))
    assert n.name == "Application"
    assert n.is_64bit
    assert n.address_space == "64-bit, 39-bit address space"
    assert n.main_thread_priority == 44
    assert n.main_thread_stack_size == 0x100000
    assert n.program_id == 0x0400_0000_0000_1000
    c = n.caps
    assert c.svcs == {0x01, 0x21, 0x2F, 0x80, 0x88}
    assert (c.highest_priority, c.lowest_priority, c.min_core, c.max_core) == (24, 59, 0, 3)
    assert c.kernel_version == (20, 5)
    assert c.handle_table_size == 1023
    assert c.program_type == 6
    assert c.misc_flags == 1
    assert n.fs_permissions == 0x8000_0000_0000_0001


def test_parses_service_access():
    n = npdm.parse(build_npdm(services=(("fsp-srv", False), ("vi:m", False), ("my:srv", True))))
    assert [(s.name, s.server) for s in n.services] == [("fsp-srv", False), ("vi:m", False), ("my:srv", True)]


def test_parses_acid_capabilities_separately():
    n = npdm.parse(build_npdm(svcs=(0x01,), acid_svcs=(0x01, 0x7F)))
    assert n.caps.svcs == {0x01}
    assert n.acid_caps is not None and n.acid_caps.svcs == {0x01, 0x7F}


@pytest.mark.parametrize("value, kind", [(0b0111, 3), (0b01111, 4), (0xFFFF, 16), (0, 0), (0xFFFFFFFF, 32)])
def test_trailing_ones(value, kind):
    assert npdm.trailing_ones(value) == kind


def test_rejects_other_files():
    with pytest.raises(npdm.NpdmError, match="META"):
        npdm.parse(b"NSO0" + bytes(0x100))
