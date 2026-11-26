import json

import pytest

from siicl import cli, paths
from tests.builders import AS, assemble, build_npdm, build_nso, build_nup


@pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed")
def test_scan_writes_report_outside_the_repo(tmp_path, capsys):
    text = assemble("svc #0x21\ncasal x0, x1, [x2]\nfadd h0, h1, h2\npaciasp\nretaa\nsvc #0x80", tmp_path)
    container = build_nup(
        [
            ("exefs/main", build_nso(text), True),
            ("exefs/main.npdm", build_npdm(svcs=(0x21, 0x80, 0x01)), True),
            ("romfs/data.bin", b"not scanned", False),
        ]
    )
    src = tmp_path / "game.nup"
    src.write_bytes(container)
    out = tmp_path / "out"

    assert cli.main(["scan", str(src), "--out", str(out), "--jobs", "1"]) == 0

    report = json.loads((out / "report.json").read_text())
    assert report["title_id"] == "0400000000001000"
    assert [m["name"] for m in report["modules"]] == ["main"]
    ids = {f["id"] for f in report["combined"]["features"]}
    assert {"FEAT_LSE", "FEAT_FP16", "FEAT_PAuth_HINT", "FEAT_PAuth"} <= ids
    assert report["combined"]["svc"] == {"0x21": 1, "0x80": 1}
    assert report["npdm"]["caps"]["svcs"] == [0x01, 0x21, 0x80]
    md = (out / "report.md").read_text()
    assert "| 0x80 | SetGpuMemoryAttribute (Switch 2) | yes | 1 |" in md
    assert "| 0x01 | SetHeapSize | yes | 0 |" in md


def test_info_lists_files(tmp_path, capsys):
    src = tmp_path / "game.nup"
    src.write_bytes(build_nup([("exefs/main", b"x" * 10, True)], metadata=[(3, b"\xff\xd8\xff")]))
    assert cli.main(["info", str(src), "--verify"]) == 0
    out = capsys.readouterr().out
    assert "exefs/main" in out
    assert "kind 3 (JPEG image): 3 bytes" in out
    assert "exefs/main: hash matches" in out


def test_refuses_output_inside_the_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    with pytest.raises(paths.OutputPathError):
        paths.ensure_outside_repo(root / "sub", root)
    assert paths.ensure_outside_repo(tmp_path / "elsewhere", root) == (tmp_path / "elsewhere").resolve()


def test_finds_this_repository_as_project_root():
    root = paths.project_root()
    assert root is not None and (root / "src" / "siicl").is_dir()
