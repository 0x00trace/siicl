import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".githooks" / "pre-commit"


@pytest.fixture
def repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _hook(repo: Path, name: str, data: bytes) -> subprocess.CompletedProcess:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_bytes(data)
    subprocess.run(["git", "-C", str(repo), "add", "-f", name], check=True)
    return subprocess.run([str(HOOK)], cwd=repo, capture_output=True, text=True)


@pytest.mark.parametrize(
    "name, data",
    [
        ("shot.png", b"\x89PNG\r\n\x1a\n"),
        ("docs/screen", b"\xff\xd8\xff\xe0 jpeg data"),
        ("main", b"NSO0" + bytes(64)),
        ("game.nup", b"NUP\x01"),
        ("npdm", b"META" + bytes(64)),
        ("clip.dat", bytes(4) + b"ftypisom"),
        ("big.txt", b"a" * (600 * 1024)),
    ],
)
def test_refuses_game_content(repo, name, data):
    result = _hook(repo, name, data)
    assert result.returncode == 1
    assert name in result.stderr


def test_allows_source_files(repo):
    assert _hook(repo, "src/module.py", b"print('hello')\n").returncode == 0


def test_allows_a_submodule(repo):
    # A submodule is staged as a commit of another repository, not as a file of this one.
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},lib/sub"], check=True
    )
    result = subprocess.run([str(HOOK)], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
