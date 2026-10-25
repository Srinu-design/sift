"""deploy.sh --from-ssh: the CI key's forced command accepts one word and never a command."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "deploy.sh"


def from_ssh(root: Path, word: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SIFT_ROOT": str(root), "SSH_ORIGINAL_COMMAND": word}
    return subprocess.run(
        ["bash", str(SCRIPT), "--from-ssh"], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    "word",
    ["", "main", "touch pwned", "a" * 40 + "; touch pwned", "$(touch pwned)", "--help", "A" * 40],
)
def test_anything_but_one_word_is_refused_before_running(tmp_path: Path, word: str) -> None:
    result = from_ssh(tmp_path, word)
    assert result.returncode == 2
    assert "expected a commit sha" in result.stderr
    assert list(tmp_path.iterdir()) == []  # no log, no lock: nothing ran


def test_a_sha_reaches_the_deploy_and_is_logged(tmp_path: Path) -> None:
    result = from_ssh(tmp_path, "0" * 40)
    assert result.returncode == 2  # no clone under SIFT_ROOT, so git fetch fails
    assert "git fetch failed" in result.stderr  # followed from the detached child's log
    assert len(list((tmp_path / "logs").iterdir())) == 1
