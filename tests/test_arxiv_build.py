"""Checks for the arXiv source package (spec: agents/specs/2026-10/arxiv-submission-build.md, R7).

arXiv compiles the LaTeX source itself, runs no BibTeX and runs on Linux. So the zip must hold the
real style file, the .bbl, and entry names with forward slashes.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
TEX = PAPER_DIR / "paper.tex"
COMMITTED_ZIP = PAPER_DIR / "arxiv-submission.zip"
COMMITTED_STY = PAPER_DIR / "neurips_2024.sty"
BUILD_SH = ROOT / "agents" / "skills" / "build-arxiv-submission" / "scripts" / "build.sh"
REFERENCE_STY = ROOT / "agents" / "skills" / "build-latex-pdf" / "references" / "neurips_2024.sty"

EXPECTED_ENTRIES = {
    "paper.tex",
    "paper.bbl",
    "references.bib",
    "neurips_2024.sty",
    "figures/fig1_point_vs_bayesian.png",
    "figures/fig2_method_matrix.png",
    "figures/fig3_blob_lora.png",
}
PROVIDES = rb"\ProvidesPackage{neurips_2024}"


def _find_bash() -> str | None:
    """Git Bash on Windows (never the WSL launcher), plain bash elsewhere."""
    if sys.platform == "win32":
        git_dir = r"C:\Program Files\Git"
        for candidate in (git_dir + r"\bin\bash.exe", git_dir + r"\usr\bin\bash.exe"):
            if os.path.exists(candidate):
                return candidate
        found = shutil.which("bash")
        if found and "system32" not in found.lower():
            return found
        return None
    return shutil.which("bash")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_zip_invariants(zip_path: Path) -> zipfile.ZipFile:
    """AT2, AT3a, AT3b, AT4a, AT4b and a real style file. Returns the open zip."""
    zf = zipfile.ZipFile(zip_path)
    # On Windows, namelist() turns "\\" into "/"; orig_filename keeps the raw name arXiv sees.
    raw_names = [info.orig_filename for info in zf.infolist()]
    assert not [n for n in raw_names if "\\" in n], f"backslash in entry names: {raw_names}"  # AT3a
    names = zf.namelist()
    assert set(names) == EXPECTED_ENTRIES, f"entries: {sorted(names)}"  # AT3b

    bbl = zf.read("paper.bbl")  # AT2
    assert len(bbl) > 0
    assert rb"\begin{thebibliography}" in bbl

    tex_lines = zf.read("paper.tex").decode("utf-8").splitlines()
    assert tex_lines[0].strip() == r"\pdfoutput=1"  # AT4a
    tex_text = "\n".join(tex_lines)
    assert r"\graphicspath{{figures/}}" in tex_text  # AT4b
    assert "../figures/" not in tex_text

    sty = zf.read("neurips_2024.sty")
    assert len(sty) > 0
    assert PROVIDES in sty

    # AT3c: arXiv runs on Linux; text entries use LF even when git checked them out as CRLF.
    for name in names:
        if not name.endswith(".png"):
            assert b"\r" not in zf.read(name), f"CR byte in {name}"
    return zf


def test_build_script_makes_valid_zip(tmp_path):
    """R7 test 1: the build script on the real paper, without Docker."""
    bash = _find_bash()
    if not BUILD_SH.exists() or bash is None:
        pytest.skip("build script (agents/) or bash not available")
    out_zip = tmp_path / "out.zip"
    tex_hash_before = _sha256(TEX)
    result = subprocess.run(
        [bash, BUILD_SH.as_posix(), TEX.as_posix(), out_zip.as_posix(), "--no-verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verify skipped" in result.stdout
    assert _sha256(TEX) == tex_hash_before  # AT4c

    zf = _check_zip_invariants(out_zip)
    if REFERENCE_STY.exists():  # AT1 (the reference may be CRLF on a Windows checkout)
        assert zf.read("neurips_2024.sty") == REFERENCE_STY.read_bytes().replace(b"\r\n", b"\n")
    zf.close()


def test_committed_zip_is_valid():
    """R7 test 2: the committed arXiv source package (runs in CI)."""
    if not COMMITTED_ZIP.exists():
        pytest.skip("paper/arxiv-submission.zip absent")
    _check_zip_invariants(COMMITTED_ZIP).close()


def test_committed_style_file_is_real():
    """R7 test 3: paper/neurips_2024.sty is the real style file, not an empty placeholder."""
    data = COMMITTED_STY.read_bytes()
    assert len(data) > 0
    assert PROVIDES in data
