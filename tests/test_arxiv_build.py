"""arXiv source package checks (spec: agents/specs/2026-10/arxiv-submission-build.md, R7, R9).

arXiv compiles the LaTeX source itself, runs no BibTeX and runs on Linux. So the zip must hold the
real style file, the .bbl, and entry names with forward slashes. Its metadata form refuses non-ASCII
characters, so the build also writes the form text as plain ASCII (R9).
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
COMMITTED_METADATA = PAPER_DIR / "arxiv-metadata.txt"
SCRIPTS = ROOT / "agents" / "skills" / "build-arxiv-submission" / "scripts"
BUILD_SH = SCRIPTS / "build.sh"
METADATA_PY = SCRIPTS / "metadata.py"
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

METADATA_LABELS = ["Title", "Authors", "Abstract", "Comments"]
ABSTRACT_LIMIT = 1920  # arXiv's limit for the abstract, in characters

# R9, AT9b: the v1 title, author and abstract source (lines re-wrapped; LaTeX joins them). A copy,
# so editing paper/paper.tex for version 2 never breaks this test.
FIXTURE_TEX = r"""\title{Epistemic Uncertainty in Language Models}
\author{
  Oleksiy Dolgykh \\
  \texttt{alexey.dolgich@gmail.com}
}
\begin{document}
\maketitle
% \includegraphics{commented-out.png} is not a figure
\begin{abstract}
We present a comparison of four Bayesian approaches to epistemic uncertainty estimation in
language models---variational and post-hoc inference across full weights and LoRA
adapters---plus a Monte Carlo (MC) Dropout baseline. All methods are evaluated on the same
76M-parameter GPT-2-style architecture trained on The Pile with domain-split OOD evaluation.
We report three main findings.
First, \textbf{LoRA-based Bayesian methods achieved the strongest OOD detection in our setup}:
BLoB (Bayesian Low-Rank Adaptation by Backpropagation) LoRA (AUROC 0.909) and TFB
(Training-Free Bayesianization) (0.917) outperform full-weight variational inference (0.874)
with $17{\times}$ fewer Bayesian parameters, though this comparison is confounded by
differences in training procedure and backbone quality (\S\ref{sec:discussion}).
Second, \textbf{diagonal Laplace approximation fails} for language model OOD
detection---the diagonal Fisher curvature is flat at convergence regardless of
parameterization (full weights or LoRA).
Third, \textbf{MC Dropout} applied to the deterministic checkpoint (zero extra training)
achieves AUROC 0.898---competitive with trained Bayesian methods (overlapping confidence
intervals with BLoB).
\textbf{TFB}, also requiring no Bayesian training, matches the best trained methods at AUROC 0.917.
Production benchmarks show $N{=}3$ MC samples capture 97\% of the full
signal.\footnote{Code: \url{https://github.com/avdolgikh/bayesian-llm}}
\end{abstract}
\includegraphics[width=0.85\textwidth]{fig1.png}
\includegraphics{fig2.png}
\includegraphics{fig3.png}
\end{document}
"""
# Written by hand from the R9 mapping (not copied from the v1 paste, which had three hand edits).
FIXTURE_ABSTRACT = (
    "We present a comparison of four Bayesian approaches to epistemic uncertainty estimation in "
    "language models, variational and post-hoc inference across full weights and LoRA adapters, "
    "plus a Monte Carlo (MC) Dropout baseline. All methods are evaluated on the same "
    "76M-parameter GPT-2-style architecture trained on The Pile with domain-split OOD "
    "evaluation. We report three main findings. First, LoRA-based Bayesian methods achieved the "
    "strongest OOD detection in our setup: BLoB (Bayesian Low-Rank Adaptation by "
    "Backpropagation) LoRA (AUROC 0.909) and TFB (Training-Free Bayesianization) (0.917) "
    "outperform full-weight variational inference (0.874) with 17x fewer Bayesian parameters, "
    "though this comparison is confounded by differences in training procedure and backbone "
    "quality. Second, diagonal Laplace approximation fails for language model OOD detection, the "
    "diagonal Fisher curvature is flat at convergence regardless of parameterization (full "
    "weights or LoRA). Third, MC Dropout applied to the deterministic checkpoint (zero extra "
    "training) achieves AUROC 0.898, competitive with trained Bayesian methods (overlapping "
    "confidence intervals with BLoB). TFB, also requiring no Bayesian training, matches the best "
    "trained methods at AUROC 0.917. Production benchmarks show N=3 MC samples capture 97% of the "
    "full signal."
)
FIXTURE_EXPECTED = (
    "Title:\nEpistemic Uncertainty in Language Models\n\n"
    "Authors:\nOleksiy Dolgykh\n\n"
    f"Abstract:\n{FIXTURE_ABSTRACT}\n\n"
    "Comments:\n8 pages, 3 figures. Code: https://github.com/avdolgikh/bayesian-llm\n"
)

# R9: rules the v1 abstract does not need but the paper body uses (a digit range, control and thin
# spaces, TeX quotes), an escaped sign and two paragraphs.
RULES_TEX = r"""\title{T}
\author{A}
\begin{abstract}
AUROC 0.909--0.917 vs.\ 0.874 on 12\,GB; a ``quoted'' word~here, Q\&A.

A second paragraph.
\end{abstract}
"""
RULES_ABSTRACT = (
    'AUROC 0.909-0.917 vs. 0.874 on 12 GB; a "quoted" word here, Q&A.\n\nA second paragraph.'
)

# AT9a: a typed em-dash and an unknown command must stop the script.
BAD_TEX = "\\title{T}\n\\author{A}\n\\begin{abstract}\nA \u2014 and \\citep{x}.\n\\end{abstract}\n"


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


def _run_metadata(*args: str) -> subprocess.CompletedProcess:
    """Run metadata.py with the test's Python (it needs only the standard library)."""
    return subprocess.run(
        [sys.executable, METADATA_PY.as_posix(), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _metadata_from(tmp_path: Path, name: str, tex_text: str, *extra: str) -> str:
    """Write tex_text to <name>.tex, run metadata.py on it, return <name>.txt (must be ASCII)."""
    tex = tmp_path / f"{name}.tex"
    tex.write_text(tex_text, encoding="utf-8")
    out = tmp_path / f"{name}.txt"
    result = _run_metadata(tex.as_posix(), out.as_posix(), *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    return out.read_bytes().decode("ascii")


def _metadata_blocks(text: str) -> dict[str, str]:
    """The labelled blocks of a metadata file: a label line ("Abstract:"), then its text."""
    blocks: dict[str, list[str]] = {}
    label = None
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.endswith(":") and line[:-1] in METADATA_LABELS:
            label = line[:-1]
            blocks[label] = []
        elif label is not None:
            blocks[label].append(line)
    return {name: "\n".join(lines).strip() for name, lines in blocks.items()}


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

    metadata = tmp_path / "arxiv-metadata.txt"  # R9: next to the zip, never next to the tex
    assert metadata.exists(), result.stdout + result.stderr
    assert "? pages" in metadata.read_text(encoding="ascii")  # no compile, so no page count


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


def test_metadata_script_writes_ascii_form_text(tmp_path):
    """R9 test 1 (AT9a-AT9c): metadata.py maps the LaTeX to ASCII, then runs on the real paper."""
    if not METADATA_PY.exists():
        pytest.skip("metadata script (agents/) not available")

    assert _metadata_from(tmp_path, "v1", FIXTURE_TEX, "--pages", "8") == FIXTURE_EXPECTED  # AT9b
    check = _run_metadata("--check", (tmp_path / "v1.txt").as_posix())
    assert check.returncode == 0, check.stdout + check.stderr

    rules = _metadata_blocks(_metadata_from(tmp_path, "rules", RULES_TEX))
    assert rules["Abstract"] == RULES_ABSTRACT

    bad_tex, bad_out = tmp_path / "bad.tex", tmp_path / "bad.txt"  # AT9a
    bad_tex.write_text(BAD_TEX, encoding="utf-8")
    result = _run_metadata(bad_tex.as_posix(), bad_out.as_posix())
    assert result.returncode == 1, result.stdout + result.stderr
    assert "U+2014" in result.stderr and "\\citep" in result.stderr
    assert not bad_out.exists()

    out = tmp_path / "paper.txt"
    result = _run_metadata(TEX.as_posix(), out.as_posix())
    assert result.returncode == 0, result.stdout + result.stderr
    data = out.read_bytes()
    assert data.isascii()
    blocks = _metadata_blocks(data.decode("ascii"))
    assert list(blocks) == METADATA_LABELS and all(blocks.values()), blocks
    assert "Oleksiy Dolgykh" in blocks["Authors"].split("\n")
    abstract = blocks["Abstract"]
    assert len(abstract) <= ABSTRACT_LIMIT
    for bad in ("---", "\u00a7", "\\"):
        assert bad not in abstract, f"{bad!r} in the abstract"


def test_committed_metadata_is_ascii(tmp_path):
    """R9 test 2: the committed form text is ASCII in four blocks (CI too) and, locally, current."""
    if not COMMITTED_METADATA.exists():
        pytest.skip("paper/arxiv-metadata.txt absent")
    data = COMMITTED_METADATA.read_bytes()
    assert data.isascii(), "non-ASCII byte in paper/arxiv-metadata.txt"
    text = data.decode("ascii").replace("\r\n", "\n")
    blocks = _metadata_blocks(text)
    assert list(blocks) == METADATA_LABELS and all(blocks.values()), blocks
    assert len(blocks["Abstract"]) <= ABSTRACT_LIMIT
    if not METADATA_PY.exists():
        return  # CI: the script lives in agents/, which CI cannot see
    pages = blocks["Comments"].split(" pages")[0]
    fresh = tmp_path / "fresh.txt"
    result = _run_metadata(TEX.as_posix(), fresh.as_posix(), "--pages", pages)
    assert result.returncode == 0, result.stdout + result.stderr
    assert fresh.read_bytes().decode("ascii") == text, (
        "paper/arxiv-metadata.txt is stale: rerun build.sh (spec R9)"
    )
