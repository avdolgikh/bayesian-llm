# Build LaTeX paper to PDF using Docker (TeX Live).
# Self-contained — this skill directory ships the .sty and everything needed.
#
# Usage:
#   powershell <skill-dir>\scripts\build.ps1 -TexFile <input.tex> [-Output <output.pdf>]
#
# The project root is auto-detected (git root, or parent of tex dir).
# The bundled neurips_2024.sty is injected via TEXINPUTS.
#
# Requirements: Docker (texlive/texlive:latest, auto-pulled on first run).

param(
    [Parameter(Mandatory=$true)]
    [string]$TexFile,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
# Skill root is one level up from scripts/
$SkillDir = (Resolve-Path "$PSScriptRoot/..").Path

# Resolve to absolute
$TexFile = (Resolve-Path $TexFile).Path
$TexDir = Split-Path $TexFile
$TexName = [System.IO.Path]::GetFileNameWithoutExtension($TexFile)

if (-not (Test-Path $TexFile)) {
    Write-Error "$TexFile not found"
    exit 1
}

# Auto-detect project root: git root if available, else parent of tex dir
try {
    $ProjectRoot = (git -C $TexDir rev-parse --show-toplevel 2>$null)
    if (-not $ProjectRoot) { throw }
    $ProjectRoot = $ProjectRoot.Trim()
} catch {
    $ProjectRoot = Split-Path $TexDir
}
$RelTexDir = $TexDir.Substring($ProjectRoot.Length + 1) -replace '\\', '/'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required. Install Docker Desktop."
    exit 1
}

$Image = "texlive/texlive:latest"

Write-Host "Compiling $(Split-Path $TexFile -Leaf)..."
Write-Host "  project root: $ProjectRoot"
Write-Host "  tex dir:      $RelTexDir/"

$ProjectRootDocker = $ProjectRoot -replace '\\', '/'
$SkillDirDocker = $SkillDir -replace '\\', '/'

# Mount project root + skill dir; TEXINPUTS injects bundled .sty
docker run --rm `
    -v "${ProjectRootDocker}:/repo" `
    -v "${SkillDirDocker}:/skill:ro" `
    -e "TEXINPUTS=/skill//:" `
    -w "/repo/$RelTexDir" `
    $Image `
    bash -c "pdflatex -interaction=nonstopmode $TexName && bibtex $TexName 2>&1; pdflatex -interaction=nonstopmode $TexName && pdflatex -interaction=nonstopmode $TexName"

$PdfPath = Join-Path $TexDir "$TexName.pdf"
if (-not (Test-Path $PdfPath)) {
    Write-Error "PDF was not generated."
    exit 1
}

if ($Output -ne "") {
    $Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
    Copy-Item $PdfPath $Output
    $PdfPath = $Output
}

# Clean intermediate files
foreach ($ext in @("aux", "bbl", "blg", "log", "out")) {
    Remove-Item (Join-Path $TexDir "$TexName.$ext") -ErrorAction SilentlyContinue
}

$SizeMB = (Get-Item $PdfPath).Length / 1MB
Write-Host ("Done: $PdfPath ({0:N1} MB)" -f $SizeMB)
