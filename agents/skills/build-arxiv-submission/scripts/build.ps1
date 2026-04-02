# Build arXiv submission zip from LaTeX source.
# PowerShell version of build.sh — see that file for full docs.
#
# Usage:
#   .\build.ps1 -TexFile <input.tex> [-OutputZip <output.zip>]

param(
    [Parameter(Mandatory=$true)]
    [string]$TexFile,

    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"

# Resolve paths
$TexFile = (Resolve-Path $TexFile).Path
$TexDir = Split-Path $TexFile -Parent
$TexName = [System.IO.Path]::GetFileNameWithoutExtension($TexFile)

if (-not (Test-Path $TexFile)) {
    Write-Error "ERROR: $TexFile not found"
    exit 1
}

# Auto-detect project root
$GitRoot = git -C $TexDir rev-parse --show-toplevel 2>$null
if ($GitRoot -and (Test-Path $GitRoot)) {
    $ProjectRoot = (Resolve-Path $GitRoot).Path
} else {
    $ProjectRoot = (Resolve-Path "$TexDir/..").Path
}

# Default output
if (-not $OutputZip) {
    $OutputZip = Join-Path $TexDir "arxiv-submission.zip"
}

# ── Parse tex file ────────────────────────────────────────────────────────────

Write-Host "Parsing $TexFile..."

$TexContent = Get-Content $TexFile -Raw

# Extract \graphicspath
$FiguresDir = $null
if ($TexContent -match '\\graphicspath\{\{([^}]+)\}\}') {
    $GraphicsPath = $Matches[1]
    $CandidateDir = Join-Path $TexDir $GraphicsPath
    if (Test-Path $CandidateDir) {
        $FiguresDir = (Resolve-Path $CandidateDir).Path
    }
}
if (-not $FiguresDir) {
    $DefaultFigures = Join-Path $ProjectRoot "figures"
    if (Test-Path $DefaultFigures) {
        $FiguresDir = $DefaultFigures
        Write-Host "  Using default figures dir: $FiguresDir"
    } else {
        Write-Host "  WARNING: No figures directory found."
    }
}

# Extract \bibliography{name}
$BibFile = $null
if ($TexContent -match '\\bibliography\{([^}]+)\}') {
    $BibName = $Matches[1]
    $BibCandidate = Join-Path $TexDir "$BibName.bib"
    if (Test-Path $BibCandidate) {
        $BibFile = $BibCandidate
    } else {
        $BibCandidate = Join-Path $ProjectRoot "$BibName.bib"
        if (Test-Path $BibCandidate) { $BibFile = $BibCandidate }
    }
}

# Find .sty files referenced by \usepackage
$StyFiles = @()
$Packages = [regex]::Matches($TexContent, '\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}')
foreach ($match in $Packages) {
    $PkgName = $match.Groups[1].Value
    $StyCandidate = Get-ChildItem -Path $ProjectRoot -Filter "$PkgName.sty" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\.git' } |
        Select-Object -First 1
    if ($StyCandidate) {
        $StyFiles += $StyCandidate.FullName
    }
}

# ── Stage files ───────────────────────────────────────────────────────────────

$Staging = Join-Path ([System.IO.Path]::GetTempPath()) "arxiv-submission-$(Get-Random)"
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

Write-Host "Staging submission..."

# Copy tex
Copy-Item $TexFile -Destination $Staging

# Copy bib
if ($BibFile -and (Test-Path $BibFile)) {
    Copy-Item $BibFile -Destination $Staging
    Write-Host "  bib: $(Split-Path $BibFile -Leaf)"
}

# Copy sty files
foreach ($sty in $StyFiles) {
    Copy-Item $sty -Destination $Staging
    Write-Host "  sty: $(Split-Path $sty -Leaf)"
}

# Copy figures
$FigCount = 0
if ($FiguresDir -and (Test-Path $FiguresDir)) {
    $FigStaging = Join-Path $Staging "figures"
    New-Item -ItemType Directory -Path $FigStaging -Force | Out-Null
    Get-ChildItem $FiguresDir -File | ForEach-Object {
        Copy-Item $_.FullName -Destination $FigStaging
        $FigCount++
    }
    Write-Host "  figures: $FigCount files"
}

# Rewrite \graphicspath
$StagedTex = Join-Path $Staging "$TexName.tex"
(Get-Content $StagedTex -Raw) -replace '\\graphicspath\{\{[^}]*\}\}', '\graphicspath{{figures/}}' |
    Set-Content $StagedTex -NoNewline
Write-Host "  graphicspath: rewritten to {figures/}"

# ── Build zip ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Building zip..."

if (Test-Path $OutputZip) { Remove-Item $OutputZip -Force }

Compress-Archive -Path "$Staging/*" -DestinationPath $OutputZip -Force

# Cleanup
Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $OutputZip)) {
    Write-Error "ERROR: Failed to create zip"
    exit 1
}

$Size = (Get-Item $OutputZip).Length
$SizeMB = [math]::Round($Size / 1MB, 1)
Write-Host ""
Write-Host "Done: $OutputZip"
Write-Host "  $($FigCount + 2 + $StyFiles.Count) files, ${SizeMB} MB (arXiv limit: 50 MB)"
