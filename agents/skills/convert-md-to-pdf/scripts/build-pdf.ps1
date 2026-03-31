# Build PDF from markdown with Mermaid diagrams + MathJax formulas
# Usage: powershell agents/skills/convert-md-to-pdf/scripts/build-pdf.ps1 docs/paper.md paper.pdf
#
# Pipeline: mmdc (Mermaid->SVG) -> md-to-pdf (MathJax + Puppeteer -> PDF)

param(
    [Parameter(Mandatory)][string]$InputFile,
    [Parameter(Mandatory)][string]$OutputFile
)

$ErrorActionPreference = "Stop"
$ConfigFile = Join-Path $PSScriptRoot "md-to-pdf.config.js"
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
$MermaidOut = "$BaseName-mermaid.md"

Write-Host "==> Step 1: Mermaid diagrams -> SVG"
& mmdc -i $InputFile -o $MermaidOut

Write-Host "==> Step 2: Markdown + MathJax -> PDF"
& md-to-pdf --config-file $ConfigFile $MermaidOut

$MermaidPdf = "$BaseName-mermaid.pdf"
if (Test-Path $MermaidPdf) {
    Move-Item $MermaidPdf $OutputFile -Force
    Write-Host "==> Done: $OutputFile"
} else {
    Write-Host "==> ERROR: PDF not generated"
    exit 1
}

Remove-Item $MermaidOut -ErrorAction SilentlyContinue
