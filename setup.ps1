$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Get-Command py -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command python -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Python 3.11+ was not found. Install Python and run setup.ps1 again." }

if ($Python.Name -eq "py.exe") {
    & $Python.Source -3 -m venv (Join-Path $Root ".venv")
} else {
    & $Python.Source -m venv (Join-Path $Root ".venv")
}
if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment." }
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip. Check network access and try again." }
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies. Check network access and try again." }

$Config = Join-Path $Root "config.json"
if (-not (Test-Path -LiteralPath $Config)) {
    Copy-Item -LiteralPath (Join-Path $Root "config.example.json") -Destination $Config
}
Write-Host "Setup complete. Edit config.json if needed, then run start.ps1."
