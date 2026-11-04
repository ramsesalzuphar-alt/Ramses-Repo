# One click: start ComfyUI and the Studio page if they aren't running, then open the page.
$comfy = "$HOME\ComfyUI"
$python = "$comfy\venv\Scripts\python.exe"
$studio = 'http://127.0.0.1:8190'

function Test-Up($url) {
  try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; $true } catch { $false }
}

if (-not (Test-Up 'http://127.0.0.1:8188/system_stats')) {
  Start-Process -FilePath $python -ArgumentList "`"$comfy\main.py`" --port 8188" -WorkingDirectory $comfy -WindowStyle Minimized
}
if (-not (Test-Up $studio)) {
  Start-Process -FilePath $python -ArgumentList "`"$PSScriptRoot\studio_server.py`"" -WorkingDirectory $PSScriptRoot -WindowStyle Minimized
}
for ($i = 0; $i -lt 30 -and -not (Test-Up $studio); $i++) { Start-Sleep -Seconds 1 }
Start-Process $studio
