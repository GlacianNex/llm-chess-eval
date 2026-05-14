$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

# Reliability at multiple Stockfish skill levels for one model (Flash Lite).
# Skill sweep gives a defensibility chart: shows the metric isn't a
# single-skill artifact and demonstrates the natural degradation with
# opponent strength.
Write-Output "============================================================"
Write-Output "Flash Lite skill sweep: skill 1, 5, 10, 15"
Write-Output "============================================================"

foreach ($skill in @(1, 5, 10, 15)) {
    Write-Output ""
    Write-Output "--- skill $skill ---"
    & $venvPy -m llm_chess_eval.cli reliability --model gemini-3.1-flash-lite --games 3 --skill $skill --max-plies 40 --max-retries 10
}
