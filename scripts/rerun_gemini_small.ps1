$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "RE-RUN (SMALLER): Gemini 3.1 Pro Preview"
Write-Output "  max_tokens=16000 (no handicap), N=2 CR + N=1 PS"
Write-Output "  Reduced N because per-call latency is ~2-3 min at full reasoning"
Write-Output "============================================================"

# CR: 2 games (was 5) — smaller sample, full methodology
& $venvPy -m llm_chess_eval.cli reliability --model gemini-3.1-pro-preview --games 2 --skill 3 --max-plies 40 --max-retries 10
Write-Output ""
# PS: 1 game (was 3)
& $venvPy -m llm_chess_eval.cli play-strength --model gemini-3.1-pro-preview --games 1 --skill 5 --max-plies 60 --max-retries 3
