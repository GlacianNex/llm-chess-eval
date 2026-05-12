$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "BENCHMARK MATRIX: Google (Gemini)"
Write-Output "============================================================"

& $venvPy -m llm_chess_eval.cli benchmark --provider google
