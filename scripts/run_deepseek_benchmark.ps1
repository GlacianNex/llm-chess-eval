$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:DEEPSEEK_API_KEY = $reg.DEEPSEEK_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "BENCHMARK MATRIX: DeepSeek"
Write-Output "============================================================"

& $venvPy -m llm_chess_eval.cli benchmark --provider deepseek
