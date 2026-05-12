$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:ANTHROPIC_API_KEY = $reg.ANTHROPIC_API_KEY
$env:OPENAI_API_KEY = $reg.OPENAI_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "BENCHMARK MATRIX: Anthropic + OpenAI"
Write-Output "============================================================"

& $venvPy -m llm_chess_eval.cli benchmark --provider anthropic
Write-Output ""
& $venvPy -m llm_chess_eval.cli benchmark --provider openai
