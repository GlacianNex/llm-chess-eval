$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:DEEPSEEK_API_KEY = $reg.DEEPSEEK_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"
Write-Output "=== CLEAN RE-RUN: DeepSeek (max_tokens=16000, final CR formula) ==="
& $venvPy -m llm_chess_eval.cli benchmark --provider deepseek
