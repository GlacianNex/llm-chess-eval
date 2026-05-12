$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:OPENAI_API_KEY = $reg.OPENAI_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"
Write-Output "=== CLEAN RE-RUN: OpenAI (max_tokens=16000, NO reasoning_effort handicap) ==="
& $venvPy -m llm_chess_eval.cli benchmark --provider openai
