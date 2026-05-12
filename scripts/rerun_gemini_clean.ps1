$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"
Write-Output "=== CLEAN RE-RUN: Gemini 3.1 Pro Preview (max_tokens=16000) ==="
& $venvPy -m llm_chess_eval.cli reliability --model gemini-3.1-pro-preview --games 5 --skill 3 --max-plies 40 --max-retries 10
& $venvPy -m llm_chess_eval.cli play-strength --model gemini-3.1-pro-preview --games 3 --skill 5 --max-plies 60 --max-retries 3
