$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "RE-RUN: Gemini 3.1 Pro Preview (token bug fix)"
Write-Output "  max_tokens: 2048 -> 8000"
Write-Output "============================================================"

# CR with new retry+penalty formula
& $venvPy -m llm_chess_eval.cli reliability --model gemini-3.1-pro-preview --games 5 --skill 3 --max-plies 40 --max-retries 10
Write-Output ""
# PS
& $venvPy -m llm_chess_eval.cli play-strength --model gemini-3.1-pro-preview --games 3 --skill 5 --max-plies 60 --max-retries 3
