$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "Gemini 2.5 Pro (frontier substitute - GA, no preview daily cap)"
Write-Output "============================================================"

& $venvPy -m llm_chess_eval.cli reliability --model gemini-2.5-pro --games 5 --skill 3 --max-plies 40 --max-retries 10
Write-Output ""
& $venvPy -m llm_chess_eval.cli play-strength --model gemini-2.5-pro --games 3 --skill 5 --max-plies 60 --max-retries 3
