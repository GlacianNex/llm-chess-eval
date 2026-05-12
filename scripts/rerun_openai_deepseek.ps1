$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:OPENAI_API_KEY = $reg.OPENAI_API_KEY
$env:DEEPSEEK_API_KEY = $reg.DEEPSEEK_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "RE-RUN: OpenAI + DeepSeek with fixed adapters"
Write-Output "  max_tokens=8000 (was 2048)"
Write-Output "  reasoning_effort='low' on GPT-5/GPT-5-mini"
Write-Output "============================================================"

& $venvPy -m llm_chess_eval.cli benchmark --provider openai
Write-Output ""
& $venvPy -m llm_chess_eval.cli benchmark --provider deepseek
