$env:ANTHROPIC_API_KEY = (Get-ItemProperty -Path "HKCU:\Environment" -Name "ANTHROPIC_API_KEY").ANTHROPIC_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "=== PLAY-STRENGTH: OPUS (skill 5, retry, 2 games) ==="
& $venvPy -m llm_chess_eval.cli play-strength --model claude-opus-4-7 --games 2 --skill 5 --max-plies 60
Write-Output ""

Write-Output "=== PLAY-STRENGTH: SONNET (skill 5, retry, 2 games) ==="
& $venvPy -m llm_chess_eval.cli play-strength --model claude-sonnet-4-6 --games 2 --skill 5 --max-plies 60
