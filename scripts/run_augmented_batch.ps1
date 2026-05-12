$env:ANTHROPIC_API_KEY = (Get-ItemProperty -Path "HKCU:\Environment" -Name "ANTHROPIC_API_KEY").ANTHROPIC_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "=== AUGMENTED LEGALITY: OPUS ==="
& $venvPy -m llm_chess_eval.cli legality --model claude-opus-4-7 --augment-legal-moves
Write-Output ""

Write-Output "=== AUGMENTED LEGALITY: SONNET ==="
& $venvPy -m llm_chess_eval.cli legality --model claude-sonnet-4-6 --augment-legal-moves
Write-Output ""

Write-Output "=== AUGMENTED CONSISTENCY: OPUS ==="
& $venvPy -m llm_chess_eval.cli consistency --model claude-opus-4-7 --augment-legal-moves
Write-Output ""

Write-Output "=== AUGMENTED CONSISTENCY: SONNET ==="
& $venvPy -m llm_chess_eval.cli consistency --model claude-sonnet-4-6 --augment-legal-moves
Write-Output ""

Write-Output "=== AUGMENTED GAMES (FORFEIT): OPUS ==="
& $venvPy -m llm_chess_eval.cli games --model claude-opus-4-7 --games 2 --skill 3 --max-plies 40 --color alternating --mode forfeit --augment-legal-moves
Write-Output ""

Write-Output "=== AUGMENTED GAMES (FORFEIT): SONNET ==="
& $venvPy -m llm_chess_eval.cli games --model claude-sonnet-4-6 --games 2 --skill 3 --max-plies 40 --color alternating --mode forfeit --augment-legal-moves
