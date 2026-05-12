# Re-run gemini-3.1-pro-preview AFTER the daily quota reset (~midnight Pacific).
#
# Why this script exists separately from run_gemini_2_5_pro.ps1:
#   - gemini-3.1-pro-preview is a PREVIEW model; Google caps it at 250 req/day
#     regardless of billing tier (no paid relief on preview-only models).
#   - A single CR+PS gauntlet burns ~300-400 requests with retries enabled.
#   - We swapped to gemini-2.5-pro (GA, no daily cap) for the published matrix.
#     This script is for collecting the 3.1-pro-preview reference numbers as
#     a "current frontier preview" data point once quota refreshes.
#
# Both fixes from commit ac2d2f2 apply to this run:
#   - games.py stepdown trigger now fires on MALFORMED_FUNCTION_CALL too
#   - gemini.py adapter wires reasoning_effort_override -> thinking_budget
#
# Run sometime after 03:00 EDT (= 00:00 Pacific) for a fresh 250-req budget.

$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

Write-Output "============================================================"
Write-Output "Gemini 3.1 Pro Preview - post-quota-reset re-run"
Write-Output "  Fixes applied: MALFORMED_FUNCTION_CALL stepdown trigger,"
Write-Output "                 thinking_budget wired to ladder."
Write-Output "  Daily quota: 250 req/day (preview-model cap)"
Write-Output "============================================================"

# CR with new retry+penalty formula
& $venvPy -m llm_chess_eval.cli reliability --model gemini-3.1-pro-preview --games 5 --skill 3 --max-plies 40 --max-retries 10
Write-Output ""
# PS
& $venvPy -m llm_chess_eval.cli play-strength --model gemini-3.1-pro-preview --games 3 --skill 5 --max-plies 60 --max-retries 3
