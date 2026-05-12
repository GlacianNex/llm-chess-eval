$reg = Get-ItemProperty -Path "HKCU:\Environment"
$env:ANTHROPIC_API_KEY = $reg.ANTHROPIC_API_KEY
$env:OPENAI_API_KEY = $reg.OPENAI_API_KEY
$env:GOOGLE_API_KEY = $reg.GOOGLE_API_KEY
$env:DEEPSEEK_API_KEY = $reg.DEEPSEEK_API_KEY
$venvPy = "C:\Users\igorc\Projects\LLM_Chess_Eval\.venv\Scripts\python.exe"

$models = @(
    @{provider="anthropic"; tier="frontier"; id="claude-opus-4-7"},
    @{provider="anthropic"; tier="budget";   id="claude-haiku-4-5-20251001"},
    @{provider="openai";    tier="frontier"; id="gpt-5"},
    @{provider="openai";    tier="budget";   id="gpt-5-mini"},
    @{provider="google";    tier="frontier"; id="gemini-3.1-pro-preview"},
    @{provider="google";    tier="budget";   id="gemini-3.1-flash-lite"},
    @{provider="deepseek";  tier="frontier"; id="deepseek-reasoner"},
    @{provider="deepseek";  tier="budget";   id="deepseek-chat"}
)

foreach ($m in $models) {
    Write-Output ""
    Write-Output "============================================================"
    Write-Output ("CR re-run (retry+penalty): " + $m.provider + " / " + $m.tier + " - " + $m.id)
    Write-Output "============================================================"
    & $venvPy -m llm_chess_eval.cli reliability --model $m.id --games 5 --skill 3 --max-plies 40 --max-retries 10
}
