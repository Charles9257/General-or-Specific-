param(
    [string[]]$Defenses = @("voice-ind", "gender-ind"),
    [double[]]$Epsilons = @(1, 5, 10, 25, 50),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$cache = "outputs\features\synth-emobase-cache.npz"

foreach ($defense in $Defenses) {
    foreach ($epsilon in $Epsilons) {
        $epsilonTag = $epsilon.ToString([Globalization.CultureInfo]::InvariantCulture)
        $audio = "outputs\audio\$defense-e$epsilonTag"
        $featureFile = "outputs\features\$defense-e$epsilonTag.npz"
        if (-not (Test-Path $audio)) {
            Write-Warning "Skipping missing synthesized audio directory: $audio"
            continue
        }
        if ((Test-Path $featureFile) -and -not $Force) { continue }
        & $Python scripts\extract_emobase.py outputs\manifest.csv --audio-root $audio `
            --output $featureFile --feature-cache $cache
        if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed: defense=$defense epsilon=$epsilonTag" }
    }
}

Write-Host "Transformed-audio EmoBase feature grid complete."
