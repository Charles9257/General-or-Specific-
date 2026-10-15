param(
    [int[]]$Folds = @(1, 2, 3),
    [string[]]$Defenses = @("voice-ind", "gender-ind"),
    [double[]]$Epsilons = @(1, 5, 10, 25, 50),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

foreach ($fold in $Folds) {
    foreach ($defense in $Defenses) {
        foreach ($epsilon in $Epsilons) {
            $epsilonTag = $epsilon.ToString([Globalization.CultureInfo]::InvariantCulture)
            $features = "outputs\features\$defense-e$epsilonTag.npz"
            if (-not (Test-Path $features)) {
                Write-Warning "Skipping absent transformed features: $features"
                continue
            }
            foreach ($role in @("shadow", "private")) {
                $result = "outputs\runs\fold$fold\$role\$defense\epsilon_$epsilonTag\ser_validation.json"
                if ((Test-Path $result) -and -not $Force) { continue }
                & $Python scripts\federated_train.py --fold $fold --role $role --defense $defense `
                    --feature-privacy $defense --epsilon $epsilon --features $features
                if ($LASTEXITCODE -ne 0) { throw "Audio defense failed: fold=$fold role=$role defense=$defense epsilon=$epsilonTag" }
            }
        }
    }
}

Write-Host "Voice-Ind/Gender-Ind FL grid complete."
