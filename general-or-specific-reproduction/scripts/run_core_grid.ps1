param(
    [int[]]$Folds = @(1, 2, 3),
    [double[]]$Epsilons = @(1, 5, 10, 25, 50),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

foreach ($fold in $Folds) {
    foreach ($role in @("shadow", "private")) {
        $baseline = "outputs\runs\fold$fold\$role\none\ser_validation.json"
        if ($Force -or -not (Test-Path $baseline)) {
            & $Python scripts\federated_train.py --fold $fold --role $role --defense none
            if ($LASTEXITCODE -ne 0) { throw "Baseline failed: fold=$fold role=$role" }
        }

        foreach ($epsilon in $Epsilons) {
            $epsilonTag = $epsilon.ToString([Globalization.CultureInfo]::InvariantCulture)
            $result = "outputs\runs\fold$fold\$role\udp\epsilon_$epsilonTag\ser_validation.json"
            if ((Test-Path $result) -and -not $Force) { continue }
            & $Python scripts\federated_train.py --fold $fold --role $role --defense udp --epsilon $epsilon
            if ($LASTEXITCODE -ne 0) {
                throw "UDP failed: fold=$fold role=$role epsilon=$epsilonTag"
            }
        }
    }
}

Write-Host "Core baseline and UDP grid complete."
