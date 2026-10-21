param(
    [int[]]$Folds = @(1, 2, 3),
    [int[]]$SampleCounts = @(5, 10, 100),
    [string[]]$Defenses = @("none", "udp", "voice-ind", "gender-ind"),
    [double[]]$Epsilons = @(1, 5, 10, 25, 50),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Fast,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

foreach ($fold in $Folds) {
    foreach ($defense in $Defenses) {
        # Keep a non-null sentinel for the no-protection run: PowerShell's
        # foreach silently skips a one-element array containing $null.
        $epsilonValues = if ($defense -eq "none") { @("none") } else { $Epsilons }
        foreach ($epsilon in $epsilonValues) {
            $noProtection = $defense -eq "none"
            $epsilonTag = if ($noProtection) { $null } else { $epsilon.ToString([Globalization.CultureInfo]::InvariantCulture) }
            $suffix = if ($noProtection) { "none" } else { "epsilon_$epsilonTag" }
            $shadow = if ($noProtection) { "outputs\runs\fold$fold\shadow\none\updates.npz" } else { "outputs\runs\fold$fold\shadow\$defense\$suffix\updates.npz" }
            $private = if ($noProtection) { "outputs\runs\fold$fold\private\none\updates.npz" } else { "outputs\runs\fold$fold\private\$defense\$suffix\updates.npz" }
            if (-not (Test-Path $shadow) -or -not (Test-Path $private)) {
                Write-Warning "Skipping missing run: fold=$fold defense=$defense epsilon=$epsilonTag"
                continue
            }
            foreach ($count in $SampleCounts) {
                $countTag = if ($noProtection) { "" } else { "_epsilon_$epsilonTag" }
                $fastTag = if ($Fast) { "_fast" } else { "" }
                $output = "outputs\runs\fold$fold\attack_${defense}${countTag}_n$count$fastTag.json"
                if ((Test-Path $output) -and -not $Force) { continue }
                $commandArgs = @("scripts\attack_eval.py", "--shadow", $shadow, "--private", $private,
                    "--num-samples", "$count", "--fold", "$fold", "--defense", $defense, "--output", $output)
                if (-not $noProtection) { $commandArgs += @("--epsilon", "$epsilonTag") }
                if ($Fast) { $commandArgs += "--fast" }
                & $Python @commandArgs
                if ($LASTEXITCODE -ne 0) { throw "Attack failed: fold=$fold defense=$defense epsilon=$epsilonTag n=$count" }
            }
        }
    }
}

Write-Host "Attack grid complete. Review actual update counts in each JSON."
