# Live view of the SMA-CLMPNet nested search. Opens in its own console window
# and redraws in place; close the window or press Q to stop.
#
# Progress is counted in FITS, not folds, because a fold is ~2.7 h and would
# leave the bar frozen for most of the run. Each "cfg c/N" line in the log
# means `inner` fits finished; each completed outer fold adds one refit.
#
# Read-only: it parses the log and the checkpoint, and never touches the run.

$P     = 'C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1'
$LOG   = "$P\logs\smaclmpnet_search.log"
$ERRF  = "$P\logs\smaclmpnet_search.err"
$JSON  = "$P\Optimized\smaclmpnet_search.json"
$OUTER = 5
$INNER = 2
$NCFG  = 6
$TOTAL = $OUTER * $INNER * $NCFG + $OUTER      # 65 fits
$EVERY = 10

$host.UI.RawUI.WindowTitle = 'SMA-CLMPNet search'
try { [Console]::CursorVisible = $false } catch {}

function Bar($frac, $w) {
    $f = [Math]::Max(0.0, [Math]::Min(1.0, $frac))
    $fill = [int][Math]::Round($f * $w)
    ([char]0x2588).ToString() * $fill + ([char]0x2591).ToString() * ($w - $fill)
}
function Line($s) {
    $w = [Math]::Max(40, $host.UI.RawUI.WindowSize.Width - 1)
    if ($s.Length -gt $w) { $s = $s.Substring(0, $w) }
    Write-Host $s.PadRight($w)
}

Clear-Host
while ($true) {
    if ([Console]::KeyAvailable) {
        $k = [Console]::ReadKey($true)
        if ($k.Key -eq 'Q' -or $k.Key -eq 'Escape') { break }
    }

    $now  = Get-Date
    $body = if (Test-Path $LOG) { Get-Content $LOG -ErrorAction SilentlyContinue } else { @() }

    $cfgLines   = @($body | Where-Object { $_ -match 'fold (\d+) cfg (\d+)/(\d+) inner bal ([\d.]+)' })
    $outerLines = @($body | Where-Object { $_ -match 'outer fold (\d+)/(\d+): chose' })
    $started    = @($body | Where-Object { $_ -match 'configurations, outer' }).Count -gt 0

    $fits = $cfgLines.Count * $INNER + $outerLines.Count
    $t0 = $null
    if ($body -and ($body[0] -match '^\[(\d+:\d+:\d+)\]')) {
        $t0 = [datetime]::ParseExact($Matches[1], 'HH:mm:ss', $null)
        if ($t0 -gt $now) { $t0 = $t0.AddDays(-1) }
    }
    $elapsed = if ($t0) { ($now - $t0).TotalMinutes } else { 0 }
    $perFit  = if ($fits -gt 0) { $elapsed / $fits } else { 0 }
    $eta     = if ($perFit -gt 0) { ($TOTAL - $fits) * $perFit } else { [double]::NaN }

    # per outer fold: how many of its configs have reported
    $byFold = @{}
    foreach ($l in $cfgLines) {
        if ($l -match 'fold (\d+) cfg (\d+)/(\d+) inner bal ([\d.]+)') {
            $f = [int]$Matches[1]
            if (-not $byFold.ContainsKey($f)) { $byFold[$f] = @() }
            $byFold[$f] += [double]$Matches[4]
        }
    }
    $doneFolds = @{}
    foreach ($l in $outerLines) {
        if ($l -match 'outer fold (\d+)/(\d+): chose.*OUTER ([\d.]+)') {
            $doneFolds[[int]$Matches[1]] = [double]$Matches[3]
        }
    }

    $alive    = @(Get-Process python -ErrorAction SilentlyContinue).Count
    $finished = @($body | Where-Object { $_ -match 'EXITCODE=' }).Count -gt 0
    $errSize  = if (Test-Path $ERRF) { (Get-Item $ERRF).Length } else { 0 }

    [Console]::SetCursorPosition(0, 0)
    Line ''
    Line '  SMA-CLMPNet recipe search - nested 5 outer x 2 inner, 6 configs'
    Line '                                        (Q or close window to exit)'
    Line ''
    Line ("  {0}  {1,6:N1}%" -f (Bar ($fits / $TOTAL) 46), ($fits / $TOTAL * 100))
    Line ("  {0} / {1} fits" -f $fits, $TOTAL)
    Line ''
    foreach ($f in 1..$OUTER) {
        $c = if ($byFold.ContainsKey($f)) { $byFold[$f].Count } else { 0 }
        $tag = if ($doneFolds.ContainsKey($f)) {
            "done  OUTER {0,6:N2}" -f $doneFolds[$f]
        } elseif ($c -gt 0) {
            "best inner {0,6:N2}" -f ($byFold[$f] | Measure-Object -Maximum).Maximum
        } else { '' }
        Line ("  fold {0}  {1} {2}/{3} cfg   {4}" -f
              $f, (Bar ($c / $NCFG) 24), $c, $NCFG, $tag)
    }
    Line ''
    $etaTxt = if ([double]::IsNaN($eta) -or $fits -eq 0) { 'measuring (first fit ~13.5 min)...' }
              else { "{0:N1} h  (~{1})" -f ($eta / 60), $now.AddMinutes($eta).ToString('ddd HH:mm') }
    Line ("  elapsed {0,5:N0} min    per fit {1,5:N1} min    eta {2}" -f
          $elapsed, $perFit, $etaTxt)
    Line ("  python {0}    stderr {1} B    {2}" -f $alive, $errSize,
          $now.ToString('HH:mm:ss'))
    Line ''
    $tail = @($body | Where-Object { $_ -match '^\[\d' } | Select-Object -Last 1)
    Line ("  " + $(if ($tail) { $tail[0] } else { 'waiting for first log line' }))
    Line ''

    if ($finished) { Line '  RUN FINISHED - see Optimized\smaclmpnet_search.json'; break }
    if ($started -and $alive -eq 0) {
        Line '  WARNING: no python.exe running and no EXITCODE line.'
        Line '  Resume with: Optimized\run_smaclmpnet_search.bat (via WMI)'
        break
    }
    Start-Sleep -Seconds $EVERY
}
try { [Console]::CursorVisible = $true } catch {}
Write-Host ''
Read-Host '  press Enter to close'
