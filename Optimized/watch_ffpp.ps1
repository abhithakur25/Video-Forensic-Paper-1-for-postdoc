# Live FF++ extraction progress. Opens in its own console window and redraws
# in place; close the window or press Q to stop. Read-only - it never touches
# the extraction, it only counts what is on disk and reads the shard logs.

$P      = 'C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1'
$FRAMES = "$P\DATASET\ffpp_frames"
$LOGS   = "$P\logs"
$TOTAL  = 2000 * 32
$EVERY  = 5          # seconds between redraws
$WINDOW = 90         # seconds of history the rate is averaged over

$host.UI.RawUI.WindowTitle = 'FF++ extraction'
try { [Console]::CursorVisible = $false } catch {}
$samples = New-Object System.Collections.ArrayList

function Bar($frac, $w) {
    $f = [Math]::Max(0.0, [Math]::Min(1.0, $frac))
    $fill = [int][Math]::Round($f * $w)
    ([char]0x2588).ToString() * $fill + ([char]0x2591).ToString() * ($w - $fill)
}

function Line($s) {
    # pad so a shorter line never leaves fragments of the previous frame
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

    $now = Get-Date
    $n = @([System.IO.Directory]::EnumerateFiles(
            $FRAMES, '*.jpg', [System.IO.SearchOption]::AllDirectories)).Count

    [void]$samples.Add([pscustomobject]@{ t = $now; n = $n })
    while ($samples.Count -gt 2 -and
           ($now - $samples[0].t).TotalSeconds -gt $WINDOW) {
        $samples.RemoveAt(0)
    }
    $rate = 0.0
    if ($samples.Count -ge 2) {
        $dt = ($now - $samples[0].t).TotalMinutes
        if ($dt -gt 0) { $rate = ($n - $samples[0].n) / $dt }
    }
    $eta = if ($rate -gt 1) { ($TOTAL - $n) / $rate } else { [double]::NaN }

    $shards = foreach ($i in 0..7) {
        $f = "$LOGS\ffpp_shard$i.log"
        $v = 0; $t = '--:--:--'; $fin = $false
        if (Test-Path $f) {
            $lines = Get-Content $f -ErrorAction SilentlyContinue
            $fin = [bool]($lines -match 'EXITCODE=')
            $last = $lines | Where-Object { $_ -match '(\d+)/250 videos' } |
                    Select-Object -Last 1
            if ($last -match '(\d+)/250 videos') { $v = [int]$Matches[1] }
            if ($last -match '^\[(\d+:\d+:\d+)\]') { $t = $Matches[1] }
        }
        [pscustomobject]@{ i = $i; v = $v; t = $t; done = $fin }
    }
    $ndone = @($shards | Where-Object { $_.done }).Count
    $alive = @(Get-Process python -ErrorAction SilentlyContinue).Count

    [Console]::SetCursorPosition(0, 0)
    Line ''
    Line '  FF++ face-crop extraction              (Q or close window to exit)'
    Line ''
    Line ("  {0}  {1,6:N2}%" -f (Bar ($n / $TOTAL) 46), ($n / $TOTAL * 100))
    Line ("  {0:N0} / {1:N0} crops" -f $n, $TOTAL)
    Line ''
    foreach ($s in $shards) {
        $tag = if ($s.done) { 'done' } else { $s.t }
        Line ("  shard {0}  {1} {2,3}/250  {3}" -f
              $s.i, (Bar ($s.v / 250) 28), $s.v, $tag)
    }
    Line ''
    $etaTxt = if ([double]::IsNaN($eta)) { 'measuring...' }
              else { "{0:N0} min  (~{1})" -f $eta, $now.AddMinutes($eta).ToString('HH:mm') }
    Line ("  rate {0,5:N0} crops/min over {1}s     eta {2}" -f $rate, $WINDOW, $etaTxt)
    Line ("  shards finished {0}/8     python {1}     {2}" -f
          $ndone, $alive, $now.ToString('HH:mm:ss'))
    Line ''

    if ($ndone -ge 8) {
        Line '  ALL 8 SHARDS FINISHED - finish_ffpp.ps1 merges and zips next.'
        break
    }
    if ($alive -eq 0) {
        Line ("  WARNING: no python.exe running with only {0}/8 finished." -f $ndone)
        Line '  Relaunch: Optimized\run_ffpp_shard.bat <0-7> via WMI.'
        break
    }
    Start-Sleep -Seconds $EVERY
}
try { [Console]::CursorVisible = $true } catch {}
Write-Host ''
Read-Host '  press Enter to close'
