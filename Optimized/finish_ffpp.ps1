# Wait for the 8 FF++ crop shards, merge their manifests, and package the
# result for Google Drive.
#
# The zip is only written if the merge succeeds, and the merge is what runs
# ffpp_prepare.py's identity-leak assertion. A dataset that leaks an identity
# across splits must never reach the training notebook - a leaked split was
# worth +27 accuracy points on the small subset, which is the whole reason the
# identity protocol exists. So: no assertion, no archive.
#
# Idempotent. Safe to re-run if the session dies before it finishes.

$P   = 'C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1'
$E   = 'C:\Users\USER\anaconda3\envs\VideoForgeryCPU'
$log = "$P\logs"
$out = "$P\DATASET\ffpp_frames"
$zip = "$P\DATASET\ffpp_frames.zip"

function Say($m) { "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m }

# --- 1. wait for all 8 shards ------------------------------------------------
while ($true) {
    $files = @(Get-ChildItem "$log\ffpp_shard*.log")
    $done  = @($files | Where-Object {
        Select-String -Path $_.FullName -Pattern 'EXITCODE=' -Quiet }).Count
    $alive = @(Get-Process python -ErrorAction SilentlyContinue).Count
    if ($done -ge 8) { Say "all 8 shards wrote EXITCODE"; break }
    if ($alive -eq 0) {
        Say "ABORT: $done of 8 shards finished, no python.exe running."
        Say "Shards died again. Relaunch with Optimized\run_ffpp_shard.bat."
        exit 1
    }
    Start-Sleep -Seconds 60
}

# --- 2. every shard must have exited clean -----------------------------------
$bad = @()
foreach ($f in Get-ChildItem "$log\ffpp_shard*.log") {
    $m = @(Select-String -Path $f.FullName -Pattern 'EXITCODE=(-?\d+)')
    $code = $m[-1].Matches[0].Groups[1].Value
    Say ("  {0}  EXITCODE={1}" -f $f.Name, $code)
    if ($code -ne '0') { $bad += "$($f.Name)=$code" }
}
if ($bad.Count) { Say "ABORT: non-zero exit: $($bad -join ', ')"; exit 1 }

# --- 3. merge the shard manifests + identity-leak assertion ------------------
$env:PATH = "$E\Library\bin;$E;$E\Scripts;$env:PATH"
$env:PYTHONWARNINGS = 'ignore'
Say "merging shard manifests"
& "$E\python.exe" -u "$P\Optimized\ffpp_prepare.py" --merge 2>&1 |
    Tee-Object -FilePath "$log\ffpp_merge.log"
if ($LASTEXITCODE -ne 0) {
    Say "ABORT: merge failed (exit $LASTEXITCODE) - see logs\ffpp_merge.log"
    Say "If it reported an identity leak, DO NOT zip or train on this data."
    exit 1
}
if (-not (Test-Path "$out\manifest.csv")) {
    Say "ABORT: merge reported success but manifest.csv is missing"; exit 1
}

# --- 4. package for Drive ----------------------------------------------------
# Store, don't deflate: the payload is 64k JPEGs, already entropy-coded, so
# compression buys ~nothing and costs many minutes over that many files.
# includeBaseDirectory=$false puts train/ val/ test/ manifest.csv at the archive
# root, which is the layout the Colab notebook's unzip step expects.
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path $zip) { Say "removing previous $([IO.Path]::GetFileName($zip))"; Remove-Item $zip -Force }
Say "zipping $out"
$t0 = Get-Date
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $out, $zip, [System.IO.Compression.CompressionLevel]::NoCompression, $false)
$mins = ((Get-Date) - $t0).TotalMinutes

$gb  = (Get-Item $zip).Length / 1GB
$n   = @(Get-ChildItem $out -Recurse -File -Filter *.jpg).Count
Say ("DONE  {0}  {1:N2} GB  {2:N0} crops  ({3:N1} min)" -f $zip, $gb, $n, $mins)
Say "Upload to Google Drive as MyDrive/ffpp_frames.zip, then run the notebook."
