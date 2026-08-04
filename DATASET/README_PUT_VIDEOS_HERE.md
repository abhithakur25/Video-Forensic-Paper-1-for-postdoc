# DATASET/ — FaceForensics++ videos go here

This tree is **empty on purpose**. FaceForensics++ has no public download; it is
released only after the authors approve a request, so the videos could not be
fetched automatically. The directory layout below is already correct — once you
have the data, drop it in and nothing else needs changing.

## How to get the data

1. Submit the access request form (linked from the dataset README):
   <https://github.com/ondyari/FaceForensics/tree/master/dataset>
   Direct form: <https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform>
2. The authors email you a link to their download script once the request is
   accepted. The script is **not** in the public repository.
3. Download only what this project reads — the FaceSwap subset and the pristine
   originals, both at c23:

   ```
   python <their_download_script>.py <this DATASET dir> -d FaceSwap       -c c23 -t videos
   python <their_download_script>.py <this DATASET dir> -d original       -c c23 -t videos
   ```

## Layout the code expects

`SubFunctions/GetData.py` lines 65-66 glob exactly these two paths:

```
DATASET/manipulated_sequences/FaceSwap/c23/videos/*.mp4     -> label 1 (forged)
DATASET/original_sequences/youtube/c23/videos/*.mp4         -> label 0 (authentic)
```

Nothing else is read. Deepfakes, Face2Face and NeuralTextures are **not** used
by this code, and neither are the raw/c40 compression levels.

## You may not need this at all

`Features/Features.pkl` (1.0 GB) already contains the features extracted from
these videos, so training and evaluation run without the raw data:

```powershell
python .claude\skills\run-video-forgery-paper1\driver.py evaluate --sweep --epochs 10 --skip BA-TFD
```

The raw videos are only required to **re-extract** features, i.e. `Main.py`'s
"Yes" branch via `ReadDataset(exec=True)`. Note that the shipped pickle holds
**50 videos** (29 authentic / 21 forged), so re-extracting from the full
FaceForensics++ release would produce a much larger — and different — feature
set than the one behind the published numbers.
