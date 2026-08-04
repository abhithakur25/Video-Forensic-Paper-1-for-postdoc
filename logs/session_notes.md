# NOTES — building run-skills for the video forgery papers

Host: Windows 11 Pro, PowerShell 5.1. NOT a Linux container.
Units: `CODE_28-04-2025(Paper1)`, `CODE_05-08-2025(Paper2)` under C:\Users\USER\Downloads\PostDoc

## Environment

- Default `python` on PATH is 3.14.6 — useless here (TF 2.10 / numpy 1.21.6 don't build for it).
- Anaconda at `C:\Users\USER\anaconda3`. Envs: `VideoForgeryCPU` (py3.8.20) matches
  requirements.txt almost exactly; `forgery_env` (py3.10, TF 2.19/keras 3) does NOT
  (keras 3 breaks `keras.applications.resnet` imports + `Model(inputs=...)` usage).
- Interpreter: `C:\Users\USER\anaconda3\envs\VideoForgeryCPU\python.exe`

## Errors hit → fixes

1. **`import skimage` hard-crashes the interpreter** (exit -1066598273 = 0xC0000409,
   "stack buffer overrun"). `-X faulthandler` traced it to
   `skimage/color/colorconv.py:396` → `scipy.linalg.inv` → Windows exception
   `0xc06d007f` = **DLL delay-load failure**.
   FIX: conda env's `Library\bin` must be on PATH. Running `python.exe` by absolute
   path (no `conda activate`) omits it. `conda activate VideoForgeryCPU` or prepend
   `<env>\Library\bin` to PATH.

2. **`pip install PySimpleGUI==4.60.5` fails** — "Could not find a version...
   (from versions: 4.60.5.1, 6.0, 6.2, 6.3)". 4.60.5 was pulled from PyPI when the
   project relicensed. FIX: `PySimpleGUI==4.60.5.1` (last free 4.x). Provides
   `popup_yes_no`, which is all Main.py uses.

3. **customtkinter**: `pip install customtkinter` gives 6.0.0; requirements pin 5.1.3.
   Installed 5.1.3 explicitly.

4. **torch vs conda MKL — mutual DLL conflict.**
   - conda `Library\bin` prepended → `import torch` = OSError WinError 182 loading
     `torch\lib\shm.dll`.
   - torch imported first → `scipy.linalg.inv` hard-crashes (0xC0000409).
   - `os.add_dll_directory` (before or after torch) does not help.
   Root cause: numpy/scipy/scikit-image are **conda-forge MKL builds** (mkl 2025.3.0,
   libblas 3.11.0 *_mkl); pip torch 1.13.1 bundles its own MKL/OpenMP.
   FIX ATTEMPT: install torch from conda so it links the same MKL:
   `conda install -n VideoForgeryCPU -c pytorch -c conda-forge pytorch=1.13.1 cpuonly`
   torch IS required — `SubFunctions/__init__` → Analysis → Model → Attention → torch,
   so it is on the import path for BOTH the plots path and GUI.py.

5. **Importing `SubFunctions` downloads 733 MB of keras weights** —
   `GetFeatures.py` instantiates `ResNet101()` and `VGG16()` at *module scope*.
   resnet101 h5 = 180 MB, vgg16 h5 = 553 MB → `~/.keras/models`. First run only,
   but it happens on plain `import`, with a progress bar that floods stdout.

6. **`Temp\\themes\\rose.json`, `Analysis\\TP\\COM_A.npy`, vendored `./mealpy`** are all
   resolved relative to CWD → driver must chdir to the project root.

7. **Blocking prompts** (why a driver is needed at all):
   - `Main.py` line 5: `popup_yes_no("Do You want Complete Execution?")` — modal.
   - `PlotResults()` defaults `show=True` → `plt.show()` per figure (~40 figures).
     Use `PlotResults(show=False, save=True)` + `matplotlib.use("Agg")`.
   - `GUI.select_data_event` → `filedialog.askopenfilename` — modal native dialog.
     Monkeypatch it before importing GUI.
   - `GUI.py` only entry is `app.mainloop()`. Driver builds `App()` and calls the
     handler methods directly, pumping `update()` manually to force repaints.

8. **DPI scaling / screenshots.** This display runs at 200%. Tk reports window coords
   in logical units; `PIL.ImageGrab` works in physical pixels → capturing
   `winfo_rootx/rooty/width/height` grabs the wrong screen region (verified: got the
   desktop behind the window). FIX: ratio = `ImageGrab.grab().size[0] /
   root.winfo_screenwidth()`, multiply the bbox. Self-correcting (1.0 if the process
   is DPI-aware — customtkinter makes itself aware — 2.0 if not).

9. **stdout buffering hides crash output.** On a hard crash (0xC0000409) buffered
   stdout is lost entirely — the run looks silent. Always use `python -u`.

10. **PowerShell `$?` lies about native exit codes.** `python -c "import seaborn"`
    printed nothing to stdout but wrote a warning to stderr; `$?` went False even
    though `$LASTEXITCODE` was 0. Check `$LASTEXITCODE`, not `$?`.

11. **No DATASET / no video files ship with either repo.** GUI's Select Video needs
    one → driver synthesizes `driver_out/sample.mp4` from the 75 sample frames in
    `Results/ImageResults/Input/`.

12. `SubFunctions` prints emoji via termcolor → UnicodeEncodeError on cp1252 consoles.
    Driver reconfigures stdout to utf-8/replace.

## Still to verify
- [ ] conda pytorch resolves the MKL conflict
- [ ] driver check passes fully
- [ ] plots path writes figures
- [ ] GUI walkthrough + screenshots
- [ ] same for Paper1
