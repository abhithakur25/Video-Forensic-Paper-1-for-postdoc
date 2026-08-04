## <div align="center"> TITLE</div>
## DESIGN AND DEVELOPMENT OF VIDEO FORGERY MODEL USING DEEP LEARNING WITH ATTENTION MECHANISMS
<br>

## HARDWARE REQUIREMENTS⚙️
<br>
    OS-Windows 11
<br>
    RAM-16GB
<br>
    ROM-More than 100 GB
<br>
    CPU-1.7 Ghz
<br>

## SOFTWARE REQUIREMENTS💻

<b>Software name: </b> Python: Version: <b>3.7.6</b>([Download link](https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe))<br>
<b>Software name: </b> Pycharm: Version: <b>2024.1.7</b>([Download link](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC))<br>
<br>

## HOW TO RUN➿
    Step 1: Loading the project in PYCHARM
        1. Open pycharm
        2. Go to File, select Open browse the project from your drive and select it. So that the
        project will get loaded into the Pycharm.
        3. For the first time, Pycharm will take some time to load the settings.
        4. Please wait if any process is loading at the bottom of the screen.
        5. Check the Project Interpreter (File -&gt; Settings -&gt; Project Interpreter).
        If this location “(C:\Users\---\AppData\Local\Programs\Python\Python37\python.exe) is
        not presented, then add this ‘python.exe’ from the installed location.

    Step 2: Generate the graphs plotted in the paper
        1. current project window in pycharm, Open ‘Main.py’, and click
        run button.
        2. When you execute this it will ask you to select anyone from two options like Yes (“Full
        analysis +Plots”) and NO (“Plots from pre evaluated data”).
        3. · If you select the first option, it will execute the complete analysis again and
        display the results. It may take time depending on the analysis.

        4. · Second option will display the results from the stored data i.e, mainly to skip the
        execution time.
        [Execution time expected: “Full analysis +Plots” — 48 hr]
        [Execution time expected: “Plots from pre evaluated data” – 30 sec]

## <div align="center">DESCRIPTION</div>

    1: preprocessing and Feature Extraction function (SRC-GetData.py : Line NO: 15-140)
        1.1. Preprocessing - (SRC-GetPreprocessing.py : Line NO: 8-48)
        1.2. Grand cam based Deep flow map - (SRC-GetFeatures.py : Line NO: 31-53)
        1.3. Hybrid Resnet 101 based statistical features - (SRC-GetFeatures.py : Line NO: 123-136)
        1.4. Hybrid vgg.16 based LDZP - (SRC-GetFeatures.py : Line NO: 140-141)
        1.5. Optical flow map - (SRC-GetFeatures.py : Line NO: 155-197)


    2: Training Percentage Analysis Part (SRC-Analysis.py : Line NO: 129-308)
        2.1. Commparative Analysis - (SRC-Analysis.py : Line NO: 143-95)
        2.2. Performance Analysis - (SRC-Analysis.py : Line NO: 197-247)
        2.2. ROC Analysis - (SRC-Analysis.py : Line NO: 249-308)


    3: KFold Analysis Part (SRC-Analysis.py : Line NO: 312-499)
        3.1. Commparative Analysis - (SRC-Analysis.py : Line NO: 387-450)
        3.2. Performance Analysis - (SRC-Analysis.py : Line NO: 452-499)


    4: 3DCNN based Distributed LSTM with modified pooling (SRC-Model.py : Line NO: 447-511)

    5: Existing Methods  (SRC-Model.py : Line NO:141-444)



See below for a installation for full documentation on training, testing.

<details open>
<summary>Packages Installation</summary>

```bash
pip install -r requirements.txt
```
</details>



## <div align="center">DATASET</div>

  <b>Dataset 1: </b> FaceForensics++ ([Download link](https://www.niessnerlab.org/projects/roessler2019faceforensicspp.html))<br>
