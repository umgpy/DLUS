
# DLUS: Deep Learning Urology Segmentation Tool


Folder `Syngo.via OAF App` contains the manifest file and `.exe` to generate OAF App used in Syngo.Via

`dlus_app_dist.zip` contains the executable file which can run on windows.

## Configuration

Create a `config.json` file inside your configuration directory. Minimal example:

```json
{
  "mode": "dicom",
  "model": "Mixed_model",
  "use_manual_OARs": false,
  "weights_dir": "C:/path/to/nnUNet_trained_models"
}
```

### Config Fields

* **`mode`**: `"dicom"` or `"nifti"`\
  Determines output format: DICOM SEG or NIfTI.

* **`model`**: `"FR_model"` or `"Mixed_model"`\
  Specifies which segmentation model to use.

* **`use_manual_OARs`**: `true` or `false`\
  Set to `true` to bypass automatic VOI extraction and use predefined masks.

* **`weights_dir`**: _(Optional)_ Path to pretrained model weights.\
  If not provided, the tool assumes models are bundled. Directory structure should look like:

  ```
  3d_fullres/
    Task108_MABUSUrethra_DM_Danielsson/
      nnUNetTrainerV2__nnUNetPlansv2.1/
    Task112_IGRTProstateVOI/
      nnUNetTrainerV2__nnUNetPlansv2.1/
    Task113_IGRTProstateVOI_Add/
      nnUNetTrainerV2__nnUNetPlansv2.1/
  ```

***

## Running the Script (Python)

```bash
python main.py \
  -i /path/to/InputDir \
  -o /path/to/OutputDir \
  -t /path/to/TempDir \
  -l /path/to/LogDir \
  -c /path/to/ConfigDir
```

### Arguments

* `-i`, `--inputdir`: Folder with input DICOMs or NIfTIs

* `-o`, `--outputdir`: Directory to store final outputs

* `-t`, `--tempdir`: Temporary working directory

* `-l`, `--logdir`: Location to write `DLUS.log`

* `-c`, `--configdir`: Folder containing `config.json`

***

## Running the Executable

### On Windows (PowerShell)

```powershell
.\dlus_app.exe `
  -i C:\Data\InputDir `
  -o C:\Data\OutputDir `
  -t C:\Data\TempDir `
  -l C:\Data\LogDir `
  -c C:\Data\ConfigDir
```

### On Linux / macOS

```bash
./dlus_app \
  -i /mnt/data/InputDir \
  -o /mnt/data/OutputDir \
  -t /mnt/data/TempDir \
  -l /mnt/data/LogDir \
  -c /mnt/data/ConfigDir
```

***

## Example Directory Structure

```
Project/
├── ConfigDir/
│   └── config.json
├── InputDir/
│   └── ... (DICOMs or NIfTIs)
├── main.py
└── dlus_app.exe (optional)
```

***

## Example Commands

### Python

```bash
python main.py \
  -i ./InputDir \
  -o ./Output \
  -t ./Tmp \
  -l ./Logs \
  -c ./ConfigDir
```

### Windows Executable

```powershell
.\dlus_app.exe `
  -i .\InputDir `
  -o .\Output `
  -t .\Tmp `
  -l .\Logs `
  -c .\ConfigDir
```
