# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

#
# STEP 1: Because __file__ isn’t defined inside a spec, use os.getcwd().
# We assume you invoke “pyinstaller main.spec” from the folder that also
# contains main.py, networks/, utils/, and (optionally) config/.
#
project_root = os.getcwd()

#
# STEP 2: Build a “hiddenimports” list so that dynamic imports (e.g. inside
# nnUNet, TensorFlow, or your own utils/ code) are not missed.
#
hiddenimports = []
hiddenimports += collect_submodules('networks')
hiddenimports += collect_submodules('utils')
# If you know other dynamic modules (e.g. highdicom, rt_utils, pydicom),
# you can append them here as strings:
# hiddenimports += ['tensorflow', 'torch', 'highdicom', 'rt_utils', 'pydicom']

#
# STEP 3: Walk “networks/” and “utils/” (and “config/” if present) and add
# every file under them into the `datas` list.  Each tuple is (source_path, target_dir).
#
datas = []

# (A) Include everything under networks/ → dist/dlus_app_dist/networks/…
networks_dir = os.path.join(project_root, 'networks')
if os.path.isdir(networks_dir):
    for root, _, files in os.walk(networks_dir):
        for fname in files:
            src_file = os.path.join(root, fname)
            # compute the relative path under “project_root”
            rel_dir = os.path.relpath(root, project_root)
            datas.append((src_file, rel_dir))

# (B) Include everything under utils/ → dist/dlus_app_dist/utils/…
utils_dir = os.path.join(project_root, 'utils')
if os.path.isdir(utils_dir):
    for root, _, files in os.walk(utils_dir):
        for fname in files:
            src_file = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, project_root)
            datas.append((src_file, rel_dir))

# (C) If you have a config/ folder with JSON files, include those too:
config_dir = os.path.join(project_root, 'config')
if os.path.isdir(config_dir):
    for root, _, files in os.walk(config_dir):
        for fname in files:
            src_file = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, project_root)
            datas.append((src_file, rel_dir))

#
# STEP 4: Build the Analysis → PYZ → EXE → COLLECT pipeline.
#
a = Analysis(
    ['main.py'],                # your entry‐point script
    pathex=[project_root],      # allow imports from project_root
    binaries=[],
    datas=datas,                # “datas” contains all source‐to‐target tuples
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Exclude “unittest” so TensorFlow doesn’t crash at runtime
    excludes=['tkinter', 'matplotlib', 'seaborn'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],                # no extra binaries
    exclude_binaries=True,
    name='dlus_app',   # name of final executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,      # keep console window (so prints/logs appear)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,           # place all “datas” under dist/dlus_app_dist/
    strip=False,
    upx=True,
    name='dlus_app_dist',

)

