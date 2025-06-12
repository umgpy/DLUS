import os
import sys
import json
import shutil
from pathlib import Path
import subprocess
import argparse
import logging
from glob import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import tempfile
from pydicom import dcmread

import multiprocessing

# Confirm GPU availability
tf.config.run_functions_eagerly(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Python executable being used:", sys.executable)

import torch

# Ensure that CUDA libraries from the torch library are accessible
#torch_cuda_path = r"C:\Users\user\Desktop\Siemens\syngo_DLUS_nnU1.7.6\Lib\site-packages\torch\lib"
#os.environ["PATH"] += os.pathsep + torch_cuda_path
#print("Updated PATH for CUDA libraries:", os.environ["PATH"])

# Confirm GPU availability
tf.config.run_functions_eagerly(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Python executable being used:", sys.executable)

# Fix for torch.jit issues with PyInstaller
def script_method(fn, _rcb=None):
    return fn

def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

torch.jit.script_method = script_method
torch.jit.script = script

# Import custom modules using absolute imports
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'networks'))

from utils.utilities import check_if_exist
from utils.load_data import load_data
from utils.voi_extraction import run_voi_extraction
from utils.download_pretrained_weights import download_pretrained_weights
from utils.postprocessing import postprocessing_native, export2dicomRT , export_rt_and_dcmseg
from utils.dm_computation import dm_computation


# Set environment variables for nnUNet
if getattr(sys, 'frozen', False):  # Support PyInstaller .exe
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

main_dir = os.path.join(script_dir, 'networks', 'SegmentationNet', 'nnUNet')

def get_bundle_basepath() -> str:
    """
    Return the absolute path where our code (and bundled data files)
    can be found.  In a normal "python main.py" run, this is the
    directory that contains main.py.  When running as a PyInstaller
    --onefile or --onedir bundle, this points inside the extracted
    folder (sys._MEIPASS) or next to the .exe, respectively.
    """
    if getattr(sys, 'frozen', False):
        # We are in a PyInstaller bundle
        #   - For one‐file mode: PyInstaller sets sys._MEIPASS to the
        #     temporary extraction location.
        #   - For one‐folder mode: _MEIPASS may not exist, but
        #     sys.executable is the path to the .exe, and all datas
        #     are next to it.
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        # Running in a normal Python interpreter
        return os.path.abspath(os.path.dirname(__file__))



def parse_arguments():
    parser = argparse.ArgumentParser(description='DLUS DICOM Processing Application')
    parser.add_argument('-i', '--inputdir', required=True, help='Directory where DICOMs or input files are stored')
    parser.add_argument('-o', '--outputdir', required=True, help='Directory to store result DICOM files and intermediate files')
    parser.add_argument('-t', '--tempdir', required=True, help='Directory to copy final DICOM folder')
    parser.add_argument('-l', '--logdir', required=True, help='Directory to store log files')
    parser.add_argument('-c', '--configdir', required=True, help='Directory containing configuration files')
    return parser.parse_args()

# Refined function for downloading pre-trained weights
def download_pretrained_weights(task_id):
    WEIGHTS_URL = {
        '112': "URL_for_FR_model_weights",
        '113': "URL_for_Mixed_model_weights",
        '108': "URL_for_Urethra_model_weights"
    }.get(task_id)

    if WEIGHTS_URL is None:
        raise ValueError(f"No WEIGHTS_URL defined for task_id {task_id}")

    weights_path = os.path.join(os.getcwd(), 'networks', 'SegmentationNet', f"weights_task_{task_id}.hdf5")
    if not os.path.exists(weights_path):
        logging.info(f"Downloading weights for task {task_id}...")
        # Download from WEIGHTS_URL to weights_path using your method of choice
        # For example, urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
        logging.info(f"Weights for task {task_id} downloaded successfully.")
    
    return weights_path


# for a lighter executable 
def copy_weights(task_id: str, src_root: str, dest_root: str):
    """
    Copy pre‐trained weights for nnU‐Net task `task_id`
    from your local `src_root` into the `dest_root/3d_fullres/<task_id>/…` layout.
    """
    src = Path(src_root) / task_id / "nnUNetTrainerV2__nnUNetPlansv2.1"
    if not src.exists():
        raise FileNotFoundError(f"Cannot find weights for task {task_id} at {src}")
    dst = Path(dest_root) / "nnUNet" / "3d_fullres" / task_id / "nnUNetTrainerV2__nnUNetPlansv2.1"
    if dst.exists():
        return   # already in place
    print(f"Copying weights for task {task_id} → {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def load_config(config_path):
    config_file = os.path.join(config_path, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at: {config_file}")
    with open(config_file, 'r') as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    return config

def setup_logging(log_dir):
    """Configure logging to write to the specified log directory."""
    log_file = os.path.join(log_dir, 'DLUS.log')
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # --- capture all future print() calls ---
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line)
        def flush(self):
            pass
    
    # replace stdout/stderr with our logger wrappers
    stdout_logger = logging.getLogger('STDOUT')
    stderr_logger = logging.getLogger('STDERR')
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

def setup_directories(args):
    """Ensure that all required directories exist."""
    dirs = {
        'config': args.configdir,
        'input': args.inputdir,
        'log': args.logdir,
        'output': args.outputdir,
        #'imgs': os.path.join(args.outputdir, 'imgs'),  # Image directory directly within output
    }

    for dir_name, path in dirs.items():
        check_if_exist(path, create=True)

    return dirs

def extract_ct_series_only(original_dicom_path):
    temp_ct_path = tempfile.mkdtemp()
    for f in glob(os.path.join(original_dicom_path, '*.dcm')):
        try:
            ds = dcmread(f, stop_before_pixels=True)
            if ds.Modality == 'CT':
                shutil.copy(f, temp_ct_path)
        except:
            continue
    return temp_ct_path


def main():
    args = parse_arguments()

    # ─── Build a tiny “fake” Input/Output tree in tempdir ───
    tmp_root       = os.path.abspath(args.tempdir)
    fake_input     = os.path.join(tmp_root, "Input")
    fake_output    = os.path.join(tmp_root, "Output")
    # clear any prior runs
    shutil.rmtree(fake_input,  ignore_errors=True)
    shutil.rmtree(fake_output, ignore_errors=True)

    # Derive a single-case “database” name from the input folder
    #ddbb    = os.path.basename(os.path.normpath(args.inputdir))
    folder_name_ddbb    = os.path.basename(os.path.normpath(args.inputdir))
    ddbb = f"{folder_name_ddbb}Data"
    case_id = "001"   # we’ll pretend there’s only one case

    # Copy *all* .dcm files from your inputdir → fake_input/<DB>/<case>/img
    dest_img = os.path.join(fake_input, ddbb, case_id, "img")
    os.makedirs(dest_img, exist_ok=True)
    for f in glob(os.path.join(args.inputdir, "**", "*.dcm"), recursive=True):
        shutil.copy(f, dest_img)


    # Now point the rest of the pipeline at these fake paths
    work_data_path   = os.path.join(fake_input,  ddbb)
    work_output_path = os.path.join(fake_output, ddbb)



    # Load configuration from JSON
    config = load_config(args.configdir)
    local_weights = config["weights_dir"] # add weights from the config

    

    # Setup logging
    setup_logging(args.logdir)
    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("DLUS DICOM Processing Started")
    logging.info("-------------------------------------------------------------------------------------------------------")

    # Setup directories
    dirs = setup_directories(args)
    #ddbb = 'testData'  # Default database name
    mode = config.get("mode", "dicom")  # Options: 'nifti', 'dicom'
    model = config.get("model", "Mixed_model")  # Options: 'FR_model', 'Mixed_model'
    use_manual_OARs = config.get("use_manual_OARs", False)  # Options: False, True
     # ─────────── Use our fake workspace instead ───────────
    
    data_path   = work_data_path
    output_path = work_output_path
    check_if_exist(data_path,   create=False)
    check_if_exist(output_path,  create=True)
    load_data(data_path, output_path, mode)
    
    


    logging.info("1. LOADING ORIGINAL IMAGES...")
    logging.info("-------------------------------------------------------------------------------------------------------")
    
    # Setup input and output paths
    #data_path = os.path.join(dirs['input'], ddbb)
    #check_if_exist(data_path, create=False)
    #output_path = os.path.join(dirs['output'], ddbb)
    #check_if_exist(output_path, create=True)
    
    # Load data
    #load_data(data_path, output_path, mode)

   
    
    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("2. VOI EXTRACTION...")
    logging.info("-------------------------------------------------------------------------------------------------------")
    
    # Set up VOI extraction paths and directories
    #checkpoint_path = os.path.normpath(os.path.join(os.getcwd(), 'networks', 'LocalizationNet', 'LocalizationNet_weights.best.hdf5'))
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'networks', 'LocalizationNet', 'LocalizationNet_weights.best.hdf5')


    dir_ddbb = os.path.normpath(os.path.join(output_path, 'imgs'))
    check_if_exist(os.path.normpath(os.path.join(output_path, 'Urethra', 'GT_VOI')), create=True)
    
    if use_manual_OARs: 
        check_if_exist(os.path.normpath(os.path.join(output_path, 'mVOIs', 'imagesTs')), create=True)
        check_if_exist(os.path.normpath(os.path.join(output_path, 'OARs', 'manual')), create=True)
    else: 
        check_if_exist(os.path.normpath(os.path.join(output_path, 'VOIs', 'imagesTs')), create=True)
    
    # Instead of direct call:
    run_voi_extraction(checkpoint_path, output_path, dir_ddbb, use_manual_OARs=use_manual_OARs)

    #print("getting into the multiprocess")
    # Do:
    #p = multiprocessing.Process(
    #    target=run_voi_extraction,
    #    args=(checkpoint_path, output_path, dir_ddbb, use_manual_OARs)
    #)
    #p.start()
    #p.join()
    
    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("3. OARs SEGMENTATION : Fine Segmentation Network")
    logging.info("-------------------------------------------------------------------------------------------------------")
    
    def skip():

        # Set environment variables for nnUNet
        if getattr(sys, 'frozen', False):  # Support PyInstaller .exe
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        main_dir = os.path.join(script_dir, 'networks', 'SegmentationNet', 'nnUNet')

        def get_bundle_basepath() -> str:
            """
            Return the absolute path where our code (and bundled data files)
            can be found.  In a normal "python main.py" run, this is the
            directory that contains main.py.  When running as a PyInstaller
            --onefile or --onedir bundle, this points inside the extracted
            folder (sys._MEIPASS) or next to the .exe, respectively.
            """
            if getattr(sys, 'frozen', False):
                # We are in a PyInstaller bundle
                #   - For one‐file mode: PyInstaller sets sys._MEIPASS to the
                #     temporary extraction location.
                #   - For one‐folder mode: _MEIPASS may not exist, but
                #     sys.executable is the path to the .exe, and all datas
                #     are next to it.
                return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            else:
                # Running in a normal Python interpreter
                return os.path.abspath(os.path.dirname(__file__))


        #
        # …later, in main() or wherever you before call nnUNet_predict…
        #
        base_path = get_bundle_basepath()

        # Now build the three nnU-Net environment variables relative to base_path:
        #   dist/dlus_app_dist/networks/SegmentationNet/nnUNet/nnUNet_trained_models/3d_fullres/…
        nnunet_root = os.path.join(
            base_path,
            'networks',
            'SegmentationNet',
            'nnUNet'
        )

        os.environ['nnUNet_raw_data_base'] = os.path.join(nnunet_root, 'nnUNet_raw_data_base')
        os.environ['nnUNet_preprocessed']  = os.path.join(nnunet_root, 'nnUNet_preprocessed')
        os.environ['RESULTS_FOLDER']       = os.path.join(nnunet_root, 'nnUNet_trained_models')

    #
    # …later, in main() or wherever you before call nnUNet_predict…
    #
    base_path = get_bundle_basepath()

    # Now build the three nnU-Net environment variables relative to base_path:
    #   dist/dlus_app_dist/networks/SegmentationNet/nnUNet/nnUNet_trained_models/3d_fullres/…
    nnunet_root = os.path.join(
        base_path,
        'networks',
        'SegmentationNet',
        'nnUNet'
    )
  
    os.environ['nnUNet_raw_data_base'] = os.path.join(nnunet_root, 'nnUNet_raw_data_base')
    os.environ['nnUNet_preprocessed']  = os.path.join(nnunet_root, 'nnUNet_preprocessed')
    os.environ['RESULTS_FOLDER']       = os.path.join(nnunet_root, 'nnUNet_trained_models')

      

    #for task in ('Task112_IGRTProstateVOI','Task113_IGRTProstateVOI_Add','Task108_MABUSUrethra_DM_Danielsson'):
    #    copy_weights(task, local_weights, nnunet_root)
        
    #os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir, 'nnUNet_raw_data_base')
    #os.environ['nnUNet_preprocessed'] = os.path.join(main_dir, 'nnUNet_preprocessed')
    #os.environ['RESULTS_FOLDER'] = os.path.join(main_dir, 'nnUNet_trained_models')
    #os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
    
    path_imgs = os.path.join(output_path, 'mVOIs', 'imagesTs') if use_manual_OARs else os.path.join(output_path, 'VOIs', 'imagesTs')
    dir_ddbb_OARs = os.path.join(output_path, 'OARs', model)
    
    logging.info(f"dir_ddbb_OARS: {dir_ddbb_OARs}")
    
    # Predict OAR segmentations
    task_id = '112' if model == 'FR_model' else '113'
    download_pretrained_weights(task_id=task_id)
    subprocess.run([
        'nnUNet_predict',
        '-i', path_imgs,
        '-o', dir_ddbb_OARs,
        '-t', task_id,
        '-tr', 'nnUNetTrainerV2',
        '-ctr', 'nnUNetTrainerV2CascadeFullRes',
        '-m', '3d_fullres',
        '-p', 'nnUNetPlansv2.1',
        '--disable_tta',
        '--num_threads_preprocessing', '1',
        '--num_threads_nifti_save', '1'
    ], check=True)
    
    # Post-processing and exportation
    for file_OARs in sorted(glob(os.path.join(dir_ddbb_OARs, '*.nii.gz'))):
        idx = os.path.basename(file_OARs).split('_')[-1].split('.')[0]
        out_OARs = sitk.ReadImage(file_OARs)
        out_VOI = sitk.ReadImage(os.path.join(path_imgs, f"{ddbb}_{idx}_0000.nii.gz"))
    
        postprocessing_native(out_OARs, output_path, str(idx), 'OARs', ddbb, use_manual_OARs=use_manual_OARs)
        postprocessing_native(out_VOI, output_path, str(idx), 'VOI', ddbb, use_manual_OARs=use_manual_OARs)
    
        
        seg_file = os.path.join(output_path, 'Native', str(idx), f"{ddbb}_{idx}_OARs.nii.gz") 
        save_path = os.path.join(output_path, 'DICOM', str(idx))
        

        original_dicom_path = os.path.join(data_path, idx, 'img')
        ct_only_path = extract_ct_series_only(original_dicom_path)
        
        #export2dicomRT(ct_only_path, seg_file, save_path, 'OARs_DLUS')
        export_rt_and_dcmseg(ct_dicom_folder=ct_only_path, seg_nifti=seg_file, out_path=save_path, seg_name="OARs_DLUS")


    
    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("4. DISTANCE MAP COMPUTATION...")
    logging.info("-------------------------------------------------------------------------------------------------------")
    
    distance_map_path = os.path.join(output_path, 'mDistanceMaps', 'imagesTs') if use_manual_OARs else os.path.join(output_path, 'DistanceMaps', 'imagesTs')
    check_if_exist(distance_map_path, create=True)
    
    dm_computation(dir_ddbb, output_path, model, use_manual_OARs)
    
    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("5. URETHRA SEGMENTATION")
    logging.info("-------------------------------------------------------------------------------------------------------")
    
    path_dm = os.path.join(output_path, 'mDistanceMaps', 'imagesTs') if use_manual_OARs else os.path.join(output_path, 'DistanceMaps', 'imagesTs')
    dir_ddbb_uretra = os.path.join(output_path, 'Urethra', 'manualDLUS' if use_manual_OARs else 'DLUS')
    
    download_pretrained_weights(task_id='108')
    subprocess.run([
        'nnUNet_predict',
        '-i', path_dm,
        '-o', dir_ddbb_uretra,
        '-t', '108',
        '-tr', 'nnUNetTrainerV2',
        '-ctr', 'nnUNetTrainerV2CascadeFullRes',
        '-m', '3d_fullres',
        '-p', 'nnUNetPlansv2.1',
        '--disable_tta'
    ], check=True)
    
    for file_out_urethra in sorted(glob(os.path.join(dir_ddbb_uretra, '*.nii.gz'))):
        idx = os.path.basename(file_out_urethra).split('_')[-1].split('.')[0]
        out_urethra = sitk.ReadImage(file_out_urethra)
        postprocessing_native(out_urethra, output_path, str(idx), 'm_urethra' if use_manual_OARs else 'urethra', ddbb, use_manual_OARs=use_manual_OARs)
        
        seg_file = os.path.join(output_path, 'Native', str(idx), f"{ddbb}_{idx}_urethra.nii.gz") 
        #save_path = os.path.join(output_path, 'Native', 'DICOM', str(idx))
        save_path = os.path.join(output_path, 'DICOM', str(idx))

       
        original_dicom_path = os.path.join(data_path, idx, 'img')
        ct_only_path = extract_ct_series_only(original_dicom_path)
        
        #export2dicomRT(ct_only_path, seg_file, save_path, 'urethra_DLUS')
        export_rt_and_dcmseg(ct_dicom_folder=ct_only_path, seg_nifti=seg_file, out_path=save_path, seg_name="urethra_DLUS")


    # Copy the DICOM folder to tempdir
    dicom_dir = os.path.join(output_path, 'DICOM')
    temp_dicom_dir = os.path.join(args.tempdir, 'DICOM')
    #shutil.copytree(dicom_dir, temp_dicom_dir, dirs_exist_ok=True)

     #
    # ─────────── Harvest final RTSTRUCTs into the real outputdir ───────────
    #
    real_rt_out = os.path.join(dirs['output'])
    temp_rt_out = os.path.join(work_output_path, "DICOM")
    for case_id in os.listdir(temp_rt_out):
        src = os.path.join(temp_rt_out, case_id)
        dst = os.path.join(real_rt_out) #, case_id)
        os.makedirs(dst, exist_ok=True)
        for f in glob(os.path.join(src, "*_rt-structs.dcm")):
            shutil.copy(f, dst)
        #for f in glob(os.path.join(src, "*_dcmseg.dcm")):
            #shutil.copy(f, dst)


    logging.info("-------------------------------------------------------------------------------------------------------")
    logging.info("DLUS DICOM Processing Completed Successfully")
    logging.info("-------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
