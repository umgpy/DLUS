# Imports
import os
import numpy as np
import SimpleITK as sitk
import multiprocessing
from glob import glob
import pandas as pd
import time

from utils.utilities import load_data, check_if_exist
from utils.voi_extraction import run_voi_extraction
from utils.dm_computation import dm_computation
from utils.postprocessing import postprocessing_native, export2dicomRT

# Set up paths
ddbb      = 'PRUEBAS'
mode      = 'dicom' #'nifti'
modality  = 'CT'
model     = 'FR_model' #'Mixed_model'
data_path = os.getcwd()+'/Input/'+ddbb
check_if_exist(data_path, create=False) # Check that the path exists
out_path  = os.path.join(os.getcwd(), 'Output', ddbb, modality)
check_if_exist(out_path)

###############################################################################################################

start_ini = time.time()
# 1. Load patients (DICOM or NIfTI images)
print("-------------------------------------------------------------------------------------------------------")
print("1. LOADING ORIGINAL IMAGES...")
print("-------------------------------------------------------------------------------------------------------")
# CT and MR data have to be loaded separately, to indicate if they are in DICOM or NIfTI format
load_data(data_path, out_path, mode, modality=modality)

# 2. VOI Extraction: Localization Network + Crop using the centroid of the coarse segmentation. Check the result to ensure that
# appropriate VOI has been created. Sometimes some images are not well predicted and it's necessary to modify this VOI manually
# to ensure that the OARs and urethra segmentations are accurate.
print("-------------------------------------------------------------------------------------------------------")
print("2. VOI EXTRACTION...")
print("-------------------------------------------------------------------------------------------------------")
checkpoint_path = os.getcwd()+'/networks/LocalizationNet/LocalizationNet_weights.best.hdf5'
dir_ddbb     = os.path.join(out_path, modality+'s')
check_if_exist(os.path.join(out_path, 'VOIs', 'imagesTs')) # Path to save generated VOIs

p = multiprocessing.Process(target=run_voi_extraction, args=(checkpoint_path, out_path, dir_ddbb))
p.start()
p.join()

# 3. OARs Segmentation: Fine Segmentation Network
print("-------------------------------------------------------------------------------------------------------")
print("3. OARs SEGMENTATION (OUTPUT 1)...")
print("-------------------------------------------------------------------------------------------------------")
# Environmnet Variables
main_dir  = os.path.join(os.getcwd(), 'networks/SegmentationNet/nnUNet')
# Can be set in the the .bashrc file or exported in the terminal:
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed']  = os.path.join(main_dir,'nnUNet_preprocessed')
os.environ['RESULTS_FOLDER']       = os.path.join(main_dir,'nnUNet_trained_models')
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"

path_imgs    = os.path.join(out_path, 'VOIs', 'imagesTs')
dir_ddbb_OARs = os.path.join(out_path, 'OARs', model)

# Predict OAR segmentations (rectum, bladder, prostate, seminal vesicles)
if model=='FR_model':
    !nnUNet_predict -i {path_imgs} -o {dir_ddbb_OARs} -t 112 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 --disable_tta
    
if model=='Mixed_model':
    !nnUNet_predict -i {path_imgs} -o {dir_ddbb_OARs} -t 113 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 --disable_tta
    
for i, file_OARs in enumerate(sorted(glob(dir_ddbb_OARs + r'/*.nii.gz'))):
    idx = file_OARs.split('/')[-1].split('_')[-1].split('.')[0]
    print(file_OARs)
    out_OARs  = sitk.ReadImage(file_OARs)
    out_VOI   = sitk.ReadImage(os.path.join(path_imgs, ddbb+'_'+idx+'_0000.nii.gz'))
    
    # 3.1 Post-processing to Native space
    print("3.1 OARs SEGMENTATION POST-PROCESSING TO NATIVE SPACE -------------------------------------------------")
    postprocessing_native(out_OARs, out_path, str(idx),'OARs', ddbb, modality)
    postprocessing_native(out_VOI, out_path, str(idx),'VOI', ddbb, modality)
    
    # 3.2 Exportation to Dicom-RT
    print("3.2 OARs SEGMENTATION EXPORTATION TO DICOM-RT ---------------------------------------------------------")
    img_file  = os.path.join(out_path, 'Native', str(idx), ddbb+'_'+str(idx)+'_VOI.nii.gz')
    seg_file = os.path.join(out_path, 'Native', str(idx), ddbb+'_'+str(idx)+'_OARs.nii.gz')
    save_path = os.path.join(out_path, 'Native', 'DICOM', str(idx))
    # For the moment, we can only translate CTs and their segmentations to DICOM; not MR images
    if modality=='CT':
        export2dicomRT(img_file, seg_file, save_path, 'OARs_DLUS', modality)
        
# 4. DISTANCE MAP COMPUTATION
print("-------------------------------------------------------------------------------------------------------")
print("4. DISTANCE MAP COMPUTATION...")
print("-------------------------------------------------------------------------------------------------------")
dir_ddbb_ct = out_path+'/CTs'
check_if_exist(out_path+'/DistanceMaps/imagesTs_2')
# Bladder Prostate Segmentation and DM Computation
for i, path in enumerate(os.listdir(dir_ddbb_ct)):
    idx = path.split('_')[1]
    # Check if image already loaded
    file_img_name = out_path+'/DistanceMaps/imagesTs_2/MABUS_'+idx+'_0000.nii.gz'
    if os.path.exists(file_img_name):
        continue
    print('Processing case: ', idx)
    file_OARs = out_path+'/OARs/'+model+'/IGRT_'+idx+'.nii.gz'
    out_OARs  = sitk.ReadImage(file_OARs)
    dm_computation(out_OARs, file_img_name, idx, False)
    
# 5. URETHRA SEGMENTATION
print("-------------------------------------------------------------------------------------------------------")
print("5. URETHRA SEGMENTATION (OUTPUT 2)...")
print("-------------------------------------------------------------------------------------------------------")

path_imgs2    = out_path+'/DistanceMaps/imagesTs_2'
path_results2 = out_path+'/Urethra/Urethra_DL_2'

!nnUNet_predict -i {path_imgs2} -o {path_results2} -t 108 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 --disable_tta

dir_ddbb_uretra = out_path+'/Urethra/Urethra_DL_2'
for i, file_out_urethra in enumerate(sorted(glob(dir_ddbb_uretra + r'/*.nii.gz'))):
    idx = file_out_urethra.split('/')[-1].split('_')[-1][:-7]
    file_urethra = dir_ddbb_uretra+'/MABUS_'+idx+'.nii.gz'
    print(file_urethra)
    out_urethra = sitk.ReadImage(file_urethra)
    # 5.1 Post-processing to Native space
    print("5.1 URETHRA SEGMENTATION POST-PROCESSING TO NATIVE SPACE ----------------------------------------------")
    postprocessing_native(out_urethra, out_path, str(idx),'urethra_DL_2', None, False)
    # 5.2 Exportation to Dicom-RT
    print("5.2 URETHRA SEGMENTATION EXPORTATION TO DICOM-RT ------------------------------------------------------")
    ct_file  = out_path+'/Native/ID_'+str(idx)+'/ID_'+str(idx)+'_voi.nii.gz'
    seg_file = out_path+'/Native/ID_'+str(idx)+'/ID_'+str(idx)+'_urethra.nii.gz'
    export2dicomRT(ct_file, seg_file, out_path+'/Native/DICOM/ID_'+str(idx), 'urethra')

print("--- Total Execution Time: %s seconds ---" % (time.time() - start_ini))
