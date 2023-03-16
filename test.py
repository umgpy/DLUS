# Imports
import os
import numpy as np
import SimpleITK as sitk
import multiprocessing
from glob import glob
import pandas as pd
import time

from utils.utilities import load_data, check_if_exist
from utils.voi_extraction import voi_extraction
from utils.dm_computation import dm_computation
from utils.postprocessing import postprocessing_native, export2dicomRT

# Set up paths
ddbb      = 'UrCTIRMRennesDICOM'
data_path = '/home/igt/Projects/PerPlanRT/temp_data/'+ddbb
out_path  = '/home/igt/Projects/PerPlanRT/FrameworkSegmentation_batch/Output/'+ddbb
check_if_exist(out_path)
mode      = 'dicom' # or nifti
model     = 'FR_model' #'Mixed_model'



###############################################################################################################

start_ini = time.time()
# 1. Load patient (Dicom or Nifti image)
print("-------------------------------------------------------------------------------------------------------")
print("1. LOADING PATIENT...")
print("-------------------------------------------------------------------------------------------------------")
load_data(data_path, out_path, mode)

# 2. Voi Extraction: Localization Network + Crop using the centroid of the coarse segmentation
print("-------------------------------------------------------------------------------------------------------")
print("2. VOI EXTRACTION...")
print("-------------------------------------------------------------------------------------------------------")
checkpoint_path = os.getcwd()+'/LocalizationNet/pretrain_model_weights.best.hdf5'
dir_ddbb_ct     = out_path+'/CTs'
check_if_exist(out_path+'/VOIs/imagesTs')

def run_voi_extraction(checkpoint_path, dir_ddbb_ct):
    metadata=[]
    for i, path in enumerate(os.listdir(dir_ddbb_ct)):
        idx = path.split('_')[1]
        # Check if image already loaded
        file_img_name = out_path+'/VOIs/imagesTs/IGRT_'+idx+'_0000.nii.gz'
        data_idx = voi_extraction(idx, dir_ddbb_ct, file_img_name, checkpoint_path)
        metadata.append(data_idx)
    df_meta = pd.DataFrame(np.array(metadata).squeeze(), columns=['idx', 'x0','y0','z0', 'res_x0','res_y0','res_z0', 'dim_x0','dim_y0','dim_z0', 'xVOI_res','yVOI_res','zVOI_res', 'res_xVOI_res','res_yVOI_res','res_zVOI_res', 'dim_xVOI_res','dim_yVOI_res','dim_zVOI_res','xoff1', 'xoff2', 'yoff1', 'yoff2', 'zoff1', 'zoff2'])
    df_meta.to_csv(out_path + '/metadata.csv')
    df_meta

p = multiprocessing.Process(target=run_voi_extraction, args=(checkpoint_path,dir_ddbb_ct,))
p.start()
p.join()

# 3. OARs Segmentation: Fine Segmentation Network
print("-------------------------------------------------------------------------------------------------------")
print("3. OARs SEGMENTATION (OUTPUT 1)...")
print("-------------------------------------------------------------------------------------------------------")
# Environmnet Variables
data_path = os.getcwd()+'/SegmentationNet'
main_dir  = os.path.join(data_path,'nnUNet')
# Can be set in the the .bashrc file or exported in the terminal:
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed']  = os.path.join(main_dir,'nnUNet_preprocessed')
os.environ['RESULTS_FOLDER']       = os.path.join(main_dir,'nnUNet_trained_models')
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"

path_imgs1    = out_path+'/VOIs/imagesTs'
path_results1 = out_path+'/OARs/'+model

if model=='FR_model':
    !nnUNet_predict -i {path_imgs1} -o {path_results1} -t 112 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 --disable_tta
    
if model=='Mixed_model':
    !nnUNet_predict -i {path_imgs1} -o {path_results1} -t 113 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 --disable_tta
    
dir_ddbb_OARs = out_path+'/OARs/'+model
for i, file_out_OARs in enumerate(sorted(glob(dir_ddbb_OARs + r'/*.nii.gz'))):
    idx = file_out_OARs.split('/')[-1].split('_')[-1][:-7]
    file_OARs = dir_ddbb_OARs+'/IGRT_'+idx+'.nii.gz'
    print(file_OARs)
    out_OARs  = sitk.ReadImage(file_OARs)
    # 3.1 Post-processing to Native space
    print("3.1 OARs SEGMENTATION POST-PROCESSING TO NATIVE SPACE -------------------------------------------------")
    postprocessing_native(out_OARs, out_path, str(idx),'OARs', None, False)
    postprocessing_native(out_OARs, out_path, str(idx),'voi', path_imgs1, False)
    # 3.2 Exportation to Dicom-RT
    print("3.2 OARs SEGMENTATION EXPORTATION TO DICOM-RT ---------------------------------------------------------")
    ct_file  = out_path+'/Native/ID_'+str(idx)+'/ID_'+str(idx)+'_voi.nii.gz'
    seg_file = out_path+'/Native/ID_'+str(idx)+'/ID_'+str(idx)+'_OARs.nii.gz'
    export2dicomRT(ct_file, seg_file, out_path+'/Native/DICOM/ID_'+str(idx), 'OARs_DL')
        
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