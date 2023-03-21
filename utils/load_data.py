import os
import numpy as np
from glob import glob
import SimpleITK as sitk
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
from .utilities import dicom_to_nifti, nifti_data

def load_data(data_path, out_path, mode:str, modality:str='CT'):
    """
    data_path : path to original data
    out_path : path to save processed images in nifti format
    mode : data type of original data --> 'nifti', 'dicom'
    modality : image modality --> 'CT', 'MR', [default: 'CT']
    """
    print('Loading data...')
    if mode == 'dicom':
        dicom_to_nifti(data_path, out_path, modality)
    elif mode == 'nifti':
        nifti_data(data_path, out_path, modality)
    else:
        print('ERROR IN DATA FORMAT. DATA NOT LOADED !')

        
        
        
# REVIEW #################################################################################################        
def load_seg_data(data_path, out_path):
    
    print('Loading data...')
    for i, patient_fold in enumerate(os.listdir(data_path)):
        
        try:
            idx = patient_fold[2:] #patient_fold[1:]
            print('\nProcessing case: ', idx)
            #--------------------------------------------------------------------#
            # CT Image
            #--------------------------------------------------------------------#    
            print('----- CT Image -----')        
            files_ct = os.path.join(data_path, patient_fold, 'CT')
            print('Loading CT... ', files_ct)
            #--------------------------------------------------------------------#
            # Segmentation
            #--------------------------------------------------------------------#
            print('----- Segmentation -----')          
            ## file_seg = glob(os.path.join(data_path, patient_fold)+'/rtss_*.dcm', recursive=True)[0]
            file_seg = glob(os.path.join(data_path, patient_fold)+'/RTStruct.dcm', recursive=True)[0]
            print('Loading segmentation... ', file_seg)
            print(list_rt_structs(file_seg))
        except:
            print('\n')
            print('*** The CT scan and/or the manual delineation are missing')
            print('\n')
            continue
            
        try:
            # Possible names for OARs structures (add new names if necessary)
            rectum_names   = ['Rectum', 'rectum', 'Retto', 'retto', 'paroirectale', 'RECTUM', 'paroi rectale']            # Rectum
            bladder_names  = ['Bladder', 'bladder', 'Vescica', 'vescica', 'Vessie', 'vessie', 'paroivesicale', 'VESSIE', 'vesssie']  # Bladder
            prostate_names = ['Prostate', 'prostate', 'prosate', 'proistate', 'prosatet', 'Prostate _CTV','CTV Prostate', 'prosatte', 'Prostate-CTV', 'prost', 'prosta', 'CTV prostate'] #'ctv','ctv1' # Prostate
            ##SV_names       = ['SV', 'sv' , 'SeminalVesicles', 'seminal_vesicles', 'vseminales']                         # Seminal Vesicles
            # Convert dicom-RTSTRUCT files to nifti
            rectumName   = list(set(list_rt_structs(file_seg)) & set(rectum_names))[0]
            bladderName  = list(set(list_rt_structs(file_seg)) & set(bladder_names))[0]
            prostateName = list(set(list_rt_structs(file_seg)) & set(prostate_names))[0]
            ##svName       = list(set(list_rt_structs(file_seg)) & set(SV_names))[0]
            OARstructs_list = [rectumName, bladderName, prostateName] # List of structures to convert
            ##OARstructs_list = [rectumName, bladderName, prostateName, svName] # List of structures to convert
            print(OARstructs_list)

            # Save as Nifty
            path_file_nii = out_path+'/GTs/'+patient_fold
            if not os.path.exists(path_file_nii):
                os.makedirs(path_file_nii)
            dcmrtstruct2nii(file_seg, files_ct, path_file_nii, structures=OARstructs_list)

            # Obtain labelmap
            file_img_rectum   = path_file_nii + '/mask_'+rectumName+'.nii.gz'
            file_img_bladder  = path_file_nii + '/mask_'+bladderName+'.nii.gz'
            file_img_prostate = path_file_nii + '/mask_'+prostateName+'.nii.gz'
            ##file_img_sv       = path_file_nii + '/mask_'+svName+'.nii.gz'

            seg_rectum   = sitk.ReadImage(file_img_rectum)
            seg_bladder  = sitk.ReadImage(file_img_bladder)
            seg_prostate = sitk.ReadImage(file_img_prostate)
            ##seg_vseminls = sitk.ReadImage(file_img_sv)
        except:
            print('\n')
            print('*** Some segmentation is missing')
            print('\n')
            continue

        rectum   = sitk.GetArrayFromImage((seg_rectum==255))
        bladder  = sitk.GetArrayFromImage((seg_bladder==255))
        prostate = sitk.GetArrayFromImage((seg_prostate==255))
        ##vseminls = sitk.GetArrayFromImage((seg_vseminls==255))

        labelMap_np = sitk.GetArrayFromImage(seg_prostate)
        labelMap_np[tuple([rectum==1])]   = 1
        labelMap_np[tuple([bladder==1])]  = 2
        labelMap_np[tuple([prostate==1])] = 3
        ##labelMap_np[tuple([vseminls==1])] = 4
        labelMap = sitk.GetImageFromArray(labelMap_np)
        labelMap.CopyInformation(seg_prostate)
        print('Labels: ', np.unique(labelMap_np))
        file_seg_name = path_file_nii+'/labelMap.nii.gz'
        print('Saving... '+ file_seg_name)
        sitk.WriteImage(labelMap, file_seg_name, True)
        print('---------------------------------------------------------------')