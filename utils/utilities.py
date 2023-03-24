import os
import shutil
import numpy as np
import SimpleITK as sitk
import warnings
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

def check_if_exist(folder_path, create = True):
    """
    Check if a folder exists, create if not (by default)
    """
    if os.path.exists(folder_path):
        print(folder_path, '.. exists')
    else:
        if create:
            os.makedirs(folder_path)
            print(folder_path, '.. created')
        else:
            warnings.warn("%s does NOT exist!!" %folder_path)
            
def get_directory_paths(dirName, only_names = False):
    list_paths = []
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if only_names:
                list_paths.append(file)
            else:
                list_paths.append(os.path.join(dirName, file))
        list_paths.sort()
    return list_paths

def get_image(patient_fold, data_path, out_path, modality:str='CT'):
    """
    Common function to get image path depending on modality. Used in functions dicom_to_nifti & nifti_data
    Outputs:
        path_im  : path of input image, located in data_path > modality
        path_nii : directory for output image 
        file_nii : path of output image in NIfTI format
    """
    ddbb_id = data_path.split('/')[-1]
    # A single modality load at a time
    if modality == 'CT':
        path_nii = out_path+'/CTs/'
        file_nii = os.path.join(path_nii, ddbb_id+'_'+patient_fold+'_0000.nii.gz')
        print('----- CT Image -----')
        path_im = os.path.join(data_path, patient_fold, 'CT')

    elif modality == 'MR':
        path_nii = out_path+'/MRs/'
        file_nii = os.path.join(path_nii, ddbb_id+'_'+patient_fold+'_0001.nii.gz')
        print('----- MR Image -----')
        path_im = os.path.join(data_path, patient_fold, 'MR')
    
    path_segs = os.path.join(data_path, patient_fold, modality+'_mOAR')
    
    return path_im, path_nii, file_nii, path_segs

def dicom_to_nifti(data_path, out_path, modality:str='CT'):
    """
    Function to transform a DICOM image to NIfTI format.
    """ 
    for i, patient_fold in enumerate(os.listdir(data_path)):
        # Remove any individual file not located in a folder
        if os.path.isfile(patient_fold) or '.ipynb_checkpoints' in patient_fold:
            continue
            print('Avoiding the file: ', patient_fold)
        
        print('\nProcessing case: ', patient_fold)
        path_im, path_nii, file_nii, path_segs = get_image(patient_fold, data_path, out_path, modality)        
        
        print(path_im)
        reader = sitk.ImageSeriesReader()
        dicomReader = reader.GetGDCMSeriesFileNames(path_im)
        reader.SetFileNames(dicomReader)
        reader.MetaDataDictionaryArrayUpdateOn() # Configure the reader to load
        reader.LoadPrivateTagsOn()               # all of the DICOM tags
        image = reader.Execute()
        print(image.GetOrigin())
        print(image.GetSpacing())
        print(image.GetSize())

        # Save as NIfTI
        print('Saving... '+ file_nii)
        if not os.path.exists(path_nii):
            os.makedirs(path_nii)
        sitk.WriteImage(image, file_nii, True)
        
        # Segmentation ###############################
        if os.path.exists(path_segs):
            print('----- Segmentation -----')    
            path_segs = get_directory_paths(path_segs)
            path_seg = path_segs[0]
            print(list_rt_structs(path_seg))
            
            # Possible names for OARs structures (add new names if necessary)
            rectum_names   = ['Rectum', 'rectum', 'Retto', 'retto', 'paroirectale', 'RECTUM', 'paroi rectale']            # Rectum
            bladder_names  = ['Bladder', 'bladder', 'Vescica', 'vescica', 'Vessie', 'vessie', 'paroivesicale', 'VESSIE', 'vesssie']  # Bladder
            prostate_names = ['Prostate', 'prostate', 'prosate', 'proistate', 'prosatet', 'prosatte',  'prost', 'prosta']   # Prostate
            SV_names       = ['SV', 'sv' , 'SeminalVesicles', 'seminal_vesicles', 'vseminales']                         # Seminal Vesicles
            # Convert dicom-RTSTRUCT files to nifti
            rectumName   = list(set(list_rt_structs(path_seg)) & set(rectum_names))[0]
            bladderName  = list(set(list_rt_structs(path_seg)) & set(bladder_names))[0]
            prostateName = list(set(list_rt_structs(path_seg)) & set(prostate_names))[0]
            OARstructs_list = [rectumName, bladderName, prostateName] # List of structures to convert
            try:
                svName       = list(set(list_rt_structs(path_seg)) & set(SV_names))[0]
                OARstructs_list.append(svName)
            except:
                print('No seminal vesicles in manual OARs segmentations')
            print(OARstructs_list)

            # Save as NIfTI
            path_file_nii = os.path.join(out_path, 'GTs', patient_fold)
            if not os.path.exists(path_file_nii):
                os.makedirs(path_file_nii)
            dcmrtstruct2nii(path_seg, path_im, path_file_nii, structures=OARstructs_list)
            
            for i, OAR in enumerate(OARstructs_list):
                file_img   = path_file_nii + '/mask_'+OAR+'.nii.gz'
                seg   = sitk.ReadImage(file_img)
                segg   = sitk.GetArrayFromImage((seg==255))
                if i==0: labelMap_np = sitk.GetArrayFromImage(seg)
                labelMap_np[tuple([segg==1])]   = i+1
            
            labelMap = sitk.GetImageFromArray(labelMap_np)
            labelMap.CopyInformation(sitk.ReadImage(path_file_nii + '/image.nii.gz'))
            print('Labels: ', np.unique(labelMap_np))
            file_seg_name = path_file_nii+'/labelMap.nii.gz'
            print('Saving... '+ file_seg_name)
            sitk.WriteImage(labelMap, file_seg_name, True)
                
        
def nifti_data(data_path, out_path, modality:str='CT'):
    """
    Function to copy NIfTI image to out_path in appropriate format.
    """ 
    ddbb_id = data_path.split('/')[-1]
    for i, patient_fold in enumerate(os.listdir(data_path)):
        # Remove any individual file not located in a folder
        if os.path.isfile(patient_fold) or '.ipynb_checkpoints' in patient_fold:
            continue
            print('Avoiding the file: ', patient_fold)
        
        print('\nProcessing case: ', patient_fold)
        path_im, path_nii, file_nii = get_image(patient_fold, data_path, out_path, modality)
        #Check if image already loaded
        if os.path.exists(file_nii):
            print(file_nii, '     already loaded')
            continue        
        path_im = get_directory_paths(path_im)[0]
        print(path_im)
        image = sitk.ReadImage(path_im)
        print(image.GetOrigin())
        print(image.GetSpacing())
        print(image.GetSize())

        # Save as NIfTI
        print('Saving... '+ file_nii)
        if not os.path.exists(path_nii):
            os.makedirs(path_nii)
        shutil.copyfile(path_im, file_nii)
        
def resample_volume(img_data, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1]):
    original_spacing = img_data.GetSpacing()
    original_size = img_data.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(img_data, new_size, sitk.Transform(), interpolator,
                         img_data.GetOrigin(), new_spacing, img_data.GetDirection(), 0,
                         img_data.GetPixelID())          
