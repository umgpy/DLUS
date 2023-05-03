import os
import numpy as np
from glob import glob
import SimpleITK as sitk
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import shutil

def load_data(data_path, out_path, mode:str, modality:str='CT'):
    """
    data_path : path to original data
    out_path : path to save processed images in nifti format
    mode : data type of original data --> 'nifti', 'dicom'
    """
    print('Loading data...')
    if mode == 'dicom':
        dicom_to_nifti(data_path, out_path)
    elif mode == 'nifti':
        nifti_data(data_path, out_path)
    else:
        print('ERROR IN DATA FORMAT. DATA NOT LOADED !')
        
        
def get_image(patient_fold, data_path, out_path):
    """
    Common function to get image paths. Used in functions dicom_to_nifti & nifti_data
    Outputs:
        path_im  : path of input image, located in data_path > img
        path_nii : directory for output image 
        file_nii : path of output image in NIfTI format
    """
    ddbb_id = data_path.split('/')[-1]
    path_nii = os.path.join(out_path, 'img')
    file_nii = os.path.join(path_nii, ddbb_id+'_'+patient_fold+'_0000.nii.gz')
    print('----- Image -----')
    path_im = os.path.join(data_path, patient_fold, 'img')
    if not os.path.exists(path_im): 
        print('%s does not exist. Check path !!!' %(path_im))
    path_segs = os.path.join(data_path, patient_fold, '_mOAR')
    
    return path_im, path_nii, file_nii, path_segs


def dicom_to_nifti(data_path, out_path):
    """
    Function to transform a DICOM image to NIfTI format.
    """ 
    for i, patient_fold in enumerate(os.listdir(data_path)):
        # Remove any individual file not located in a folder
        if os.path.isfile(patient_fold) or '.ipynb_checkpoints' in patient_fold:
            continue
            print('Avoiding the file: ', patient_fold)
        
        print('\nProcessing case: ', patient_fold)
        path_im, path_nii, file_nii, path_segs = get_image(patient_fold, data_path, out_path)        
        
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
            
            try:

                # Possible names for OARs structures (add new names if necessary)
                rectum_names   = ['rectum', 'retto', 'paroirectale', 'paroi rectale', 'rectum_ext']  # Rectum
                rectum_names += [a+'_ct' for a in rectum_names] + [a+'_irm' for a in rectum_names] + [a+'_irm_abr' for a in rectum_names]
                bladder_names  = ['bladder', 'vescica', 'vessie', 'paroivesicale', 'vesssie', 'vessie_ext']  # Bladder
                bladder_names += [a+'_ct' for a in bladder_names]+ [a+'_irm' for a in bladder_names] + [a+'_irm_abr' for a in bladder_names]
                prostate_names = ['prostate', 'prosate', 'proistate', 'prosatet', 'prosatte',  'prost', 'prosta']  # Prostate
                prostate_names += [a+'_ct' for a in prostate_names] + [a+'_irm' for a in prostate_names] + [a+'_irm_abr' for a in prostate_names]
                SV_names       = ['sv' , 'seminalvesicles', 'seminal_vesicles', 'vseminales', 'vs', 'vesicules_seminales']  # Seminal Vesicles
                SV_names += [a+'_ct' for a in SV_names] + [a+'_irm' for a in SV_names] + [a+'_irm_abr' for a in SV_names]

                # Convert dicom-RTSTRUCT files to nifti
                listt = list_rt_structs(path_seg)
                rectumName   = listt[[i for i, _ in enumerate(listt) if _.casefold()==(list(set([l.casefold() for l in listt]) & set(rectum_names))[0])][0]]
                bladderName  = listt[[i for i, _ in enumerate(listt) if _.casefold()==(list(set([l.casefold() for l in listt]) & set(bladder_names))[0])][0]]
                prostateName = listt[[i for i, _ in enumerate(listt) if _.casefold()==(list(set([l.casefold() for l in listt]) & set(prostate_names))[0])][0]]
                OARstructs_list = [rectumName, bladderName, prostateName] # List of structures to convert
                try:
                    svName       = listt[[i for i, _ in enumerate(listt) if _.casefold()==(list(set([l.casefold() for l in listt]) & set(SV_names))[0])][0]]
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
            
            except:
                print('\n --------------------------- ERROR WHEN LOADING THIS CASE mOAR SEGMENTATIONS --------------------------------- \n')
                continue
                
        
def nifti_data(data_path, out_path):
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
        path_im, path_nii, file_nii = get_image(patient_fold, data_path, out_path)
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
