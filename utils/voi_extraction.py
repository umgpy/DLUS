import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tensorflow.keras.optimizers import Adam
from skimage.measure import regionprops, label
from skimage.transform import resize

from networks.LocalizationNet.net_3DUnet import unet3d
from networks.LocalizationNet.utils import get_weighted_sparse_categorical_crossentropy, dice_coefficient
from .utilities import resample_volume


def run_voi_extraction(checkpoint_path, out_path, dir_ddbb):
    metadata=[]
    for i, path in enumerate(os.listdir(dir_ddbb)):
        if '.ipynb_checkpoints' not in path:
            idx = path.split('_')[1]
            print('\nProcessing case: ', idx)
            VOI_path = os.path.join(out_path, 'VOIs', 'imagesTs', path)
            data_idx = voi_extraction(idx, os.path.join(dir_ddbb, path), VOI_path, checkpoint_path)
            metadata.append(data_idx)
    df_meta = pd.DataFrame(np.array(metadata).squeeze(axis=2), columns=['idx', 'x0','y0','z0', 'res_x0','res_y0','res_z0', 'dim_x0','dim_y0','dim_z0', 'xVOI_res','yVOI_res','zVOI_res', 'res_xVOI_res','res_yVOI_res','res_zVOI_res', 'dim_xVOI_res','dim_yVOI_res','dim_zVOI_res','xoff1', 'xoff2', 'yoff1', 'yoff2', 'zoff1', 'zoff2'])
    df_meta.to_csv(os.path.join(out_path, 'VOIs', 'metadata.csv'))
    df_meta
    
    
def voi_extraction(idx:str, img_path, VOI_path, checkpoint_path):
    """
    Calls the Localization Network (LocalizationNet), a 3DUNet with input size 128x128x128, and loads the pretrained weights. This network creates a VOI of size 224x224x224 from the original image centered on the prostate.
    Parameters:
        idx : case id
        img_path : directory containing all of the images (CT/MR) NIfTI files
        VOI_path : path to save the VOI in NIfTI format
        checkpoint_path : path to LocalizationNet weights
    Returns:
        data_idx : metadata of VOI extraction
    """

    # LOAD DATA-----------------------------------------------------------------------------------------------------------------
    print('Loading scan >> ', img_path)
    img    = sitk.ReadImage(img_path)
    origin, resolution = img.GetOrigin(), img.GetSpacing()

    # PREPROCESSING-------------------------------------------------------------------------------------------------------------
    # Resize images to a size of 128x128x128 voxels to input into LocalizationNet
    out_size = (128, 128, 128)
    img_resized = resize(sitk.GetArrayFromImage(img), out_size)[np.newaxis,...]
    img_resized = np.transpose(img_resized, (0, 2, 3, 1))
    # Data Normalization: rescaling images to have mean zero and unit variance
    stats = [-3.417609237173173e-07, 1.9904431281376725e-07]
    img_resized = (img_resized - stats[0])/stats[1]
    print('Image resized ', img_resized.shape)

    # COARSE SEGMENTATION PREDICTION--------------------------------------------------------------------------------------------
    # LocalizationNet parameters
    model = unet3d((128,128,128,1))
    model.load_weights(checkpoint_path) # Loads the weights
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss=get_weighted_sparse_categorical_crossentropy(weights=[0.1, 1]), optimizer=optimizer, metrics=['accuracy', dice_coefficient])
    pred_test = model.predict(img_resized)
    # Hard segmentation map: we assign the most likely class for each pixel.
    pred_lab  = np.argmax(pred_test, axis=-1).squeeze()
    
    # Resize Coarse Segmentation to the original image size
    original_size = (img.GetSize()[0], img.GetSize()[1], img.GetSize()[2])
    pred_lab_resized = np.round(resize(pred_lab, original_size, preserve_range=True))
    pred_lab_resized = pred_lab_resized.transpose(2,0,1)
    pred_lab_resized = sitk.GetImageFromArray(pred_lab_resized)
    pred_lab_resized.CopyInformation(img)

    # Resample the scan, labelmap and Coarse Segmentation a spatial resolution of (1x1x1) mm
    img_resample       = resample_volume(img, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1])
    pred_lab_resample = resample_volume(pred_lab_resized, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
    print('Origin image resample' , img_resample.GetOrigin() , 'Origin pred resample' , pred_lab_resample.GetOrigin())
    print('Spacing image resample', img_resample.GetSpacing(), 'Spacing pred resample', pred_lab_resample.GetSpacing())
    print('Size image resample'   , img_resample.GetSize()   , 'Size pred resample'   , pred_lab_resample.GetSize())
    origin_img_res, resolution_img_res = img_resample.GetOrigin(), img_resample.GetSpacing()

    # Compute the centroid of the largest connected component
    pred_resample_np = sitk.GetArrayFromImage(pred_lab_resample).transpose(1,2,0)
    regions, r_area   = regionprops(label(pred_resample_np)), 0
    for i, reg in enumerate(regions):
        if reg.area>r_area:
            r_area = reg.area
            x_cent, y_cent, z_cent = reg.centroid
    print('x:', x_cent, 'y:', y_cent, 'z:', z_cent)

    # EXTRACT VOI----------------------------------------------------------------------------------------------------------------
    # Set the centroid of the prostate as the center of the VOI with fixed sizes of (img_x x img_y x img_z) voxels
    img_x, img_y, img_z = 224, 224, 224 ### DO NOT TOUCH. VOI size required for OAR Segmentation Network
    VOI_img, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(sitk.GetArrayFromImage(img_resample), img_x, img_y, img_z, x_cent, y_cent, z_cent)
    VOI_img = sitk.GetImageFromArray(VOI_img)
    new_origin = (yoff1*(resolution_img_res[0])+origin_img_res[0], xoff1*(resolution_img_res[1])+origin_img_res[1], zoff1*(resolution_img_res[2])+origin_img_res[2])
    VOI_img.SetOrigin((0,0,0))
    VOI_img.SetSpacing(img_resample.GetSpacing())
    VOI_img.SetDirection(img_resample.GetDirection())

    print('Saving... '+ VOI_path)
    sitk.WriteImage(VOI_img, VOI_path, True)

    # Save meta-data
    data_idx = np.zeros((25,1))
    data_idx[0] = str(idx)
    data_idx[1:4], data_idx[4:7], data_idx[7:10] = np.array(origin)[:,np.newaxis], np.array(resolution)[:,np.newaxis], np.array(original_size)[:,np.newaxis]
    data_idx[10:13], data_idx[13:16], data_idx[16:19] = np.array(new_origin)[:,np.newaxis], np.array(VOI_img.GetSpacing())[:,np.newaxis], np.array(VOI_img.GetSize())[:,np.newaxis]
    data_idx[19:25] = np.array([xoff1, xoff2, yoff1, yoff2, zoff1, zoff2])[:,np.newaxis]
    
    return data_idx                      
                         

def createdatatensor(img, img_x, img_y, img_z, x_cent, y_cent, z_cent):
    x = np.ndarray((img_x, img_y, img_z), dtype=np.float32)
        
    xoff1 = int(x_cent) - int(img_x/2)
    xoff2 = int(x_cent) + int(img_x/2)
    yoff1 = int(y_cent) - int(img_y/2)
    yoff2 = int(y_cent) + int(img_y/2)
    zoff1 = int(z_cent) - int(img_z/2)
    zoff2 = int(z_cent) + int(img_z/2)

    if xoff2 > img.shape[1]: xoff1, xoff2 = (img.shape[1]-img_x), img.shape[1]
    if xoff1 < 0: xoff1, xoff2 = 0, img_x
    if yoff2 > img.shape[2]: yoff1, yoff2 = (img.shape[2]-img_y), img.shape[2]
    if yoff1 < 0: yoff1, yoff2 = 0, img_y
    if zoff2 > img.shape[0]: zoff1, zoff2 = (img.shape[0]-img_z), img.shape[0]
    if zoff1 < 0: zoff1, zoff2 = 0, img_z
        
    print('x_offset',xoff1,':',xoff2, 'y_offset',yoff1,':',yoff2, 'z_offset',zoff1,':',zoff2)
    print(img.shape)
    x = img[zoff1:zoff2,xoff1:xoff2,yoff1:yoff2]
    
    return x, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2


# REVIEW ######################################################################################################        

##############################################################################
##############################################################################
##############################################################################
def voi_seg_extraction(out_path, urethra3 = False):
    
    metadata=[]
    img_x, img_y, img_z = 224, 224, 224
    
    for i, patient_fold in enumerate(os.listdir(out_path+'/GTs')):

        if 'ipynb_checkpoints' in patient_fold: continue
        if 'VOIs' in patient_fold: continue

        file_labelmap  = out_path+'/GTs/'+patient_fold+'/labelMap.nii.gz'
        file_urethra   = out_path+'/GTs/'+patient_fold+'/mask_Urethra.nii.gz'     #####
        
        if not os.path.exists(file_labelmap):
            print('MISSING LABELMAP for patient: ' + patient_fold)
            continue
        else:
            idx = patient_fold[2:] #[1:]
            print('\nProcessing case: ', idx)
            data_idx = np.zeros((25,1))
        
            labelmap       = sitk.ReadImage(file_labelmap)
            origin_lmap, resolution_lmap, size_lmap = labelmap.GetOrigin(), labelmap.GetSpacing(), labelmap.GetSize()
            prostate_gt    = (labelmap==3)
            # Resample the labelmap to a spatial resolution of (1x1x1) mm
            prostate_gt_res = resample_volume(prostate_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
            labelmap_res    = resample_volume(labelmap, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
            origin_lmap_res, resolution_lmap_res, size_lmap_res = prostate_gt_res.GetOrigin(), prostate_gt_res.GetSpacing(), prostate_gt_res.GetSize()
            prostate_gt_res_np = sitk.GetArrayFromImage(prostate_gt_res).transpose(1,2,0)
            print(prostate_gt_res_np.shape)

            # Compute the centroid of the largest connected component of the prostate
            regions, r_area   = regionprops(label(prostate_gt_res_np)), 0
            for i, reg in enumerate(regions):
                print(reg.centroid)
                print(reg.area)
                if reg.area>r_area:
                    r_area = reg.area
                    x_cent, y_cent, z_cent = reg.centroid
                    print('id:', i, 'x:', x_cent, 'y:', y_cent, 'z:', z_cent)

            # EXTRACT VOI----------------------------------------------------------------------------------------------------------------
            VOI_labelmap, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(sitk.GetArrayFromImage(labelmap_res), img_x, img_y, img_z, x_cent, y_cent, z_cent)
            ##VOI_labelmap = sitk.GetImageFromArray(VOI_labelmap)
            new_origin = (yoff1*(resolution_lmap_res[0])+origin_lmap_res[0], xoff1*(resolution_lmap_res[1])+origin_lmap_res[1], zoff1*(resolution_lmap_res[2])+origin_lmap_res[2])
            ##VOI_labelmap.SetOrigin((0,0,0))
            ##VOI_labelmap.SetSpacing(labelmap_res.GetSpacing())
            ##VOI_labelmap.SetDirection(labelmap_res.GetDirection())
            ##file_voi_name = out_path+'/VOIs/IGRT_'+idx+'.nii.gz'
            ##print('Saving... '+ file_voi_name)
            ##sitk.WriteImage(VOI_labelmap, file_voi_name, True)

            urethra_res    = resample_volume(sitk.ReadImage(file_urethra), interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1]) 
            VOI_urethra  = sitk.GetArrayFromImage(urethra_res)[zoff1:zoff2,xoff1:xoff2,yoff1:yoff2]  #####

            VOI_labelmap = sitk.GetImageFromArray(VOI_labelmap)
            VOI_urethra  = sitk.GetImageFromArray(VOI_urethra)  #####


            VOI_labelmap.SetOrigin((0,0,0))
            VOI_labelmap.SetSpacing(labelmap_res.GetSpacing())
            VOI_labelmap.SetDirection(labelmap_res.GetDirection())

            #####
            VOI_urethra.SetOrigin((0,0,0))
            VOI_urethra.SetSpacing(labelmap_res.GetSpacing())
            VOI_urethra.SetDirection(labelmap_res.GetDirection())
            #####

            sitk.WriteImage(VOI_labelmap, out_path+'/GTs/VOIs/labelsTs/IGRT_'+idx+'.nii.gz', True)
            sitk.WriteImage(VOI_urethra, out_path+'/GTs/VOIs/urethraTs/IGRT_'+idx+'.nii.gz', True)   #####

            if urethra3:
                file_urethra3  = out_path+'/GTs/'+patient_fold+'/mask_Urethra-3mm.nii.gz' #####
                urethra3_res   = resample_volume(sitk.ReadImage(file_urethra3), interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
                VOI_urethra3 = sitk.GetArrayFromImage(urethra3_res)[zoff1:zoff2,xoff1:xoff2,yoff1:yoff2] #####
                VOI_urethra3 = sitk.GetImageFromArray(VOI_urethra3) #####
                VOI_urethra3.SetOrigin((0,0,0))
                VOI_urethra3.SetSpacing(labelmap_res.GetSpacing())
                VOI_urethra3.SetDirection(labelmap_res.GetDirection())
                sitk.WriteImage(VOI_urethra3, out_path+'/GTs/VOIs/urethra3Ts/IGRT_'+idx+'.nii.gz', True) #####

            # Save meta-data
            data_idx[0] = str(idx)
            data_idx[1:4]  , data_idx[4:7]  , data_idx[7:10]  = np.array(origin_lmap)[:,np.newaxis], np.array(resolution_lmap)[:,np.newaxis], np.array(size_lmap)[:,np.newaxis]
            data_idx[10:13], data_idx[13:16], data_idx[16:19] = np.array(new_origin)[:,np.newaxis], np.array(VOI_labelmap.GetSpacing())[:,np.newaxis], np.array(VOI_labelmap.GetSize())[:,np.newaxis]
            data_idx[19:25] = np.array([xoff1, xoff2, yoff1, yoff2, zoff1, zoff2])[:,np.newaxis]

            metadata.append(data_idx)

    df_meta = pd.DataFrame(np.array(metadata).squeeze(), columns=['idx', 'x0','y0','z0', 'res_x0','res_y0','res_z0', 'dim_x0','dim_y0','dim_z0', 'xVOI_res','yVOI_res','zVOI_res', 'res_xVOI_res','res_yVOI_res','res_zVOI_res', 'dim_xVOI_res','dim_yVOI_res','dim_zVOI_res','xoff1', 'xoff2', 'yoff1', 'yoff2', 'zoff1', 'zoff2'])
    df_meta.to_csv(out_path + '/metadata_manual.csv')
    ##print(df_meta)

def voi_seg_extraction_MRI(out_path):
    
    metadata=[]
    img_x, img_y, img_z = 224, 224, 224
    
    for i, patient_fold in enumerate(os.listdir(out_path+'/GTs')):

        if 'ipynb_checkpoints' in patient_fold: continue
        if 'VOIs' in patient_fold: continue

        file_labelmap  = out_path+'/GTs/'+patient_fold+'/MRI_labelMap.nii.gz'
        file_urethra   = out_path+'/GTs/'+patient_fold+'/mask_Urethra.nii.gz'     #####
        
        if not os.path.exists(file_labelmap):
            print('MISSING LABELMAP for patient: ' + patient_fold)
            continue
        else:
            idx = patient_fold[2:] #[1:]
            print('\nProcessing case: ', idx)
            data_idx = np.zeros((25,1))
        
            labelmap       = sitk.ReadImage(file_labelmap)
            origin_lmap, resolution_lmap, size_lmap = labelmap.GetOrigin(), labelmap.GetSpacing(), labelmap.GetSize()
            prostate_gt    = (labelmap==3)
            # Resample the labelmap to a spatial resolution of (1x1x1) mm
            prostate_gt_res = resample_volume(prostate_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
            labelmap_res    = resample_volume(labelmap, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
            origin_lmap_res, resolution_lmap_res, size_lmap_res = prostate_gt_res.GetOrigin(), prostate_gt_res.GetSpacing(), prostate_gt_res.GetSize()
            prostate_gt_res_np = sitk.GetArrayFromImage(prostate_gt_res).transpose(1,2,0)
            print(prostate_gt_res_np.shape)

            # Compute the centroid of the largest connected component of the prostate
            regions, r_area   = regionprops(label(prostate_gt_res_np)), 0
            for i, reg in enumerate(regions):
                print(reg.centroid)
                print(reg.area)
                if reg.area>r_area:
                    r_area = reg.area
                    x_cent, y_cent, z_cent = reg.centroid
                    print('id:', i, 'x:', x_cent, 'y:', y_cent, 'z:', z_cent)

            # EXTRACT VOI----------------------------------------------------------------------------------------------------------------
            VOI_labelmap, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(sitk.GetArrayFromImage(labelmap_res), img_x, img_y, img_z, x_cent, y_cent, z_cent)
            new_origin = (yoff1*(resolution_lmap_res[0])+origin_lmap_res[0], xoff1*(resolution_lmap_res[1])+origin_lmap_res[1], zoff1*(resolution_lmap_res[2])+origin_lmap_res[2])

            urethra_res    = resample_volume(sitk.ReadImage(file_urethra), interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1]) 
            VOI_urethra  = sitk.GetArrayFromImage(urethra_res)[zoff1:zoff2,xoff1:xoff2,yoff1:yoff2]  #####

            VOI_labelmap = sitk.GetImageFromArray(VOI_labelmap)
            VOI_urethra  = sitk.GetImageFromArray(VOI_urethra)  #####


            VOI_labelmap.SetOrigin((0,0,0))
            VOI_labelmap.SetSpacing(labelmap_res.GetSpacing())
            VOI_labelmap.SetDirection(labelmap_res.GetDirection())

            #####
            VOI_urethra.SetOrigin((0,0,0))
            VOI_urethra.SetSpacing(labelmap_res.GetSpacing())
            VOI_urethra.SetDirection(labelmap_res.GetDirection())
            #####

            sitk.WriteImage(VOI_labelmap, out_path+'/GTs_MRI/VOIs/labelsTs/IGRT_'+idx+'.nii.gz', True)
            sitk.WriteImage(VOI_urethra, out_path+'/GTs_MRI/VOIs/urethraTs/IGRT_'+idx+'.nii.gz', True)   #####

            # Save meta-data
            data_idx[0] = idx
            data_idx[1:4]  , data_idx[4:7]  , data_idx[7:10]  = np.array(origin_lmap)[:,np.newaxis], np.array(resolution_lmap)[:,np.newaxis], np.array(size_lmap)[:,np.newaxis]
            data_idx[10:13], data_idx[13:16], data_idx[16:19] = np.array(new_origin)[:,np.newaxis], np.array(VOI_labelmap.GetSpacing())[:,np.newaxis], np.array(VOI_labelmap.GetSize())[:,np.newaxis]
            data_idx[19:25] = np.array([xoff1, xoff2, yoff1, yoff2, zoff1, zoff2])[:,np.newaxis]

            metadata.append(data_idx)

    df_meta = pd.DataFrame(np.array(metadata).squeeze(), columns=['idx', 'x0','y0','z0', 'res_x0','res_y0','res_z0', 'dim_x0','dim_y0','dim_z0', 'xVOI_res','yVOI_res','zVOI_res', 'res_xVOI_res','res_yVOI_res','res_zVOI_res', 'dim_xVOI_res','dim_yVOI_res','dim_zVOI_res','xoff1', 'xoff2', 'yoff1', 'yoff2', 'zoff1', 'zoff2'])
    df_meta.to_csv(out_path + '/metadata_MRI.csv')
