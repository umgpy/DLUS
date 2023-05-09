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


def run_voi_extraction(checkpoint_path, out_path, dir_ddbb, use_manual_OARs=False):
    metadata=[]
    for i, path in enumerate(os.listdir(dir_ddbb)):
        if '.ipynb_checkpoints' not in path:
            idx = path.split('_')[1]
            print('\nProcessing case: ', idx)
            try:
                urethra_path = os.path.join(out_path, 'Urethra', 'GT_VOI', path.split('_')[0]+'_'+path.split('_')[1]+'.nii.gz')
                prostate_path = os.path.join(out_path, 'GTs', idx, 'mask_Prostate.nii.gz')
                if use_manual_OARs:
                    VOI_path = os.path.join(out_path, 'mVOIs', 'imagesTs', path)
                    VOI_path_labels = os.path.join(out_path, 'OARs', 'manual', path.split('_')[0]+'_'+path.split('_')[1]+'.nii.gz')
                    if os.path.exists(prostate_path):
                        data_idx = voi_extraction_manual(idx, os.path.join(dir_ddbb, path), prostate_path, VOI_path, VOI_path_labels, urethra_path)
                    else:
                        data_idx = voi_extraction(idx, os.path.join(dir_ddbb, path), VOI_path, checkpoint_path, prostate_path, urethra_path)
                    metadata_file = os.path.join(out_path, 'mVOIs', 'metadata.csv')
                else:
                    VOI_path = os.path.join(out_path, 'VOIs', 'imagesTs', path)
                    data_idx = voi_extraction(idx, os.path.join(dir_ddbb, path), VOI_path, checkpoint_path, prostate_path, urethra_path)
                    metadata_file = os.path.join(out_path, 'VOIs', 'metadata.csv')
                metadata.append(data_idx)
            except:
                print('--------------------- ERROR PROCESSING THIS CASE ------------------------------')
    df_meta = pd.DataFrame(np.array(metadata).squeeze(axis=2), columns=['idx', 'x0','y0','z0', 'res_x0','res_y0','res_z0', 'dim_x0','dim_y0','dim_z0', 'xVOI_res','yVOI_res','zVOI_res', 'res_xVOI_res','res_yVOI_res','res_zVOI_res', 'dim_xVOI_res','dim_yVOI_res','dim_zVOI_res','xoff1', 'xoff2', 'yoff1', 'yoff2', 'zoff1', 'zoff2'])
    df_meta.to_csv(metadata_file)
    df_meta
    
    
def voi_extraction(idx:str, img_path, VOI_path, checkpoint_path, prostate_path, urethra_path):
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
    
    # In case the uretra is available:
    try:
        urethra_path_GT = os.path.join('/'.join(prostate_path.split('/')[:-1]), 'mask_Urethra.nii.gz')
        urethra_gt = sitk.ReadImage(urethra_path_GT)
        urethra_gt_res = resample_volume(urethra_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
        urethra_gt = np.where(sitk.GetArrayFromImage(urethra_gt_res)==255, 1, 0)
        VOI_urethra, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(urethra_gt, img_x, img_y, img_z, x_cent, y_cent, z_cent)
        VOI_urethra = sitk.GetImageFromArray(VOI_urethra)
        VOI_urethra.CopyInformation(VOI_img)
        sitk.WriteImage(VOI_urethra, urethra_path, True)
    except: pass

    # Save meta-data
    data_idx = np.zeros((25,1))
    try: data_idx[0] = idx
    except: data_idx[0] = int(''.join([str(s) for s in idx if s.isdigit()]))
    data_idx[1:4], data_idx[4:7], data_idx[7:10] = np.array(origin)[:,np.newaxis], np.array(resolution)[:,np.newaxis], np.array(original_size)[:,np.newaxis]
    data_idx[10:13], data_idx[13:16], data_idx[16:19] = np.array(new_origin)[:,np.newaxis], np.array(VOI_img.GetSpacing())[:,np.newaxis], np.array(VOI_img.GetSize())[:,np.newaxis]
    data_idx[19:25] = np.array([xoff1, xoff2, yoff1, yoff2, zoff1, zoff2])[:,np.newaxis]
    
    return data_idx                      


def voi_extraction_manual(idx:str, img_path, prostate_path, VOI_path, VOI_path_labels, urethra_path):
    """
    Calls the Localization Network (LocalizationNet), a 3DUNet with input size 128x128x128, and loads the pretrained weights. This network creates a VOI of size 224x224x224 from the original image centered on the prostate.
    Parameters:
        idx : case id
        img_path : directory containing all of the images (CT/MR) NIfTI files
        prostate_path : path of the GT prostate to generate the VOI
        VOI_path : path to save the VOI in NIfTI format
        VOI_path_labels : path to save the GT OAR masks in the VOI in NIfTI format
        urethra_path : path to save the GT VOI urethra mask in NIfTI format
    Returns:
        data_idx : metadata of VOI extraction
    """
    
    # Set the centroid of the prostate as the center of the VOI with fixed sizes of (img_x x img_y x img_z) voxels
    img_x, img_y, img_z = 224, 224, 224 ### DO NOT TOUCH. VOI size required for OAR Segmentation Network
    # Load labelmap NIfTI file
    prostate_gt       = sitk.ReadImage(prostate_path)
    img    = sitk.ReadImage(img_path)
    origin, resolution = prostate_gt.GetOrigin(), prostate_gt.GetSpacing()
    original_size = (prostate_gt.GetSize()[0], prostate_gt.GetSize()[1], prostate_gt.GetSize()[2])
    # Resample the labelmap to a spatial resolution of (1x1x1) mm
    prostate_gt_res = resample_volume(prostate_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
    img_resample    = resample_volume(img, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1])
    origin_lmap_res, resolution_lmap_res, size_lmap_res = prostate_gt_res.GetOrigin(), prostate_gt_res.GetSpacing(), prostate_gt_res.GetSize()
    prostate_gt_res_np = sitk.GetArrayFromImage(prostate_gt_res).transpose(1,2,0)

    # Compute the centroid of the largest connected component of the prostate
    regions, r_area   = regionprops(label(prostate_gt_res_np)), 0
    for i, reg in enumerate(regions):
        if reg.area>r_area:
            r_area = reg.area
            x_cent, y_cent, z_cent = reg.centroid
    print('x:', x_cent, 'y:', y_cent, 'z:', z_cent)
    
    # Resample other available OAR masks
    bladder_path = os.path.join('/'.join(prostate_path.split('/')[:-1]), 'mask_Bladder.nii.gz')
    rectum_path = os.path.join('/'.join(prostate_path.split('/')[:-1]), 'mask_Rectum.nii.gz')
    semves_path = os.path.join('/'.join(prostate_path.split('/')[:-1]), 'mask_SeminalVesicles.nii.gz')
    if os.path.exists(bladder_path):
        bladder_gt = sitk.ReadImage(bladder_path)
        bladder_gt_res = resample_volume(bladder_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
    if os.path.exists(rectum_path):
        rectum_gt = sitk.ReadImage(rectum_path)
        rectum_gt_res = resample_volume(rectum_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
    if os.path.exists(semves_path):
        semvesicles_gt = sitk.ReadImage(semves_path)
        semvesicles_gt_res = resample_volume(semvesicles_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])

    # EXTRACT VOI----------------------------------------------------------------------------------------------------------------
    VOI_img, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(sitk.GetArrayFromImage(img_resample), img_x, img_y, img_z, x_cent, y_cent, z_cent)
    VOI_img = sitk.GetImageFromArray(VOI_img)
    new_origin = (yoff1*(resolution_lmap_res[0])+origin_lmap_res[0], xoff1*(resolution_lmap_res[1])+origin_lmap_res[1], zoff1*(resolution_lmap_res[2])+origin_lmap_res[2])
    VOI_img.SetOrigin((0,0,0))
    VOI_img.SetSpacing(img_resample.GetSpacing())
    VOI_img.SetDirection(img_resample.GetDirection())

    print('Saving... '+ VOI_path)
    sitk.WriteImage(VOI_img, VOI_path, True)  

    # EXTRACT VOI FOR AVAILABLE OAR AND JOIN INTO LABELMAP
    labelmap_gt = np.where(sitk.GetArrayFromImage(prostate_gt_res)==255, 3, 0)
    if os.path.exists(rectum_path): labelmap_gt = np.where(sitk.GetArrayFromImage(rectum_gt_res)==255, 1, labelmap_gt)
    if os.path.exists(bladder_path): labelmap_gt = np.where(sitk.GetArrayFromImage(bladder_gt_res)==255, 2, labelmap_gt)
    if os.path.exists(semves_path): labelmap_gt = np.where(sitk.GetArrayFromImage(semvesicles_gt_res)==255, 4, labelmap_gt)
        
    VOI_labelmap, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(labelmap_gt, img_x, img_y, img_z, x_cent, y_cent, z_cent)
    VOI_labelmap = sitk.GetImageFromArray(VOI_labelmap)
    VOI_labelmap.CopyInformation(VOI_img)
    sitk.WriteImage(VOI_labelmap, VOI_path_labels, True)
    
    # In case the uretra is available:
    try:
        urethra_path_GT = os.path.join('/'.join(prostate_path.split('/')[:-1]), 'mask_Urethra.nii.gz')
        urethra_gt = sitk.ReadImage(urethra_path_GT)
        urethra_gt_res = resample_volume(urethra_gt, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
        urethra_gt = np.where(sitk.GetArrayFromImage(urethra_gt_res)==255, 1, 0)
        VOI_urethra, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(urethra_gt, img_x, img_y, img_z, x_cent, y_cent, z_cent)
        VOI_urethra = sitk.GetImageFromArray(VOI_urethra)
        VOI_urethra.CopyInformation(VOI_img)
        sitk.WriteImage(VOI_urethra, urethra_path, True)
    except: pass
    
    # Save meta-data
    data_idx = np.zeros((25,1))
    try: data_idx[0] = idx
    except: data_idx[0] = int(''.join([str(s) for s in idx if s.isdigit()]))
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
