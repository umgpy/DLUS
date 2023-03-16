import SimpleITK as sitk
import pandas as pd
import numpy as np
from glob import glob
import shutil
import os

from LocalizationNet.net_3DUnet import unet3d
from tensorflow.keras.optimizers import Adam
from LocalizationNet.utils import get_weighted_sparse_categorical_crossentropy, dice_coefficient

from skimage.measure import regionprops, label
from skimage.transform import resize

def createdatatensor(ct_img, img_x, img_y, img_z, x_cent, y_cent, z_cent):
    
    # Set the centroid of the prostate as the center of the VOI with fixed sizes of (img_x x img_y x img_z) voxels
    x = np.ndarray((img_x, img_y, img_z), dtype=np.float32)
        
    xoff1 = int(x_cent) - int(img_x/2)
    xoff2 = int(x_cent) + int(img_x/2)
    yoff1 = int(y_cent) - int(img_y/2)
    yoff2 = int(y_cent) + int(img_y/2)
    zoff1 = int(z_cent) - int(img_z/2)
    zoff2 = int(z_cent) + int(img_z/2)

    if xoff2 > ct_img.shape[1]:
        xoff1, xoff2 = (ct_img.shape[1]-img_x), ct_img.shape[1]
    if xoff1 < 0:
        xoff1, xoff2 = 0, img_x
    if yoff2 > ct_img.shape[2]:
        yoff1, yoff2 = (ct_img.shape[2]-img_y), ct_img.shape[2]
    if yoff1 < 0:
        yoff1, yoff2 = 0, img_y
    if zoff2 > ct_img.shape[0]:
        zoff1, zoff2 = (ct_img.shape[0]-img_z), ct_img.shape[0]
    if zoff1 < 0:
        zoff1, zoff2 = 0, img_z
        
    print('x_offset',xoff1,':',xoff2, 'y_offset',yoff1,':',yoff2, 'z_offset',zoff1,':',zoff2)
    
    print(ct_img.shape)
    x = ct_img[zoff1:zoff2,xoff1:xoff2,yoff1:yoff2]

    return x, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2

def resample_volume(img_data, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1]):
    original_spacing = img_data.GetSpacing()
    original_size = img_data.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(img_data, new_size, sitk.Transform(), interpolator,
                         img_data.GetOrigin(), new_spacing, img_data.GetDirection(), 0,
                         img_data.GetPixelID())

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
            data_idx[0] = idx
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
    ##print(df_meta)
    
def voi_extraction(idx, data_path, file_img_name, checkpoint_path):
    
    # Localization Network
    model = unet3d((128,128,128,1))
    model.load_weights(checkpoint_path) # Loads the weights
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss=get_weighted_sparse_categorical_crossentropy(weights=[0.1, 1]), optimizer=optimizer, metrics=['accuracy', dice_coefficient])

    #print("-------------------------------------------------------------------------------------------------------")
    #print("INITIALIZING VARIABLES")
    #print("-------------------------------------------------------------------------------------------------------")
    img_x, img_y, img_z = 224, 224, 224

    # Obtain predictions and compute the centroid in the original image size
    print('Processing case: ', idx)
    data_idx = np.zeros((25,1))

    # LOAD DATA-----------------------------------------------------------------------------------------------------------------
    # CT scan
    file_ct   = data_path+'/IGRT_'+idx+'_0000.nii.gz'
    print('Loading CT scan >> ', file_ct)
    ct_img    = sitk.ReadImage(file_ct)
    origin_ct, resolution_ct = ct_img.GetOrigin(), ct_img.GetSpacing()

    # PREPROCESSING-------------------------------------------------------------------------------------------------------------
    # Resize images to a size of 128x128x128 voxels
    out_size = (128, 128, 128)
    ct_resized = resize(sitk.GetArrayFromImage(ct_img), out_size)[np.newaxis,...]
    ct_resized = np.transpose(ct_resized, (0, 2, 3, 1))
    # Data Normalization: rescaling images to have mean zero and unit variance
    stats = [-3.417609237173173e-07, 1.9904431281376725e-07]
    ct_resized = (ct_resized - stats[0])/stats[1]
    print('Ct resized ',ct_resized.shape)

    # COARSE SEGMENTATION PREDICTION--------------------------------------------------------------------------------------------
    pred_test = model.predict(ct_resized)
    # Hard segmentation map: we assign the most likely class for each pixel.
    pred_lab  = np.argmax(pred_test, axis=-1).squeeze()

    # Resize Coarse Segmentation to the original image size
    original_size = (ct_img.GetSize()[0], ct_img.GetSize()[1], ct_img.GetSize()[2])
    pred_lab_resized = np.round(resize(pred_lab, original_size, preserve_range=True))
    pred_lab_resized = pred_lab_resized.transpose(2,0,1)
    pred_lab_resized = sitk.GetImageFromArray(pred_lab_resized)
    pred_lab_resized.CopyInformation(ct_img)

    # Resample the CT scan, labelmap and Coarse Segmentation a spatial resolution of (1x1x1) mm
    ct_resample       = resample_volume(ct_img, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1])
    pred_lab_resample = resample_volume(pred_lab_resized, interpolator = sitk.sitkNearestNeighbor, new_spacing = [1, 1, 1])
    print('Origin CT resample' , ct_resample.GetOrigin() , 'Origin pred resample' , pred_lab_resample.GetOrigin())
    print('Spacing CT resample', ct_resample.GetSpacing(), 'Spacing pred resample', pred_lab_resample.GetSpacing())
    print('Size CT resample'   , ct_resample.GetSize()   , 'Size pred resample'   , pred_lab_resample.GetSize())
    origin_ct_res, resolution_ct_res = ct_resample.GetOrigin(), ct_resample.GetSpacing()

    # Compute the centroid of the largest connected component
    pred_resample_np = sitk.GetArrayFromImage(pred_lab_resample).transpose(1,2,0)
    print(pred_resample_np.shape)
    regions, r_area   = regionprops(label(pred_resample_np)), 0
    for i, reg in enumerate(regions):
        print(reg.centroid)
        print(reg.area)
        if reg.area>r_area:
            r_area = reg.area
            x_cent, y_cent, z_cent = reg.centroid
            print('id:', i, 'x:', x_cent, 'y:', y_cent, 'z:', z_cent)

    # EXTRACT VOI----------------------------------------------------------------------------------------------------------------
    VOI_ct, xoff1, xoff2, yoff1, yoff2, zoff1, zoff2 = createdatatensor(sitk.GetArrayFromImage(ct_resample), img_x, img_y, img_z, x_cent, y_cent, z_cent)
    VOI_ct = sitk.GetImageFromArray(VOI_ct)
    new_origin = (yoff1*(resolution_ct_res[0])+origin_ct_res[0], xoff1*(resolution_ct_res[1])+origin_ct_res[1], zoff1*(resolution_ct_res[2])+origin_ct_res[2])
    VOI_ct.SetOrigin((0,0,0))
    VOI_ct.SetSpacing(ct_resample.GetSpacing())
    VOI_ct.SetDirection(ct_resample.GetDirection())

    print('Saving... '+ file_img_name)
    sitk.WriteImage(VOI_ct, file_img_name, True)

    # Save meta-data
    data_idx[0] = idx
    data_idx[1:4], data_idx[4:7], data_idx[7:10] = np.array(origin_ct)[:,np.newaxis], np.array(resolution_ct)[:,np.newaxis], np.array(original_size)[:,np.newaxis]
    data_idx[10:13], data_idx[13:16], data_idx[16:19] = np.array(new_origin)[:,np.newaxis], np.array(VOI_ct.GetSpacing())[:,np.newaxis], np.array(VOI_ct.GetSize())[:,np.newaxis]
    data_idx[19:25] = np.array([xoff1, xoff2, yoff1, yoff2, zoff1, zoff2])[:,np.newaxis]
    
    return data_idx
