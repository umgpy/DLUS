import os, sys, time
import SimpleITK as sitk
import pandas as pd
import numpy as np
from utilities import check_if_exist
from skimage.measure import regionprops, label
from skimage.transform import resize
from rt_utils import RTStructBuilder

def resample_volume(img_data, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1]):
    original_spacing = img_data.GetSpacing()
    original_size = img_data.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(img_data, new_size, sitk.Transform(), interpolator,
                         img_data.GetOrigin(), new_spacing, img_data.GetDirection(), 0,
                         img_data.GetPixelID())

def postprocessing_reference_img(pred_seg, out_path, i, seg, file):
    
    if os.path.exists(file):
        file_pred_seg = file+'/IGRT_'+i+'_0000.nii.gz'
        pred_seg      = sitk.ReadImage(file_pred_seg)
    
    ref_origin  = np.array([0.,0.,0.])
    ref_spacing = [1.171875,1.171875,1.171875]
    ref_size    = [512, 512, 321]
    ref_center  = np.array([300.,300.,188.0859375])
    # Meatdata
    df_meta      = pd.read_csv(out_path + '/metadata.csv')
    df_meta_i = df_meta.loc[df_meta['idx'] == float(i)]
    idx       = np.array(df_meta_i['idx'])
    # Original image characteristics (Native space)
    origin_ct = df_meta_i['x0'], df_meta_i['y0'], df_meta_i['z0']
    res_ct    = df_meta_i['res_x0'], df_meta_i['res_y0'], df_meta_i['res_z0']
    dim_ct    = df_meta_i['dim_x0'], df_meta_i['dim_y0'], df_meta_i['dim_z0']
    # Resized image characteristics (Common space)
    origin_voi = df_meta_i['xVOI_res'], df_meta_i['yVOI_res'], df_meta_i['zVOI_res']
    res_voi    = df_meta_i['res_xVOI_res'], df_meta_i['res_yVOI_res'], df_meta_i['res_zVOI_res']
    dim_voi    = df_meta_i['dim_xVOI_res'], df_meta_i['dim_yVOI_res'], df_meta_i['dim_zVOI_res']
    xoff1, xoff2 = np.array(df_meta_i['xoff1']), np.array(df_meta_i['xoff2'])
    yoff1, yoff2 = np.array(df_meta_i['yoff1']), np.array(df_meta_i['yoff2'])
    zoff1, zoff2 = np.array(df_meta_i['zoff1']), np.array(df_meta_i['zoff2'])  
    
    print('Processing case:', i)             
    #----------------------------------------------------------------------------------------------------------------------#
    # CT Scan Original
    #----------------------------------------------------------------------------------------------------------------------#
    ct_file = out_path +'/CTs/IGRT_'+i+'_0000.nii.gz'
    ct_img  = sitk.ReadImage(ct_file)
    
    #----------------------------------------------------------------------------------------------------------------------#
    # Predicted segmentation
    #----------------------------------------------------------------------------------------------------------------------#              
    prediction    = np.zeros((ref_size[2], ref_size[1], ref_size[0]), dtype=np.int32)              
    pred_seg_np = sitk.GetArrayFromImage(pred_seg)
    prediction[int(zoff1):int(zoff2),int(xoff1):int(xoff2),int(yoff1):int(yoff2)] = pred_seg_np
    print('id', idx, 'x_offset',int(xoff1),':',int(xoff2), 'y_offset',int(yoff1),':',int(yoff2), 'z_offset',int(zoff1),':',int(zoff2))
    pred_img = sitk.GetImageFromArray(prediction)
    pred_img.CopyInformation(ct_file)
    
    # Reference image information
    dimension   = ct_img.GetDimension()
    print('Reference: Origin=',ref_origin,' Spacing=',ref_spacing,' Size=',ref_size,' Center=',ref_center)
    
    # Set Transformation, which maps from the ref_image to the current ct image.
    transform = sitk.AffineTransform(dimension)                         # Use affine transform with 3 dimensions
    transform.SetMatrix(ct_img.GetDirection())                          # Set the cosine direction matrix
    transform.SetTranslation(np.array(ct_img.GetOrigin()) - ref_origin) # Set the translation (mapping origins ref and ct image)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(ct_img.TransformContinuousIndexToPhysicalPoint(np.array(ct_img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - ref_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Obtain inverse transform
    inv_centered_transform = centered_transform.GetInverse()
    res_pred_img = sitk.Resample(pred_img, ct_img.GetSize(), inv_centered_transform, sitk.sitkNearestNeighbor, ct_img.GetOrigin(), ct_img.GetSpacing(), ct_img.GetDirection())

    print('Origin CT',ct_img.GetOrigin(),'Origin labelmap',pred_img.GetOrigin(),'Origin labelmap resample',res_pred_img.GetOrigin())
    print('Spacing CT', ct_img.GetSpacing(),'Spacing labelmap',pred_img.GetSpacing(),'Spacing labelmap resample',res_pred_img.GetSpacing())
    print('Size CT',ct_img.GetSize(),'Size labelmap',pred_img.GetSize(),'Size labelmap resample',res_pred_img.GetSize())
    
    out_path_native = out_path+'/Native/ID_'+i     
    check_if_exist(out_path_native)
    file_seg_name = out_path_native +'/ID_'+i+'_'+seg+'.nii.gz'
    print('Saving... '+ file_seg_name)
    sitk.WriteImage(res_pred_img, file_seg_name, True)
                  
def postprocessing_native(pred_seg, out_path, i, seg, file, manual):

    out_path_native = out_path+'/Native/ID_'+i
    file_seg_name = out_path_native +'/ID_'+i+'_'+seg+'.nii.gz'
    if os.path.exists(file_seg_name):
        print('Output in the native space for case '+i+' already exists')
        
    else:
        print('Obtaining output in the native space...')
    
        if file is not None:
            file_pred_seg = file+'/IGRT_'+i+'_0000.nii.gz'
            pred_seg      = sitk.ReadImage(file_pred_seg)

        # Meatdata
        if manual:
            df_meta   = pd.read_csv(out_path + '/metadata_manual.csv')
            df_meta_i = df_meta.loc[df_meta['idx'] == i]
        else:
            df_meta   = pd.read_csv(out_path + '/metadata.csv')
            df_meta_i = df_meta.loc[df_meta['idx'] == float(i)]
            
        idx       = np.array(df_meta_i['idx'])
        # Original image characteristics (Native space)
        origin_ct = [df_meta_i['x0'].values[0], df_meta_i['y0'].values[0], df_meta_i['z0'].values[0]]
        res_ct    = [df_meta_i['res_x0'].values[0],df_meta_i['res_y0'].values[0], df_meta_i['res_z0'].values[0]]
        dim_ct    = [df_meta_i['dim_x0'].values[0], df_meta_i['dim_y0'].values[0], df_meta_i['dim_z0'].values[0]]
        # Resized image characteristics (Common space)
        origin_voi = [df_meta_i['xVOI_res'].values[0], df_meta_i['yVOI_res'].values[0], df_meta_i['zVOI_res'].values[0]]
        res_voi    = [df_meta_i['res_xVOI_res'].values[0], df_meta_i['res_yVOI_res'].values[0], df_meta_i['res_zVOI_res'].values[0]]
        dim_voi    = [df_meta_i['dim_xVOI_res'].values[0], df_meta_i['dim_yVOI_res'].values[0], df_meta_i['dim_zVOI_res'].values[0]]
        xoff1, xoff2 = np.array(df_meta_i['xoff1']), np.array(df_meta_i['xoff2'])
        yoff1, yoff2 = np.array(df_meta_i['yoff1']), np.array(df_meta_i['yoff2'])
        zoff1, zoff2 = np.array(df_meta_i['zoff1']), np.array(df_meta_i['zoff2'])

        print('Processing case:', i)
        #----------------------------------------------------------------------------------------------------------------------#
        # CT Scan Original
        #----------------------------------------------------------------------------------------------------------------------#
        if manual:
            ##ct_file = out_path +'/GTs/P'+i+'/image.nii.gz'
            ##ct_file = '/home/igt/Projects/PerPlanRT/temp_data/UrCTIRMRennesDICOM/GTs/ID'+i+'/image.nii.gz'  ###
            ##ct_file = '/home/igt/Projects/PerPlanRT/temp_data/AUTOPLAN_ANONYMIZED_Nifti/'+i+'/CT.nii.gz'
            ct_file = '/home/igt/Projects/PerPlanRT/FrameworkSegmentation_batch/Output/ReToxiRennes/GTs/'+i+'/image.nii.gz'
            ct_img  = sitk.ReadImage(ct_file)
            
        else:
            ct_file = out_path +'/CTs/IGRT_'+i+'_0000.nii.gz'
            ct_img  = sitk.ReadImage(ct_file)

        #----------------------------------------------------------------------------------------------------------------------#
        # Predicted segmentation
        #----------------------------------------------------------------------------------------------------------------------#

        # Resampling
        print('Native: res',res_ct,'origin',origin_ct)
        print('VOI: res',res_voi,'origin',origin_voi)
        res_pred_img = resample_volume(pred_seg, interpolator=sitk.sitkNearestNeighbor, new_spacing=res_ct)
        print('Origin CT',ct_img.GetOrigin(),'Origin labelmap',pred_seg.GetOrigin(),'Origin labelmap resample',res_pred_img.GetOrigin())
        print('Spacing CT', ct_img.GetSpacing(),'Spacing labelmap',pred_seg.GetSpacing(),'Spacing labelmap resample',res_pred_img.GetSpacing())
        print('Size CT',ct_img.GetSize(),'Size labelmap',pred_seg.GetSize(),'Size labelmap resample',res_pred_img.GetSize())
        origin_pred, res_pred = res_pred_img.GetOrigin(), res_pred_img.GetSpacing()

        #new_origin = (yoff1*(res_pred[0])+origin_pred[0], xoff1*(res_pred[1])+origin_pred[1], zoff1*(res_pred[2])+origin_pred[2])
        res_pred_img.SetOrigin(origin_voi)
        res_pred_img.SetSpacing(ct_img.GetSpacing())
        res_pred_img.SetDirection(ct_img.GetDirection())

        out_path_native = out_path+'/Native/ID_'+i    
        check_if_exist(out_path_native)
        file_seg_name = out_path_native +'/ID_'+i+'_'+seg+'.nii.gz'
        print('Saving... '+ file_seg_name)
        sitk.WriteImage(res_pred_img, file_seg_name, True)
    
def writeSlices(writer, series_tag_values, new_img, out_dir, i, thickness):
    
    ##print(i)
    image_slice = new_img[:,:,i]
    
    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
        
    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
    
    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    ##print('\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i)))))
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    
    image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
    
    image_slice.SetMetaData("0020|1041", str(thickness*i)) # Slice Location

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir,str(i)+'.dcm'))
    writer.Execute(image_slice)

def ct_to_dicom(file_ct, out_path):
    
    #----------------------------------------------------------------------------------------------------------------------#
    # CT Image
    #----------------------------------------------------------------------------------------------------------------------#
    ct_image = sitk.ReadImage(file_ct)
    print(ct_image.GetSize())
    print('Loading CT image from >> '+ file_ct)
    ct_image_np = sitk.GetArrayFromImage(ct_image)

    new_img = sitk.GetImageFromArray(ct_image_np)
    new_img.SetSpacing(ct_image.GetSpacing())
    new_img.SetOrigin(ct_image.GetOrigin())

    # NIFTI TO DICOM
    # Write the 3D image as a series
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = ct_image.GetDirection()
    thickness = ct_image.GetSpacing()[2]
    series_tag_values = [("0008|0031",modification_time), # Series Time
                      ("0008|0021",modification_date),    # Series Date
                      ("0008|0008","DERIVED\\SECONDARY\\AXIAL"), # Image Type: DERIVED (the pixel values have been derived pixel value of the ct image) and SECONDARY (image created after examination)
                      ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                      ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],            # Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                      ("0008|103e", "CT-image"),          # Series Description
                      ("0018|0050", str(thickness)) ]     # Slice Thickness

                      ##("0010|0010", "V1_IGRT_0"+id_case), # Patient Name
                      ##("0010|0020", "V1_IGRT_0"+id_case), # Patient ID 

    series_tag_values = series_tag_values + [
    ("0028|1053", "1"),      # Rescale slope
    ("0028|1052", "-1024"),  # Rescale intercept
    ("0028|0100", "16"),     # Bits allocated
    ("0028|0101", "16"),     # Bits stored
    ("0028|0102", "15"),     # High bit
    ("0028|0103", "1"),      # Pixel representation
    ("0028|1054", "HU")]     # Rescale Type

    # Write slices to output directory
    list(map(lambda i: writeSlices(writer, series_tag_values, new_img, out_path, i, thickness), range(ct_image.GetDepth())))
    print('Exportation to DICOM finished')
      
def export2dicomRT(file_ct, file_seg, out_path, seg):
    
    out_path_CT = out_path+'/CT'
    if not os.path.exists(out_path_CT):
        os.makedirs(out_path_CT)
        ct_to_dicom(file_ct, out_path_CT)
        
    if seg == 'OARs':
        out_path_CT = file_ct
        print(out_path_CT)
    #----------------------------------------------------------------------------------------------------------------------#
    # Segmentation
    #----------------------------------------------------------------------------------------------------------------------#
    seg_image = sitk.ReadImage(file_seg)
    print(seg_image.GetSize())
    print('Loading Segmentation from >> '+ file_seg)

    if 'OARs' in seg:
        labelMap_np = sitk.GetArrayFromImage(seg_image)
        rectum   = (labelMap_np==1).transpose((1, 2, 0))
        bladder  = (labelMap_np==2).transpose((1, 2, 0))
        prostate = (labelMap_np==3).transpose((1, 2, 0))
        vseminls = (labelMap_np==4).transpose((1, 2, 0))

        # NIFTI TO DICOM
        # Create new RT Struct. Requires the DICOM series path for the RT Struct.
        rtstruct = RTStructBuilder.create_new(dicom_series_path=out_path_CT)

        # Add the 3D mask (Numpy array of type bool) as an ROI.
        # Setting the color, description, and name
        rtstruct.add_roi(mask=rectum,   color=[128, 174, 128], name="Rectum")
        rtstruct.add_roi(mask=bladder,  color=[241, 214, 145], name="Bladder")
        rtstruct.add_roi(mask=prostate, color=[183, 156, 220], name="Prostate")
        rtstruct.add_roi(mask=vseminls, color=[111, 184, 210], name="SV")
        rtstruct.save(out_path+'/'+seg+'_rt-structs')
        
    if seg=='urethra':
        labelMap_np = sitk.GetArrayFromImage(seg_image)
        urethra     = (labelMap_np==1).transpose((1, 2, 0))

        # NIFTI TO DICOM
        # Create new RT Struct. Requires the DICOM series path for the RT Struct.
        rtstruct = RTStructBuilder.create_new(dicom_series_path=out_path_CT)

        # Add the 3D mask (Numpy array of type bool) as an ROI.
        # Setting the color, description, and name
        rtstruct.add_roi(mask=urethra,   color=[255, 0, 0], name="Urethra")
        rtstruct.save(out_path+'/'+seg+'_rt-structs')
    
