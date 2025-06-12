import os
import time
import SimpleITK as sitk
import pandas as pd
import numpy as np
from rt_utils import RTStructBuilder

from highdicom.seg.content import  SegmentDescription
from highdicom.sr.coding import Code
from highdicom.seg.sop import Segmentation
from highdicom.seg import SegmentAlgorithmTypeValues
from highdicom import AlgorithmIdentificationSequence
import pydicom
from pydicom.uid import generate_uid
import glob, shutil



from .utilities import check_if_exist, resample_volume
from glob import glob




def postprocessing_native(pred_file, out_path, idx, vol: str, ddbb: str, use_manual_OARs=False):
    """
    Function to translate the predicted volume or OARs back to the native space.
    Parameters:
        pred_file : loaded volume with SimpleITK, it should be the VOI or OARs segmentation
        out_path : directory to save results
        idx : case id (e.g. '0001')
        vol : type of volume to translate --> 'OARs' or 'VOI'
        ddbb : name of the database (e.g. 'testData')
        use_manual_OARs: whether to use manual OARs
    """
    import os
    import pandas as pd
    import numpy as np
    import SimpleITK as sitk
    from utils.utilities import check_if_exist

    print('\n')
    # prepare output directory & filename
    out_path_native = os.path.join(out_path, 'Native', idx)
    check_if_exist(out_path_native, create=True)
    file_vol_name = os.path.join(out_path_native, f'{ddbb}_{idx}_{vol}.nii.gz')

    if os.path.exists(file_vol_name):
        print(f'Output in the native space for case {idx} already exists')
        return

    print('Obtaining output in the native space...')

    # --- Load metadata for VOI cropping ---
    meta_dir = 'mVOIs' if use_manual_OARs else 'VOIs'
    df_meta = pd.read_csv(os.path.join(out_path, meta_dir, 'metadata.csv'))
    df_meta_i = df_meta.loc[df_meta['idx'] == float(''.join([s for s in idx if s.isdigit()]))]

    origin_voi = [df_meta_i[f'{c}VOI_res'].values[0] for c in ('x', 'y', 'z')]
    res_voi    = [df_meta_i[f'res_{c}VOI_res'].values[0] for c in ('x', 'y', 'z')]

    # --- Load original CT for reference ---
    ct_file = os.path.join(out_path, 'imgs', f'{ddbb}_{idx}_0000.nii.gz')
    or_img  = sitk.ReadImage(ct_file)

    print('Processing case:', idx)
    print('CT spacing/origin:', or_img.GetSpacing(), or_img.GetOrigin())
    print('VOI spacing/origin:', res_voi, origin_voi)

    # --- Bake VOI‐seg’s physical metadata into pred_file ---
    pred_file.SetOrigin(origin_voi)
    pred_file.SetSpacing(res_voi)
    pred_file.SetDirection(or_img.GetDirection())

    # --- One‐shot resample into exactly the CT grid ---
    res_pred_img = sitk.Resample(
        pred_file,
        or_img.GetSize(),            # target size = CT size [X, Y, Z]
        sitk.Transform(),            # identity
        sitk.sitkNearestNeighbor,    # preserve labels
        or_img.GetOrigin(),          # CT origin
        or_img.GetSpacing(),         # CT spacing
        or_img.GetDirection(),       # CT direction
        0,                            # fill value outside VOI
        pred_file.GetPixelID()       # same pixel type
    )

    print('Resampled segmentation:')
    print('  spacing:', res_pred_img.GetSpacing())
    print('  origin :', res_pred_img.GetOrigin())
    print('  size   :', res_pred_img.GetSize())

    # --- Save NIfTI in native space ---
    print('Saving...', file_vol_name)
    sitk.WriteImage(res_pred_img, file_vol_name, True)




def export2dicomRT(ct_dicom_folder: str, file_seg: str, out_path: str, seg: str):
    """
    Convert a NIfTI segmentation into a DICOM RTSTRUCT, perfectly aligned to the
    original CT series.

    Parameters
    ----------
    ct_dicom_folder : str
        Path to the ORIGINAL CT DICOM folder (one folder containing all .dcm CT slices).
    file_seg : str
        Path to the segmentation NIfTI (already resampled to CT voxel grid, but may
        have wrong metadata).
    out_path : str
        Directory where the RTSTRUCT (and a copy of the CTs) will be written.
    seg : str
        Identifier for the segmentation ('OARs_DLUS' or 'urethra_DLUS').
    """
    # 1) make sure output dir exists
    os.makedirs(out_path, exist_ok=True)

     #Copy original CT DICOMs into out_path
    if os.path.isdir(ct_dicom_folder):  # Assuming DICOM folder
        for f in glob(os.path.join(ct_dicom_folder, '*.dcm')):
            shutil.copy(f, out_path)

    # 3) read the CT volume (to grab its spacing/origin/direction)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_dicom_folder)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {ct_dicom_folder}")
    file_names = reader.GetGDCMSeriesFileNames(ct_dicom_folder, series_ids[0])
    reader.SetFileNames(file_names)
    ct_img = reader.Execute()

    # 4) read the NIfTI seg and *overwrite* its geometry so it matches the CT exactly
    seg_img = sitk.ReadImage(file_seg)
    seg_img.CopyInformation(ct_img)

    # 5) extract the numpy mask
    mask_np = sitk.GetArrayFromImage(seg_img)  # shape (z,y,x)

    # 6) build RTSTRUCT using the ORIGINAL DICOM folder
    rtstruct = RTStructBuilder.create_new(dicom_series_path=ct_dicom_folder)

    if 'OAR' in seg:
        organs = [
            (1, "Rectum",          [128,174,128]),
            (2, "Bladder",         [241,214,145]),
            (3, "Prostate",        [183,156,220]),
            (4, "SeminalVesicles", [111,184,210]),
        ]
        for label, name, color in organs:
            roi = (mask_np == label).transpose(1,2,0)
            if roi.any():
                rtstruct.add_roi(mask=roi, name=name, color=color)
        # --- NEW: tag this RTSTRUCT with y DLUS tool name ---
        rtstruct.set_series_description(f"DLUS_OAR")  

    else:  # urethra
        roi = (mask_np == 1).transpose(1,2,0)
        if roi.any():
            rtstruct.add_roi(mask=roi, name="Urethra", color=[255,0,0])
        rtstruct.set_series_description(f"DLUS_Urethra")  

    # 7) save it
    rtstruct.save(os.path.join(out_path, seg + "_rt-structs.dcm"))




def export_rt_and_dcmseg(
    ct_dicom_folder: str,
    seg_nifti: str,
    out_path: str,
    seg_name: str
):
    """
    Export both an RTSTRUCT and a DICOM SEG from a NIfTI mask.

    Parameters
    ----------
    ct_dicom_folder : str
        Path to a folder of CT .dcm files (will be copied into out_path for RTSTRUCT).
    seg_nifti : str
        Path to the labeled NIfTI (integer labels).
    out_path : str
        Directory in which to save RTSTRUCT and DCMSEG.
    seg_name : str
        Basename for saved files, e.g. 'OARs_DLUS' or 'urethra_DLUS'.
    """
    os.makedirs(out_path, exist_ok=True)

    # 1) Copy CT slices into a subfolder that RTStructBuilder will read
    ct_dest = os.path.join(out_path, "CT_FOR_RT")
    shutil.rmtree(ct_dest, ignore_errors=True)
    shutil.copytree(ct_dicom_folder, ct_dest)

    # 2) Load your NIfTI mask
    seg_img = sitk.ReadImage(seg_nifti)
    mask_np = sitk.GetArrayFromImage(seg_img)  # shape (Z,Y,X)

    # 3) Build the RTSTRUCT
    rtstruct = RTStructBuilder.create_new(dicom_series_path=ct_dest)
    if "OAR" in seg_name:
        rtstruct.set_series_description(f"DLUS_AI_Tool_Prediction_OARs")
    else:
        rtstruct.set_series_description(f"DLUS_AI_Tool_Prediction_Urethra")

    # Decide which labels to export
    if "OAR" in seg_name:
        # multi-label OARs: (label, name, color)
        definitions = [
            (1, "Rectum",          [128,174,128]),
            (2, "Bladder",         [241,214,145]),
            (3, "Prostate",        [183,156,220]),
            (4, "SeminalVesicles", [111,184,210]),
        ]
    else:
        # single-label urethra
        definitions = [
            (1, "Urethra", [255,0,0])
        ]

    # add each ROI to RTSTRUCT
    for label_val, label_name, color in definitions:
        roi = (mask_np == label_val).transpose(1,2,0)  # SimpleITK (Z,Y,X) → rt_utils expects (X,Y,Z)
        if roi.any():
            rtstruct.add_roi(mask=roi, name=label_name, color=color)

    rt_out = os.path.join(out_path, f"{seg_name}_rt-structs.dcm")
    rtstruct.save(rt_out)
    print("Wrote RTSTRUCT to", rt_out)

    # 4) Build a DICOM SEG
    # 4a) Gather CT frames in instance order
    ct_files = sorted([
        os.path.join(ct_dest,f)
        for f in os.listdir(ct_dest)
        if f.lower().endswith('.dcm')
    ])
    ct_ds = [pydicom.dcmread(p) for p in ct_files]
    ct0 = ct_ds[0]

    # 4b) Prepare segment descriptions
    if "OAR" in seg_name:
        seg_ids = [d[0] for d in definitions]
    else:
        seg_ids = [1]

    segment_descriptions = []
    for i in seg_ids:

        label_name = next(n for (val,n,_) in definitions if val==i)

        # build a proper AlgorithmIdentificationSequence
        algo_id = AlgorithmIdentificationSequence(
            name='DLUS AI Tool',
            family=Code('121000','DCM','Segmentation'),
            version='1.0.0',
            source='YourOrganization',      # optional
            parameters={'model':'Mixed_model'}
        )

        segment_descriptions.append(
            SegmentDescription(
                segment_number               = i,
                segment_label                = label_name,
                segmented_property_category = Code('T-D0050','SRT','Organ'),
                segmented_property_type     = Code('T-D0050','SRT',label_name),
                algorithm_type               = SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification     = algo_id
            )
    )
    


    # 4c) Rearrange mask to (numSegments, Z, Y, X)
    #pixel_array = np.stack([(mask_np==i).astype('uint8') for i in seg_ids], axis=0)
    # 4c) Rearrange mask to (Z, Y, X, numSegments)
    pixel_array = np.stack([(mask_np == i).astype('uint8') for i in seg_ids], axis=-1)

    #seg_ds = Segmentation(
    #    source_images=ct_ds,
    #    segmentation_type='BINARY',
    #    pixel_array=pixel_array,
    #    segment_descriptions=segment_descriptions,
    #    series_instance_uid=generate_uid(),
    #    sop_instance_uid=generate_uid(),
    #   instance_number=1,
    #    manufacturer='DLUS AI Tool',
    #)

    # build the DICOM-SEG
    seg_ds = Segmentation(
        source_images           = ct_ds,
        segmentation_type       = 'BINARY',
        pixel_array             = pixel_array,
        segment_descriptions    = segment_descriptions,
        # UIDs
        series_instance_uid     = generate_uid(),
        sop_instance_uid        = generate_uid(),
        # Instance numbering
        series_number           = ct0.SeriesNumber,             # use the same SeriesNumber as your CT
        instance_number         = 1,                            # first/only instance
        # Manufacturer/device metadata
        manufacturer            = ct0.Manufacturer,             # e.g. “Siemens”
        manufacturer_model_name = getattr(ct0, 'ManufacturerModelName', 'DLUS-AI-Model'),
        device_serial_number    = getattr(ct0, 'DeviceSerialNumber', '0000'),
        software_versions       = getattr(ct0, 'SoftwareVersions', ['1.0']),
        omit_empty_frames=False,

    )

      

    seg_out = os.path.join(out_path, f"{seg_name}_dcmseg.dcm")
    seg_ds.save_as(seg_out)
    print("Wrote DICOM-SEG to", seg_out)


        
def ct_to_dicom(file_ct, out_path):  ## can be deleted as not being used in the new pipeline
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

        
# REVIEW #########################################################################################################################
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
    
def writeSlices(writer, series_tag_values, new_img, out_dir, i, thickness): # can be deleted as not being used in the new pipeline
    
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
        
        
def postprocessing_native_manual(pred_seg, out_path, i, seg, file, manual):

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
            df_meta   = pd.read_csv(os.path.join(out_path, 'VOIs', 'metadata.csv'))
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
