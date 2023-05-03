import os
import SimpleITK as sitk
import warnings

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

        
def resample_volume(img_data, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1]):
    original_spacing = img_data.GetSpacing()
    original_size = img_data.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(img_data, new_size, sitk.Transform(), interpolator,
                         img_data.GetOrigin(), new_spacing, img_data.GetDirection(), 0,
                         img_data.GetPixelID())
