# DLUS
GitHub repository for DLUS: Deep Learning-based Segmentation of Prostatic Urethra on Computed Tomography Scans for Treatment Planning

STILL IN CONSTRUCTION ................................................................................................................

Full paper available at: https://www.sciencedirect.com/science/article/pii/S2405631623000222                                                                           
Please cite as :                                                                                                                                                       
Cubero, L., García-Elcano, L, Mylona, E., Boue-Rafle, A., Cozzarini, C., Ubeira Gabellini, M.G. et al. "Deep learning-based segmentation of prostatic urethra on computed tomography scans for treatment planning." Phys Imaging Radiat Oncol (2023), https://doi.org/10.1016/j.phro.2023.100431.

![Figure1](https://user-images.githubusercontent.com/83298381/226644663-d59dfd54-1c1d-40e8-9a87-089862e4a396.png)

0. VARIABLE AND PATH DEFINITIONS                                                                                                                                       
Please, change the following options accordingly:                                                                                                                       

    ddbb             = ...              -->    Database name. No spaces, no underscore _                                                                               
    mode             = 'dicom'          -->    Image mode. Options: 'nifti', 'dicom'                                                                                   
    model            = 'Mixed_model'    -->    Model for OAR segmentation. Options: 'FR_model' (French rectum), 'Mixed_model' (French + Italian databases)             
    use_manual_OARs  = False            -->    Option to use manual OAR segmentations instead of automatic ones. Options : False, True                                   
                    

1. LOAD ORIGINAL IMAGES                                                                                                                                                 
Data must be structured in the following way:                                                                                                                         
  Two directories are needed for each database ddbb:                                                                                                                   
    input data    [data_path] : 'Input' > ddbb                                                                                                                         
    output data    [out_path] : 'Output' > ddbb                                                                                                                       
    
    Organization of input data: The ddbb folder should contain a different folder for each case to process. In each case folder, the image scan should be saved in a sub-folder named "img", and the manual OAR segmentations - if available - in a sub-folder named "mOAR".
    
    The loaded images will be saved in 'Output' > ddbb > 'imgs'
    
    If the manual OARs masks (for the bladder, rectum, prostate, and seminal vesicles) are available and saved in the sub-folder "mOAR", they will also be loaded. To load them in DICOM format, we have included a series of typical names used in the clinic to describe the rectum, bladder, prostate and seminal vesicles. These names can be found in utils.load_data.dicom_to_nifti(), and can be updated to include other terminologies by adding them to the available lists. Only the available masks of interest will be saved. 
    
    The loaded mOARs will be saved in 'Output' > ddbb > 'GTs'

2. VOI EXTRACTION                                                                                                                                                 
Localization Network + Crop using the centroid of the coarse prosate segmentation. Check the result to ensure that appropriate VOI has been created. Sometimes some images are not well predicted and it's necessary to modify this VOI manually to ensure that the OARs and urethra segmentations are accurate.

    The cropped VOIs will be saved in 'Output' > ddbb > 'VOIs' > 'imagesTs'

3. OARs SEGMENTATION : Fine Segmentation Network                                                                                                                      
A trained nnU-Net is called to predict the segmentations of the bladder, rectum, prostate and seminal vesicles. Depending of the model selected (FR_model or Mixed_model), the network called for inference differs. The main distinction between both networks is the protocol to segment the rectum, based on French delineations with FR_model, or trained with joined data from a French and two Italian medical institutions with Mixed_model.

    The predicted segmentations will be saved in 'Output' > ddbb > 'OARs' > model
    
    Then, a post-processing step allows to bring the OAR segmentations back to native space and to export them to DICOM format. 
    
    The VOI and predicted segmentations in the native space will be saved in 'Output' > ddbb > 'Native'
    
    The VOI and predicted segmentations in DICOM will be saved in 'Output' > ddbb > 'DICOM'

4. DISTANCE MAP COMPUTATION                                                                                                                                    
The bladder and prostate OARs segmentations are used to calculate the distance maps, which are then used to guide the urethra segmentation.

    The computed distance maps are saved in 'Output' > ddbb > 'DistanceMaps'
    
5. URETHRA SEGMENTATION                                                                                                                                        
A trained nnU-Net is called to predict the segmentation of the urethra based on the previously computed distance maps. 

    The predicted segmentations will be saved in 'Output' > ddbb > 'Urethra' > 'DLUS'
    
    Then, a post-processing step allows to bring the OAR segmentations back to native space and to export them to DICOM format. 
    
    The VOI and predicted segmentations in the native space will be saved in 'Output' > ddbb > 'Native'
    
    The VOI and predicted segmentations in DICOM will be saved in 'Output' > ddbb > 'DICOM'
