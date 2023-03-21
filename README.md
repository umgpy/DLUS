# DLUS
GitHub repository for DLUS: Deep Learning-based Segmentation of Prostatic Urethra on Computed Tomography Scans for Treatment Planning

![Figure1](https://user-images.githubusercontent.com/83298381/226644663-d59dfd54-1c1d-40e8-9a87-089862e4a396.png)



1. LOADING ORIGINAL IMAGES...                                                                                                                                         
Data must be structured in the following way:                                                                                                                         
  Two directories are needed for each database ddbb:                                                                                                                   
    input data    [data_path] : 'Input' > ddbb                                                                                                                         
    output data    [out_path] : 'Output' > ddbb                                                                                                                       
    
  Organization of input data: For each case folder, a sub-folder must be created for each modality. For example, sub-folder 'CT' contains all the DICOM files for the CT of case 0001, whereas sub-folder 'MR' contains all of the DICOM files for the MR of case 0001.
  ![im1](https://user-images.githubusercontent.com/83298381/226653596-96d43af4-352e-447c-9c73-69abf16466d7.png)
  
  
  
  The output data will be automatically saved as: 
  
  
  ![im2](https://user-images.githubusercontent.com/83298381/226653874-78ef7ed7-9747-447d-a813-503143160753.png)
