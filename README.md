# DLUS
GitHub repository for DLUS: Deep Learning-based Segmentation of Prostatic Urethra on Computed Tomography Scans for Treatment Planning

STILL IN CONSTRUCTION ................................................................................................................

Full paper available at: https://www.sciencedirect.com/science/article/pii/S2405631623000222
Please cite as :
Cubero, L., García-Elcano, L, Mylona, E., Boue-Rafle, A., Cozzarini, C., Ubeira Gabellini, M.G. et al. "Deep learning-based segmentation of prostatic urethra on computed tomography scans for treatment planning." Phys Imaging Radiat Oncol (2023), https://doi.org/10.1016/j.phro.2023.100431.

![Figure1](https://user-images.githubusercontent.com/83298381/226644663-d59dfd54-1c1d-40e8-9a87-089862e4a396.png)


1. LOADING ORIGINAL IMAGES...                                                                                                                                         
Data must be structured in the following way:                                                                                                                         
  Two directories are needed for each database ddbb:                                                                                                                   
    input data    [data_path] : 'Input' > ddbb                                                                                                                         
    output data    [out_path] : 'Output' > ddbb                                                                                                                       
    
  Organization of input data: For each case folder, a sub-folder must be created for each modality. For example, sub-folder 'CT' contains all the DICOM files for the CT of case 0001, whereas sub-folder 'MR' contains all of the DICOM files for the MR of case 0001.
  
  
![im1](https://user-images.githubusercontent.com/83298381/226656731-c304ab0e-67ea-4be0-a3a4-e6b92797272e.png)

  
  
  The output data will be automatically saved as: 
  
  ![im2](https://user-images.githubusercontent.com/83298381/226656757-c1e38fdb-710d-4431-8343-3dec33ca8c94.png)


