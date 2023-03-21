# DLUS
GitHub repository for DLUS: Deep Learning-based Segmentation of Prostatic Urethra on Computed Tomography Scans for Treatment Planning
![Figure1](https://user-images.githubusercontent.com/83298381/226644663-d59dfd54-1c1d-40e8-9a87-089862e4a396.png)



1. LOADING ORIGINAL IMAGES...
Data must be structured in the following way:
  Two directories are needed for each database ddbb:
    input data    [data_path] : 'Input' > ddbb 
    output data    [out_path] : 'Output' > ddbb 
    
  Organization of input data:
    'Input'
        |___ ddbb
              |___ 0001
                      |___ CT
                      |___ MR
              |___ 0002
                      |___ CT
                      |___ MR
              |___ 0003
                      |___ CT
                      |___ MR
              |___ ...
      
  The output data will be automatically saved as:
    'Output'
        |___ ddbb
              |___ CTs
                    |___ ddbb_0001_0000.nii.gz
                    |___ ddbb_0002_0000.nii.gz
                    |___ ddbb_0003_0000.nii.gz
                    |___ ...
              |___ MRs
                    |___ ddbb_0001_0001.nii.gz
                    |___ ddbb_0002_0001.nii.gz
                    |___ ddbb_0003_0001.nii.gz
                    |___ ...
