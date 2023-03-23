- Add to README description of the project (maybe abstract of the paper), figure of workflow, how to cite.
- Prerequirements: add all libraries in PerPlanRT_env + nnUNet
- For the moment, manual segs used only from DICOM --> include from NIfTI
- img to DICOM only possible with CT --> extend to MR


TO REVIEW
- utils.load_data.load_seg_data
- utils.voi_extraction.voi_seg_extraction & utils.voi_extraciton.voi_seg_extraction_MRI
- utils.postprocessing.postprocessing_reference_img, utils.postprocessing.writeSlices, utils.postprocessing.postprocessing_native_manual
