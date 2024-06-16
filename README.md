<h2>Abstract</h2>
Abdominal trauma refers to any injury to the abdomen, which is the region of the body between the chest and the pelvis. The physical examination is not enough for diagnosing traumatic abdominal injuriesand accurate and immediate diagnosis is critical to initiate timely interventions. Among the different diagnostic methods, computed tomography (CT) stands out as an indispensable tool for accurately detecting abdominal injuries due to its ability to provide detailed cross-sectional images.Blunt force abdominal injuries are the most common type of abdominal injury and are often caused by car accidents. They can lead to damage to internal organs and bleeding, sometimes without any pain. Therefore, detecting and classifying these injuries is critical for effective treatment. So, I worked on my graduation project to utilize deep learning capabilities to help medical professionals rapidly detect abdominal injuries accuraterly.

<h2>Notes to be discussed first</h2>
* Data we worked on doesn't contain all abdominal organs. only 4 abdomnial organs (which is liver, spleen, kidney and bowel) and existence of extravasation .
* medical field use image format called DICOM, briefly it considered as the image as an array with additional importatn attributes used by medical field.
* briefly, NIfTI is a file format used to store three-dimensional (3D) image data, used segmentation model input is nifti file.

<h2>Final Methodology: </h2>
1. Convert DICOM files into NIFTI using dcm2niix package. you can see more information about it from the link https://github.com/rordenlab/dcm2niix
```
def Dcm2Nii(dcm_pth,nii_pth):
    print("converting...")
    os.system(f'dcm2niix -o {nii_pth} {dcm_pth}')
    print("DONE")
```
2. Genrating masks only for specified abdominal organs to get region of interest later. we used Total Segmentator as segmentation model to know more about total semgentator go to the link: https://github.com/wasserth/TotalSegmentator
```
def genMasks(nii_pth, ms_pth):
    print("generating masks...")
    input_pth  = j(nii_pth,[f for f in l(nii_pth) if f.endswith('.nii')][0])
    os.system(f"TotalSegmentator -i {input_pth} -o {ms_pth} --fast --roi_subset liver urinary_bladder")
    print("Done")
```
3.  Extract region of interest (ROI) from generated mask by takiong only the first indices for first layer in the liver and the last layer for bowel 
