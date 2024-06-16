<h2>Abstract</h2>
Abdominal trauma refers to any injury to the abdomen, which is the region of the body between the chest and the pelvis. The physical examination is not enough for diagnosing traumatic abdominal injuriesand accurate and immediate diagnosis is critical to initiate timely interventions. Among the different diagnostic methods, computed tomography (CT) stands out as an indispensable tool for accurately detecting abdominal injuries due to its ability to provide detailed cross-sectional images.Blunt force abdominal injuries are the most common type of abdominal injury and are often caused by car accidents. They can lead to damage to internal organs and bleeding, sometimes without any pain. Therefore, detecting and classifying these injuries is critical for effective treatment. So, I worked on my graduation project to utilize deep learning capabilities to help medical professionals rapidly detect abdominal injuries accuraterly.

<h2>Notes to be discussed first</h2>

* Data we worked on doesn't contain all abdominal organs. only 4 abdomnial organs (which is liver, spleen, kidney and bowel) and existence of extravasation .
* medical field use image format called DICOM, briefly it considered as the image as an array with additional importatn attributes used by medical field.
* briefly, NIfTI is a file format used to store three-dimensional (3D) image data, used segmentation model input is nifti file.

<h2>Final Methodology: </h2>

1. **Convert DICOM files into NIFTI** using dcm2niix package. you can see more information about it from the link https://github.com/rordenlab/dcm2niix

```python
def Dcm2Nii(dcm_pth,nii_pth):
    print("converting...")
    os.system(f'dcm2niix -o {nii_pth} {dcm_pth}')
    print("DONE")
```
2. **Genrating masks** only for specified abdominal organs to get region of interest later. we used Total Segmentator as segmentation model to know more about total semgentator go to the link: https://github.com/wasserth/TotalSegmentator
   
```python
def genMasks(nii_pth, ms_pth):
    print("generating masks...")
    input_pth  = j(nii_pth,[f for f in l(nii_pth) if f.endswith('.nii')][0])
    os.system(f"TotalSegmentator -i {input_pth} -o {ms_pth} --fast --roi_subset liver urinary_bladder")
    print("Done")
```

3.  **Extract region of interest (ROI)** from generated mask by takiong only the first indices for first layer in the liver and the last layer for bowel
   
```python
def getROI(mfolder_pth, thresh):

    print("Extract ROI...")
    ROI =[]
    for m_name in l(mfolder_pth):
        m_pth = j(mfolder_pth,m_name)
        m = process_mask(m_pth)
        for idx,f in enumerate(range(m.shape[0])):
            if calc_black_pct(m[f,:,:]) < thresh: 
                ROI.append(idx)
                break
    print("DONE")
    return tuple(ROI)
```

4. preparation for DICOM data by
   - getting pixel array, increase spacing between frames by reduce the slice thickness to only 5 millimeters, skipping the intermediate scans (slice thickness range from 0.5 to 5mm) based on current thickness from each dicom frame metadata.
   - **resize data** from 512x512 to 256x256.
   - **Bit Depth Adjustment and Photometric Interpretation:** If the DICOM image has a Photometric Interpretation of "MONOCHROME1", it inverts the pixel values to ensure they are correctly interpreted.
   - **Hounsfield Unit Transformation:** The pixel values are transformed to Hounsfield units using the Rescale Intercept and Rescale Slope provided in the DICOM metadata.
   - **Windowing:** Windowing is applied to focus on a specific range of pixel values defined by the Window Center and Window Width attributes. Pixel values outside this range are clipped to ensure the resulting image has appropriate contrast.
   - **Normalization:** Finally, the pixel values are normalized to the range [0, 1] by subtracting the minimum value and dividing by the range (maximum value - minimum value). This ensures that the pixel values are standardized and suitable for processing or display, at the end it multiply to  255 for visibility.

```python
def preprocess_dcm2img (dicom_image):

    pixel_array = dicom_image.pixel_array
    
    if dicom_image.PixelRepresentation == 1:
        bit_shift = dicom_image.BitsAllocated - dicom_image.BitsStored
        dtype = pixel_array.dtype 
        new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dicom_image)
    
    if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = 1 - pixel_array
    
    # transform to hounsfield units
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope
    pixel_array = pixel_array * slope + intercept
    
    # windowing
    window_center = int(dicom_image.WindowCenter)
    window_width = int(dicom_image.WindowWidth)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    pixel_array = pixel_array.copy()
    pixel_array[pixel_array < img_min] = img_min
    pixel_array[pixel_array > img_max] = img_max
    
    # normalization
    pixel_array = (pixel_array - pixel_array.min())/(pixel_array.max() - pixel_array.min())
    
    return (pixel_array * 255).astype(np.uint8)
```

```python
def preprocess_single_dcm(dcm_path, jpgScan_pth, SIZE):

    dcm = pydicom.dcmread(dcm_path)
    img = preprocess_dcm2img(dcm) 
    img = cv2.resize(img, (SIZE, SIZE))

    #convert dicom to jpeg
    px = basename(dcm_path)
    out_pth = os.path.join(jpgScan_pth, px.replace(".dcm", ".jpeg"))
    
    cv2.imwrite(out_pth, img)
```
```python
def initPreprocess(dicom_scanFolder_pth, jpgScan_pth, strt, end, TICK=5, SIZE=256):
    
    print("Initial preprocessing based ROI...")

    dcms = sorted([int(x.replace(".dcm","")) for x in l(dicom_scanFolder_pth)])
    dcms = [str(x)+".dcm" for x in dcms]

    # get tick
    curr_tick = pydicom.dcmread(j(dicom_scanFolder_pth, dcms[0])).SliceThickness
    step = round(TICK/curr_tick)

    dcm_paths = []

    cpu_cores = multiprocessing.cpu_count()

    for idx,i in enumerate(range(0, len(dcms), step)):
        if idx >= strt and idx <=end:
            dcm_paths.append(j(dicom_scanFolder_pth, dcms[i]))
            _ = Parallel(n_jobs=(cpu_cores/2))(delayed(preprocess_single_dcm)(path, jpgScan_pth, SIZE) for path in dcm_paths)
            
    print("DONE")
```
