<h2>Abstract</h2>
Abdominal trauma refers to any injury to the abdomen, which is the region of the body between the chest and the pelvis. The physical examination is not enough for diagnosing traumatic abdominal injuriesand accurate and immediate diagnosis is critical to initiate timely interventions. Among the different diagnostic methods, computed tomography (CT) stands out as an indispensable tool for accurately detecting abdominal injuries due to its ability to provide detailed cross-sectional images.Blunt force abdominal injuries are the most common type of abdominal injury and are often caused by car accidents. They can lead to damage to internal organs and bleeding, sometimes without any pain. Therefore, detecting and classifying these injuries is critical for effective treatment. So, I worked on my graduation project to utilize deep learning capabilities to help medical professionals rapidly detect abdominal injuries accuraterly.

<h2>Notes to be discussed first</h2>
![image](https://github.com/ahmed-kamal91/AI-diagnosis-for-Abdominal-trauma-in-CT-scans/assets/91970695/397e5ea7-2894-43f4-8078-5c8b9bbf90bd)

* briefly, Data is unbalanced so we will count on F1-score, recall and precision for model performance evaluation
* Data we worked on doesn't contain all abdominal organs. only 4 abdomnial organs (which is liver, spleen, kidney and bowel) and existence of extravasation .
* medical field use image format called DICOM, briefly it considered as the image as an array with additional importatn attributes used by medical field.
* briefly, NIfTI is a file format used to store three-dimensional (3D) image data, used segmentation model input is nifti file.

<h2>Final Methodology: </h2>

![image](https://github.com/ahmed-kamal91/AI-diagnosis-for-Abdominal-trauma-in-CT-scans/assets/91970695/a93764df-48ad-4fc0-b87e-0a79eeb5eb24)

```python
#pipeline
reindex(dicom_scanFolder_pth)
Dcm2Nii(dicom_scanFolder_pth, nifti_scanFolder_pth)
genMasks(nifti_scanFolder_pth, masks_scanFolder_pth) 
strt, end = getROI(masks_scanFolder_pth, thresh = 96.5)
initPreprocess(dicom_scanFolder_pth, jpg_scanFolder_pth, strt, end) #(53.8s)
modelInput = prepareModelInput(jpg_scanFolder_pth, num_frames=10)
b, e, k, l, s = runModel(model_pth, modelInput)
br, er, kr, lr, sr = getResults(b, e, k, l, s, thresh=0.5)
print(f"bowel result >>> {br}\nextravasation result >>> {er}\nKidney result >>> {kr}\nLiver result >>> {lr}\nSpleen result >>> {sr}")
```

1. **Convert DICOM files into NIFTI** using dcm2niix package. you can see more information about it from the link https://github.com/rordenlab/dcm2niix

```python
def Dcm2Nii(dcm_pth,nii_pth):
    print("converting...")
    os.system(f'dcm2niix -o {nii_pth} {dcm_pth}')
    print("DONE")
```
2. **Generating masks** only for specified abdominal organs to get region of interest later. we used Total Segmentator as segmentation model to know more about total semgentator go to the link: https://github.com/wasserth/TotalSegmentator
   
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
   - getting pixel array, increase spacing between frames by reduce the slice thickness to only 5 millimeters, skipping the intermediate scans (slice thickness range from 0.5 to 5mm) based on current thickness from each dicom frame metadata.</br>

   - **resize data** from 512x512 to 256x256.</br>
   
   - **Bit Depth Adjustment and Photometric Interpretation:** If the DICOM image has a Photometric Interpretation of "MONOCHROME1", it inverts the pixel values to ensure they are correctly interpreted.</br>
   
   - **Hounsfield Unit Transformation:** The pixel values are transformed to Hounsfield units using the Rescale Intercept and Rescale Slope provided in the DICOM metadata.</br>
   
   - **Windowing:** Windowing is applied to focus on a specific range of pixel values defined by the Window Center and Window Width attributes. Pixel values outside this range are clipped to ensure the resulting image has appropriate contrast.</br>
   
   - **Normalization:** Finally, the pixel values are normalized to the range [0, 1] by subtracting the minimum value and dividing by the range (maximum value - minimum value). This ensures that the pixel values are standardized and suitable for processing or display, at the end it multiply to  255 for visibility.</br>

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
5. Data preprocessing as model input: using ROI (start and end indices) we will split scan frames into 9 parts to take indices in between to always get fixed number of frames as input for model architecture.
![image](https://github.com/ahmed-kamal91/AI-diagnosis-for-Abdominal-trauma-in-CT-scans/assets/91970695/84344375-28fa-40a7-87a9-44753ddd30ac)
figure assume splitting into 3 parts to get 4 indicies.
```python
def prepareModelInput(jpg_scanFolder_pth, num_frames=10):
    
    print("input preparing...")
    
    frame_pths = sorted([int(f.split('.')[0]) for f in l(jpg_scanFolder_pth)])
    frame_pths = [str(f)+'.jpeg' for f in frame_pths]

    frame_pths = select_elements_with_spacing( frame_pths, num_frames)

    images = []
    for f in frame_pths:
        image = preprocess_jpeg(jpg_scanFolder_pth,f)
        images.append(image)
        
    images = np.stack(images)
    images = np.expand_dims(images, axis=0)
    image = torch.tensor(images, dtype = torch.float)

    print("DONE")
    
    return image
```

6. forward data into the trained model architecture which is mainly count on effcientnetb0
![image](https://github.com/ahmed-kamal91/AI-diagnosis-for-Abdominal-trauma-in-CT-scans/assets/91970695/1928cf0c-8bdd-46fc-ba93-7cbbd0f569eb)
```python
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input = nn.Conv2d(10, 3, kernel_size = 3)
        
        model = models.efficientnet_b0(weights = 'IMAGENET1K_V1')
        
        self.features = model.features
        self.avgpool = model.avgpool
        
        #heads
        self.bowel = nn.Linear(1280, 1) #1,0

        self.extravasation = nn.Linear(1280, 1) #1.0

        self.kidney = nn.Linear(1280, 3)

        self.liver = nn.Linear(1280,3) 

        self.spleen = nn.Linear(1280, 3)
    
    def forward(self, x):
        
        # extract features
        x = self.input(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # output logits
        bowel = self.bowel(x)
        extravsation = self.extravasation(x)
        kidney = self.kidney(x)
        liver = self.liver(x)
        spleen = self.spleen(x)
        
        return bowel, extravsation, kidney, liver, spleen
```
loading trained model and get results:
```python
def runModel(model_path, modelInput): 
    "initial model..."
    model = CNNModel()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()  
    print('predicting...')
    with torch.no_grad():
        bowel, extravasation, kidney, liver, spleen = model(modelInput)
        print('DONE')
        return bowel, extravasation, kidney, liver, spleen
    
```
7. get result as final readable outptut
```python
def getResults(b, e, k, l, s, thresh=0.5):
    
    getdic = lambda name: {0: f"Healthy {name}", 1: f"Low Injury {name}", 2: f"High Injury {name}"}

    br = "Healthy bowel" if b < 0.5 else "injured bowel"
    er = "NO extravasation detected" if e < 0.5 else "extravasation DETECTED"

    kr = np.argmax(k).item()
    kr = getdic("Kidneys")[kr]

    lr = np.argmax(l).item()
    lr = getdic("Liver")[lr]

    sr = np.argmax(s).item()
    sr = getdic("Spleen")[sr]

    return br, er, kr, lr, sr
```
<h2>Final Performance:</h2>
In most classification problem cases Accuracy metrics are being used as a main for performance measurement. accuracy measures the overall correctness of predictions, but it doesn't account for class imbalances. In cases where one class greatly outnumbers the other(s), a model can achieve high accuracy by simply predicting the majority class most of the time. However, this doesn't necessarily indicate that the model is performing well for the minority class(es). so, it’s imperative to evaluate the model's performance metrics that accurately reflect its effectiveness in handling class imbalances across predicted categories which is crucial for obtaining a comprehensive understanding of the model's capabilities and its real-world applicability.

|        | Accuracy      | AUC-ROC       | f1-SCORE      | RECALL        | PRECISION      |
|--------|---------------|---------------|---------------|---------------|----------------|
|        | train | valid | train | valid | train | valid | train | valid | train  | valid |
| **Bowel**  | 98.23% | 98.56% | 88.19% | 90.49% | 32.98% | 35.54% | 28.24% | 44.17% | 48.73% | 42.08% |
| **Extravasation**  | 92.42% | 91.41% | 85.28% | 87.92% | 40.33% | 43.11% | 51.95% | 60.94% | 34.14% | 35.64% |
| **Liver**  | 90.76% | 90.33% | 77.41% | 80.63% | 87.32% | 86.95% | 90.75% | 90.33% | 86.65% | 86.27% |
| **Kidney**  | 94.14  | 93.86% | 74.68% | 77.82% | 91.71% | 91.24% | 94.13% | 93.86% | 91.41% | 90.38% |
| **Spleen**  | 90.64% | 90.84% | 75.27% | 77.74% | 87.04% | 87.37% | 90.64% | 90.84% | 86.19% | 86.41% |
