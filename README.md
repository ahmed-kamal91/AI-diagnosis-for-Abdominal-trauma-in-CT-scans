<h2>1.1 Data preparation:</h2>
Based on observations on EDA, problems can be solved in preparation step is:
* Large Number of Frames: we developed method to increase spacing between frames by reduce the slice thickness to only 5 millimeters, skipping the intermediate scans (slice thickness range from 0.5 to 5mm) based on current thickness from each dicom frame metadata.

```
curr_tick = df_dicom.loc[(p, s), "SliceThickness"].mean()
step = round(TICK/curr_tick)

dcm_paths = []
  for i in range(0, n_d, step):
     dcm_paths.append(J(s_folder, dcms[i]))

```

* Frame Size Relative to Quantity: simply we will resize data from 512x512 to 256x256.
* Solving problem related to datatype:
	- Bit Depth Adjustment and Photometric Interpretation: If the DICOM image has a Photometric Interpretation of "MONOCHROME1", it inverts the pixel values to ensure they are correctly interpreted.

```
if dicom_image.PixelRepresentation == 1:
    bit_shift = dicom_image.BitsAllocated - dicom_image.BitsStored
    dtype = pixel_array.dtype 
    new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array,
    dicom_image)
if dicom_image.PhotometricInterpretation == "MONOCHROME1":
    pixel_array = 1 - pixel_array

```
`
	- Hounsfield Unit Transformation: The pixel values are transformed to Hounsfield units using the Rescale Intercept and Rescale Slope provided in the DICOM metadata.
	- Windowing: Windowing is applied to focus on a specific range of pixel values defined by the Window Center and Window Width attributes. Pixel values outside this range are clipped to ensure the resulting image has appropriate contrast.
	- Normalization: Finally, the pixel values are normalized to the range [0, 1] by subtracting the minimum value and dividing by the range (maximum value - minimum value). This ensures that the pixel values are standardized and suitable for processing or display, at the end it multiply to  255 for visibility.

<h2>Data retrieval technique as input:</h2>
Given the persistently large number of frames and the variability in their quantity within each scan, we've developed an initial strategy to address this. We'll extract a subset of frames from each scan, focusing on the abdominal region. Specifically, we'll select four frames from the middle section of each scan using a predetermined threshold. 

using the code below:

```
def select_elements_with_spacing(input list, spacing):
    if len(input_list) < spacing * 4:
        raise ValueError("List should contain at least 4 * spacing elements.")
        
    # We want to select elements in the middle part of the abdomen
    lower_bound = int(len(input_list) * 0.4)
    upper_bound = int(len(input_list) * 0.6)

    spacing = (upper_bound - lower_bound) // 3
        
    selected_indices = [lower_bound, lower_bound + spacing, lower_bound + (2*spacing), upper_bound]
    
    selected_elements = [input_list[index] for index in selected_indices]
    
    return selected_elements
```

Initially, we assumed that each scan encompasses the entire CT body, although this wasn't universally the case. To extract the middle section from the body, we set the upper bound index at 40% of the scan length and the lower bound at 60%. We then divided this segment into thirds to obtain the two indices situated in the middle between the upper and lower bounds.

![image](https://github.com/ahmed-kamal91/AI-diagnosis-for-Abdominal-trauma-in-CT-scans/assets/91970695/5a04c46f-4ff9-4de0-9c53-dd484181eee1)
 
this cut organs one time at least (taking at least one slice from specified organ) as beginning. In future steps we will increase number of slices based on this idea of retrieval.
