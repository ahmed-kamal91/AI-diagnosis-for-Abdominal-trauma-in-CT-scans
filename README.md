<h2>Data preparation:</h2>

Based on observations on EDA, problems can be solved in preparation step
is:

1.  **Large Number of Frames:** we developed method to increase spacing between frames by reduce the
slice thickness to only 5 millimeters, skipping the intermediate scans
(slice thickness range from 0.5 to 5mm) based on current thickness from
each dicom frame metadata.

2.  **Frame Size Relative to Quantity:** simply we will resize data from
    512x512 to 256x256.

3.  Solving problem related to datatype:

    1.  **Bit Depth Adjustment and Photometric Interpretation:** If the
        DICOM image has a Photometric Interpretation of \"MONOCHROME1\",
        it inverts the pixel values to ensure they are correctly
        interpreted.

    2.  **Hounsfield Unit Transformation:** The pixel values are
        transformed to Hounsfield units using the Rescale Intercept and
        Rescale Slope provided in the DICOM metadata.

    > *\# Transform to hounsfield units*
    >
    > intercept = dicom_image.RescaleIntercept
    >
    > slope = dicom_image.RescaleSlope
    >
    > pixel_array = pixel_array \* slope + intercept

    3.  **Windowing:** Windowing is applied to focus on a specific range of
        pixel values defined by the [Window Center]{.underline} and [Window
        Width]{.underline} attributes. Pixel values outside this range are
        clipped to ensure the resulting image has appropriate contrast.

    > *\# windowing*
    >
    > window_center = int(dicom_image.WindowCenter)
    >
    > window_width = int(dicom_image.WindowWidth)
    >
    > img_min = window_center - window_width // 2
    >
    > img_max = window_center + window_width // 2
    >
    > pixel_array = pixel_array.copy()
    >
    > pixel_array\[pixel_array \< img_min\] = img_min
    >
    > pixel_array\[pixel_array \> img_max\] = img_max

    4.  **Normalization:** Finally, the pixel values are normalized to the
        range \[0, 1\] by subtracting the minimum value and dividing by the
        range (maximum value - minimum value). This ensures that the pixel
        values are standardized and suitable for processing or display, at
        the end it multiply to 255 for visibility.

    > *\# normalization*
    >
    > denominator *=* (pixel_array - pixel_array.min())
    >
    > *numerator =* (pixel_array.max() - pixel_array.min())
    >
    > pixel_array = denominator / *numerator*

This marks the initial stage of data preprocessing, marking the starting
point for our model development journey.
