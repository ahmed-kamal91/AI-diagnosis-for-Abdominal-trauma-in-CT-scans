Exploratory Data Analysis:
from exploration we conclude:
1-Data unbalancing
During exploratory data analysis (EDA), we uncovered the issue of unbalanced data, which is a typical occurrence in medical research, given that a significant proportion of patients tend to be in good health.  The following visualization depicts the distribution of healthy instances versus injuries, along with specific breakdowns within each category. For bowel and extravasation, there are binary classes with the following percentages: healthy (98%) versus injury (2%) for bowel, and healthy (93.6%) versus injury (6.4%) for extravasation. Additionally, for kidneys, liver, and spleen, which have multi-class classifications, the percentages are as follows: healthy (94.2%) versus low injury (3.7%) versus high injury (2.1%) for kidneys, healthy (89.8%) versus low injury (8.2%) versus high injury (2%) for liver, and healthy (88.8%) versus low injury (6.3%) versus high injury (4.9%) for spleen.

CATEGORY	HEALTHY	INJURY (LOW)	HIGH INJURY
KIDNEY	94.2%	3.7%	2.1%
LIVER	89.8%	8.2%	2%
SPLEEN	88.8%	6.3%	4.9%
BOWEL	98%	2%
EXTRAVASATION	93.6%	6.4%



In most classification problem cases Accuracy metrics are being used as a main for performance measurement. accuracy measures the overall correctness of predictions, but it doesn't account for class imbalances. In cases where one class greatly outnumbers the other(s), a model can achieve high accuracy by simply predicting the majority class most of the time. However, this doesn't necessarily indicate that the model is performing well for the minority class(es). so, it’s imperative to evaluate the model's performance metrics that accurately reflect its effectiveness in handling class imbalances across predicted categories which is crucial for obtaining a comprehensive understanding of the model's capabilities and its real-world applicability.

Based on observation above:
 In our evaluation of model performance, we will utilize the following metrics:

	F1 Score: the harmonic mean of precision and recall, offers a balanced assessment of both metrics. This makes it particularly effective for gauging model performance in the presence of imbalanced data.
F1 Score = 2 *((Precision * Recall))/((Precision + Recall) )

	Precision: quantifies the ratio of true positive predictions to all positive predictions. It serves as a valuable metric for evaluating the model's accuracy, especially concerning predictions for the minority class.
Precisison=  TP/(TP+FP)

	Recall: measures the ratio of true positive predictions to all actual positive instances. Given our focus on the medical field, where correctly identifying ill patients is paramount, recall takes precedence over precision. It ensures that instances classified erroneously as healthy are minimized, thus prioritizing patient safety and accurate diagnosis.
Recall=  TP/(TP+FN)

	ROC AUC Score: provides a comprehensive assessment of model performance by quantifying the area under the ROC curve which evaluates the trade-off between true positive and false positive rates across various classification thresholds, offering robust insights into model efficacy that are unaffected by class imbalances.



2- Huge number of frames, big size and its variability per scan:
Based on our previous findings, the dataset comprises a hierarchical structure, with a main folder containing patient directories identified by unique IDs. Each patient directory may contain multiple subdirectories representing scans, each spanning a substantial portion of the body from lung to hip. Our analysis reveals that the spacing between frames within scans is minimal, with negligible variation observed between consecutive slices. Furthermore, the number of frames per scan varies significantly, presenting challenges in model training due to data complexity.

Based on observation above:
These variations pose several challenges for model training:

	Large Number of Frames: The abundance of frames within each scan poses computational challenges and makes it impractical to process each scan as a single unit during model training.
	Variability in Number of Frames: The inconsistent number of frames per scan further complicates model input, as most model architectures require a fixed number of frames for training (due to collect data from different source).
	Frame Size Relative to Quantity: The relatively large size of individual frames, coupled with the high number of frames per scan, presents computational and memory constraints during training.
3-Datatype:
There are several problems that may be mentioned in previous sections:
	Briefly, Data has dicom type which has additional metadata help doctors and hospitals. Additional metadata consider burden on model development to extract frame from them every time for training.

	In a DICOM image with a "MONOCHROME1" photometric interpretation, the pixel values represent the brightness of each pixel inversely. In other words, a pixel value of 0 represents the brightest possible intensity, while the maximum pixel value represents the darkest intensity. This is opposite to what we typically expect, where higher pixel values indicate brighter intensities. So not handling the "MONOCHROME1" photometric interpretation would lead to misinterpretation of pixel intensities. This misinterpretation could result in inverted or incorrectly displayed images.




	DICOM images typically store pixel values as raw intensity values, which are arbitrary and lack standardized units. These values represent the attenuation of X-rays as they pass through the patient's body during imaging. So, we need to transform pixels to Hounsfield units.

	DICOM images have a wide dynamic range of pixel values, representing the full range of X-ray attenuation within the scanned anatomy. NOT all this information is relevant or useful for visualization or analysis. We need to do operation called Windowing that allows us to adjust the dynamic range of pixel values displayed within a specific window width and center, focusing on the relevant anatomical structures or tissue densities.

important note: proudly all codes provided on this documentation are “Manually Designed”
 1.1 Data preparation:
Based on observations on EDA, problems can be solved in preparation step is:
	Large Number of Frames:
we developed method to increase spacing between frames by reduce the slice thickness to only 5 millimeters, skipping the intermediate scans (slice thickness range from 0.5 to 5mm) based on current thickness from each dicom frame metadata.







	Frame Size Relative to Quantity: simply we will resize data from 512x512 to 256x256.

	Solving problem related to datatype:

	 Bit Depth Adjustment and Photometric Interpretation: If the DICOM image has a Photometric Interpretation of "MONOCHROME1", it inverts the pixel values to ensure they are correctly interpreted.


	Hounsfield Unit Transformation: The pixel values are transformed to Hounsfield units using the Rescale Intercept and Rescale Slope provided in the DICOM metadata.

  # Transform to hounsfield units
  intercept = dicom_image.RescaleIntercept
  slope = dicom_image.RescaleSlope

  pixel_array = pixel_array * slope + intercept


	Windowing: Windowing is applied to focus on a specific range of pixel values defined by the Window Center and Window Width attributes. Pixel values outside this range are clipped to ensure the resulting image has appropriate contrast.
   
   # windowing
   window_center = int(dicom_image.WindowCenter)
   window_width = int(dicom_image.WindowWidth)

   img_min = window_center - window_width // 2
   img_max = window_center + window_width // 2

   pixel_array = pixel_array.copy()

   pixel_array[pixel_array < img_min] = img_min
   pixel_array[pixel_array > img_max] = img_max



	Normalization: Finally, the pixel values are normalized to the range [0, 1] by subtracting the minimum value and dividing by the range (maximum value - minimum value). This ensures that the pixel values are standardized and suitable for processing or display, at the end it multiply to  255 for visibility.
   
	# normalization

	denominator = (pixel_array - pixel_array.min())
	numerator = (pixel_array.max() - pixel_array.min())

    pixel_array = denominator / numerator
		    


This marks the initial stage of data preprocessing, marking the starting point for our model development journey.


Data retrieval technique as input:
Given the persistently large number of frames and the variability in their quantity within each scan, we've developed an initial strategy to address this. We'll extract a subset of frames from each scan, focusing on the abdominal region. Specifically, we'll select four frames from the middle section of each scan using a predetermined threshold. using the code below:


de¬f select_elements_with_spacing(input list, spacing):
    
    if len(input_list) < spacing * 4:
        raise ValueError("List should contain at least 4 * spacing elements.")
        
        
    # We want to select elements in the middle part of the abdomen
    lower_bound = int(len(input_list) * 0.4)
    upper_bound = int(len(input_list) * 0.6)

    spacing = (upper_bound - lower_bound) // 3
        
    selected_indices = [lower_bound, lower_bound + spacing, lower_bound + (2*spacing), upper_bound]
    
    selected_elements = [input_list[index] for index in selected_indices]
    
    return selected_elements


Initially, we assumed that each scan encompasses the entire CT body, although this wasn't universally the case. To extract the middle section from the body, we set the upper bound index at 40% of the scan length and the lower bound at 60%. We then divided this segment into thirds to obtain the two indices situated in the middle between the upper and lower bounds.
 
this cut organs one time at least (taking at least one slice from specified organ) as beginning. In future steps we will increase number of slices based on this idea of retrieval.
Model Architecture:
Model made with combination of: 
	Utilization of convolutional 2D layer apply three filters 3x3 for feature extraction.

	Integration of a pretrained model, specifically 'EfficientNet-B0' that we heavily counting on to leverage its learned representations. The effectiveness of model relies heavily on the baseline network:

 
	Implementation of five separate heads to address the multi-class nature of the detection problem:

	Bowel Head: consists of an output layer with a single unit, predicting either 0 for healthy or 1 for injured.

	Extravasation Head: Similarly, this head contains a single-unit output layer, indicating the presence (1) or absence (0) of extravasation in the abdomen.

	Liver, Spleen, and Kidneys Heads: Each of these heads is equipped with an output layer comprising three units. These units provide binary predictions for classes such as 'healthy', 'low injury', and 'high injury', offering an array of predictions for each organ in a binary format.

class CNNModel(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.input = nn.Conv2d(4, 3, kernel_size = 3)
        model = models.efficientnet_b0(weights = 'IMAGENET1K_V1')
        
        self.features = model.features
        self.avgpool = model.avgpool
        
        self.bowel = nn.Linear(1280, 1)
        self.extravasation = nn.Linear(1280, 1)
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


Model additional details:
	Optimizer: Adam (Adaptive Moment estimation)
	learning rate = 10-4, epochs = 12, batch size = 8 (due to RAM capacity)
	Loss calculation: 
	Binary Cross Entropy (BCE) for bowel and extravasation:
L(y,ŷ) = - (1/N) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]

	Cross Entropy (CE) for spleen , kidneys and liver:
L(y,ŷ) = - (1/N) * Σ [ yᵢ * log(ŷᵢ) ]

Data preprocessing
There is transformations are applied to the input data before model training to augment the dataset, enhance its diversity, and improve the robustness of the model :
1. Random Horizontal Flipping: This transformation randomly flips images horizontally, effectively doubling the size of the dataset. It helps the model learn invariant features and reduces the risk of overfitting by presenting mirrored versions of the original images.
2. Random Adjust Brightness: Random adjustments in brightness alter the intensity of pixel values within the image. By randomly modifying the brightness levels, the model learns to recognize objects under varying lighting conditions, making it more resilient to changes in illumination during inference.
3. Random Shear Transformation: Shear transformations introduce distortion by shifting one part of the image relative to the other along the x or y-axis. This can simulate perspective changes and deformations in the input data, aiding the model in learning to recognize objects from different viewpoints.
4. Random Zoom Transformation: Random zoom transformations alter the scale of the image, either zooming in or out by a random factor. This variation in scale helps the model learn to detect objects at different sizes and distances, improving its ability to generalize to objects of varying scales in real-world scenarios.
Initial overview Approach:
 
model Input:
so, we use developed method to standardize the extraction of a fixed number of frames from every scan for each patient, subsequently stacking them and converting them into tensors. Following this, we apply a suite of random transformations, including random horizontal flipping, random adjustment of brightness, random shear transformation, and random zoom transformation. Afterward, we derive the ground truth for each scan by associating it with the corresponding patient ID from the label table (data.csv), integrating a coarse dropout mechanism. Finally, we organize all processed data into a dictionary for streamlined extraction using keys.

  self.transform = Compose([

                RandomHorizontalFlip(),  # Randomly flip images left-right
                ColorJitter(brightness=0.2),  # Randomly adjust brightness
                ColorJitter(contrast=0.2),  # Randomly adjust contrast
                RandomAffine(degrees=0, shear=10),  # Apply shear transformation
                RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Apply zoom transformation
                RandomErasing(p=0.2, scale=(0.02, 0.2)), # Coarse dropout
                        ])



    def __getitem__(self, idx):
        
        # sample 4 image instances
        dicom_images = select_elements_with_spacing(self.img_paths[idx],
                                                    spacing = 2)
        patient_id = dicom_images[0].split('/')[-3]
        images = []
        

        
        for d in dicom_images:
            image = preprocess_jpeg(d)
            images.append(image)
            
        images = np.stack(images)
        image = torch.tensor(images, dtype = torch.float).unsqueeze(dim = 1)
        
        image = self.transform(image).squeeze(dim = 1)
        
        label = self.df[self.df.patient_id == int(patient_id)].values[0][1:-1]
        
        # labels
        bowel = np.argmax(label[0:2], keepdims = True)
        extravasation = np.argmax(label[2:4], keepdims = True)
        kidney = np.argmax(label[4:7], keepdims = False)
        liver = np.argmax(label[7:10], keepdims = False)
        spleen = np.argmax(label[10:], keepdims = False)
        
        
        return {
            'image': image,
            'bowel': bowel,
            'extravasation': extravasation,
            'kidney': kidney,
            'liver': liver,
            'spleen': spleen,
            'patient': int(patient_id) #added
        }

Additionally, we made cross-validation which allows us to obtain a more reliable and robust estimate of our model's performance. By repeatedly splitting the data into 4 training and validation folds, cross-validation mitigates the risk of overfitting and provides a better assessment of how well our model generalizes to unseen data. And training on folds.
Performance result:

We stabilize randomness before training to get stable measurement for performance.
Note: Accuracy is provided for reference purposes only and should not be solely relied upon for model development and evaluation. While accuracy can offer insight into model performance, it may not always be the most reliable metric. 


	Accuracy	AUC-ROC	F1-SCORE	Precision	Recall
	train	valid	train	valid	train	valid	train	valid	train	valid
B	97.98%	97.35%	24.00%	13.79%	85.71%	18.18%	13.95%	11.11%	86.82%	88.53%
E	93.58%	84.39%	44.24%	32.88%	47.76%	23.68%	41.20%	53.73%	88.15%	84.50%
L	90.32%	90.76%	88.09%	88.32%	87.65%	89.33%	90.32%	90.76%	81.38%	82.95%
K	94.32%	94.48%	92.38%	93.71%	92.63%	94.00%	94.32%	94.48%	82.68%	80.42%
S	88.46%	88.96%	85.11%	85.38%	85.68%	85.74%	88.46%	88.96%	76.66%	79.97%

Result Analysis:
There is clearly seen that there is general poor performance:
	AUC-ROC (train/valid), F1 score(valid) and precision(train/valid) confirm that model perform very poor in bowel 

	AUC-ROC(train/valid), F1 score(train/valid) and precision(train/valid) confirm poor model performance on predicting extravasation.

	Model performs worse on bowel more than extravasation  due to data unbalancing where extravsation has more positive cases than bowel, however in general both has less positive cases than other categories so both of them has low performance compared to others. 

	We confirm that Accuracy is not reliable in our case as its values close to ratio of healthy patient in each category that there is strong guess that it is highly biased


Note based exploration:
	we found very important reason effect on  general performance. Windowing operation on dicom data conversion cause corruption in 40% of data by making frames of scan almost black and due to the operation took so long time and data are big and we altered on our main data by mistake and almost all corrupted scans classified as healthy we decided to remove them especially that under-sampling need to be applied to reduce number of healthy patient (majority class)

	Due to high effect of data unbalancing we will try to reduce its effect by increase weight assigned to the positive class in loss calculation.

1.2 Data preparation
1.2.1 Handling corrupted data
We developed method to classify corrupted scans, we reach to each scan path in each patient, add frame name to the path,  read every frame using opencv. On working on X, Y axis, we count number of black pixels and divide it by total number of pixels per frame to get black pixel percentage then apply threshold (= 98.0) to confirm its black or not. If the scan under threshold it will be removed.

# Calculate black pixels percentage in single frame (X, Y)
#---------------------------------------------------------
def calculate_black_percentage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    black_pixels = np.sum(img == 0)
    total_pixels = img.shape[0] * img.shape[1]
    return (black_pixels / total_pixels) * 100


On working on Z  axe, we count number of black frames based on previous mentioned and  divide it by total number of frames to get black percentage for the whole scan, then counting the whole scan as black (corrupted) using another threshold (78%).



def get_scan_pct(folder_path, threshold):
    img_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        black_percentages = list(executor.map(calculate_black_percentage, img_paths))
    n_black_images = sum(1 for pct in black_percentages if pct >= threshold)
    return (n_black_images / len(black_percentages)) * 100



def Classify_Scans(data_path, threshold=98.0, scan_thresh=78):
    patient_ids = os.listdir(data_path)
    corrupted_scans = []

    total_patients = len(patient_ids)
    progress_bar = tqdm(total=total_patients, desc="Processing Patients")

    for patient_id in patient_ids:
        patient_path = os.path.join(data_path, patient_id)
        scan_ids = os.listdir(patient_path)
        for scan_id in scan_ids:
            scan_path = os.path.join(patient_path, scan_id)

            if get_scan_pct(scan_path, threshold) >= scan_thresh:
                corrupted_scans.append(os.path.join(patient_path, scan_id))
        progress_bar.update(1)

    progress_bar.close()
    return corrupted_scans

# remove corrupted scans first
#-----------------------------
def remove_corrupted_scans(address_list):
    for address in address_list:
        if os.path.isdir(address):
            shutil.rmtree(address)

# remove whole corrupted patients
#--------------------------------
def remove_empty_folders(paths):
    removed_patients = []
    for path in paths:
        if not os.listdir(path):
            removed_patients.append(path)
            os.rmdir(path)
    return removed_patients


We have guessed that not all scans for every patient are corrupted, so we remove corrupted scans first then ensure that  the patient folder empty or not. If it is empty, unfortunately that’s mean that the whole patient scans are corrupted and then remove the empty patient folder but fortunately our guesses are true and we could save 144 patient from removal. 
Finally, increase weight assigned to the positive class in loss calculation using this line of code:

 optimizer = torch.optim.Adam(model.parameters(), lr = LR)
 bce_b = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([4.0]).to('cuda'))
 bce_e = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([6.0]).to('cuda'))
 cce = nn.CrossEntropyLoss(label_smoothing = 0.05, weight = torch.tensor([2.0, 3.0, 5.0]).to('cuda'))


Performance result (2)
Input, Model architecture, model details still the same, changes only in cleaning data and increasing assigned weight for positive cases:

	Accuracy	AUC-ROC	F1-SCORE	Precision	Recall
	train	valid	train	valid	train	valid	train	valid	train	valid
B	98.92%	99.08%	57.58%	66.67%	79.17%	85.71%	45.24%	54.55%	98.61%	99.79%
E	96.61%	95.84%	72.50%	69.66%	67.05%	56.36%	78.91%	91.18%	97.01%	98.73%
L	92.53%	92.60%	91.60%	92.30%	91.45%	92.41%	92.53%	92.60%	92.12%	95.48%
S	95.34%	94.30%	94.12%	93.05%	94.51%	92.72%	95.34%	94.30%	91.57%	97.47%
K	91.83%	91.68%	90.44%	91.25%	90.85%	92.65%	91.83%	91.68%	89.41%	93.81%

Result Analysis:
In general, performance dramatically increase:
1-  AUC Scores: 
	AUC values close to 1 indicate excellent performance.
	AUC values around 0.5 suggest random guessing.
	Observation:
	In Bowel, model in training perform poorly close to guessing than predicting (57%), while getting good performance for validation due to data unbalancing where positive represent 1.5% so it maybe all validation data is negative. 

	In Extravasation, model perform poorly but train and validation measurement more normal than bowel due to having more positive cases than bowel (6.4% positive cases)
2-  F1 Scores:
	Higher F1 scores indicate a better balance between precision and recall.
	Observation:
	F11 score confirm that model in Extravasations perform poorly.
	F1 score is the most metric that didn’t’ effect too much with data unbalancing on getting  the performance.

3- Precision and Recall:
	High precision means when model predicts a positive case, it is highly likely to be correct.
	High recall indicates that the model can identify a large portion of positive cases.
	Observation:
	Precision confirmed on poor performance for the model to predict the injury in bowel .
	Recall highly affected with data unbalancing.

Notes based exploration:
All of them are logical error need to be solved :
	We also, found out that not all scans that the shape we expected to have frames from upper lungs to lower hip, so it is wrong logic to use constant threshold to detect the region of interest (ROI) which is the abdomen part containing the organs and any extravasation existence.

	 It is logically wrong to learn model only on 4 frames for scan which may not have the injured part. so, we need to increase number of frames per scan on model training.


Second Approach 
	Organ’s classification maybe biased by what around  that consider as noise in model learning. So, we need to segment organs (spleen, liver, kidneys) except extravasation that didn’t find in specific part so it only need complete abdomen part slices ROI and I guess bowel due to hard extraction for it but I suggest we need to subtract the other organs from the slices

So, we will use segmentation model called “Total Segmentator” to make two model, first for bowel and extravasation using total Segmentator for only getting ROI and the other model for  3d segment the data, then apply ensemble learning  technique to combine  two models 



 TotalSegmentator
Tool for segmentation of over 117 classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images. A large part of the training dataset can be downloaded from Zenodo (1228 subjects). You can also try the tool online at totalsegmentator.com.
 
Created by the department of Research and Analysis at University Hospital Basel. 
for more details: 
GitHub - wasserth/TotalSegmentator: Tool for robust segmentation of >100 important anatomical structures in CT images



Mask generation:
total Segmentator is being used to generate masks for 117 categories till now which mentioned on image above . Generating masks for each category individually in Neuroimaging Informatics Technology Initiative (NIfTI) format which offers a versatile framework providing a unified format for storing various modalities including CT scans. enabling the storage of volumetric datasets with varying dimensions and resolutions.
In default mode which called ‘total’ model took scan as input and generating all 117 categories as nifti files. If category not in the scan, mask will be completely black. Next to the default task (total) there are more subtasks with more classes openly available for any usage:
	total: default task containing 117 main classes (see here for a list of classes)
	lung_vessels: lung_vessels (cite paper), lung_trachea_bronchia
	body: body, body_trunc, body_extremities, skin
	cerebral_bleed: intracerebral_hemorrhage (cite paper)
	hip_implant: hip_implant
	coronary_arteries: coronary_arteries
	pleural_pericard_effusion: pleural_effusion (cite paper), pericardial_effusion 
	heartchambers_highres: myocardium, atrium_left, ventricle_left, atrium_right, ventricle_right, aorta, pulmonary_artery (trained on sub-millimeter resolution)
	appendicular_bones: patella, tibia, fibula, tarsal, metatarsal, phalanges_feet, ulna, radius, carpal, metacarpal, phalanges_hand
	tissue_types: subcutaneous_fat, torso_fat, skeletal_muscle
	vertebrae_body: vertebral body of all vertebrae (without the vertebral arch)
	face: face_region
note: These models are not trained on the full TotalSegmentator dataset but on some small other datasets. Therefore, expect them to work less robustly.

We can generate masks using command line interface (CLI) using this parameter specified for total Segmentator:
	--device: Choose CPU or GPU
	--fast: For faster runtime and less memory requirements use this option. It will run a lower resolution model (3mm instead of 1.5mm).
	--roi_subset: Takes a space-separated list of class names (e.g. spleen colon brain) and only predicts those classes. Saves a lot of runtime and memory. Might be less accurate especially for small classes (e.g. prostate).
	--preview: This will generate a 3D rendering of all classes, giving you a quick overview if the segmentation worked and where it failed (see preview.png in output directory).
	--ml: This will save one nifti file containing all labels instead of one file for each class. Saves runtime during saving of nifti files. (see here for index to class name mapping).
	--statistics: This will generate a file statistic. Json with volume (in mm³) and mean intensity of each class.
	--radiomics: This will generate a file statistics_radiomics.json with the radiomics features of each class. You must install pyradiomics to use this (pip install pyradiomics).
But basically, this is the command for mask generation taking  basic requirements (input / output)

  TotalSegmentator -i ct.nii.gz -o segmentations 


you also can download it as extension for 3D Slicer from ‘Extension Manager’ to get mask for individual scan
   
n our scenario, utilizing a command line interface for generating masks proves impractical due to the extensive size of our dataset. Therefore, we opt for a Python API to facilitate mask generation, allowing for greater automation and scalability in handling the data.

Python API:

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    totalsegmentator(input_path, output_path)

    # option 2: provide input and output as nifti image objects
    input_img = nib.load(input_path)
    output_img = totalsegmentator(input_img)
    nib.save(output_img, output_path)

for full argument:

def totalsegmentator(input: Union[str, Path, Nifti1Image], output: Union[str, Path, None]=None,
               ml=False, nr_thr_resamp=1, nr_thr_saving=6,
               fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
               statistics=False, radiomics=False, crop_path=None, body_seg=False,
               force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
               skip_saving=False, device="gpu", license_number=None,
               statistics_exclude_masks_at_border=True, no_derived_masks=False,
               v1_order=False, fastest=False, roi_subset_robust=None):


For our specific problem, it's unnecessary to consider all 117 classes within our dataset. Not only does this approach consume more time and computational resources, but it also adds complexity to our task without providing significant benefits. Instead, we can streamline our process by narrowing our focus to generating masks for regions of interest, namely the liver, spleen, kidney, and bowel. By doing so, we can allocate our resources more efficiently and pave the way for smoother integration into our workflow for future applications. This targeted approach ensures that we address the core aspects of our problem while minimizing unnecessary overhead.

liver and spleen masks is one piece except bowel consists of  several small classes (duodenum, small bowel, colon) and Kidneys consist of (kidney_right, kidney_left) ,, so we use argument ‘roi_subset’ from total Segmentator to assign needed classes.

Input for total Segmentator:
 If you notice total Segmentator use only nifti files or folder contain dicom frames, so we can temporarily convert our data to NIfTI file using this code: 

def convert_to_nifti(input_folder, output_file):
    # Get list of files in the input folder
    files = os.listdir(input_folder)
    files.sort()
    # Read the first DICOM file to get image properties
    first_dcm = pydicom.dcmread(os.path.join(input_folder, files[0]))
    image_shape = (int(first_dcm.Rows), int(first_dcm.Columns), len(files))
    pixel_spacing = (float(first_dcm.PixelSpacing[0]), float(first_dcm.PixelSpacing[1]), float(first_dcm.SliceThickness))
    # Initialize a numpy array to hold the image data
    image_data = np.zeros(image_shape, dtype=np.int16)
    # Read each DICOM file and store the pixel data in the numpy array
    for i, filename in enumerate(files):
        dcm = pydicom.dcmread(os.path.join(input_folder, filename))
        image_data[:, :, i] = dcm.pixel_array
    # Create a SimpleITK image from the numpy array
    sitk_image = sitk.GetImageFromArray(image_data, isVector=False)
    sitk_image.SetSpacing(pixel_spacing)
    # Write the SimpleITK image to a NIfTI file
    sitk.WriteImage(sitk_image, output_file)


Time consumed for generation: 9 hours
Note based exploration 
	There is 5% error in the masks but in general it is pretty good
	we decide to combine these masks together and see the difference between the whole scan and the whole segmented part  in training.

Masks preprocessing:
Orientation transformation, binary conversion and applying down sampling:
Alignment of the generated masks does not match the orientation of the data, necessitating careful adjustment. Typically, to address this misalignment, we rely on the metadata embedded in DICOM files to determine the correct orientation. However, as part of preprocessing for model training, we have already removed this metadata. Thankfully, we can rely on the consistency of orientation across all data instances. Consequently, our task involves only adjusting the orientation of the masks to match that of the data. We transpose the dimensions of the mask data to swap the first and second axes, rotates the mask data by 90 degrees counterclockwise along the second and third axes, flips the mask data along the first axis then transposes the dimensions again to revert to the original orientation, effectively achieving a 90-degree rotation.

 # Transform orientation, binary conversion, resize, increase tickness
 #--------------------------------------------------------------------
 def process_mask(pth, p, s, series_tick):
     m = nib.load(pth).get_fdata()
 
     m = np.transpose(m, [1, 0, 2])
     m = np.rot90(m, 1, (1, 2))
     m = m[::-1, :, :]
     m = np.transpose(m, [1, 0, 2])
 
     m = m.astype(np.float32)
     m[m < 0.5] = 0
     m[m>=0.5] =1
     m = m.astype(np.uint8)
 
     t = series_tick[f'{p}_{s}']
 
     m = m[::t,::2,::2]
 
     return m


after removal of dicom metadata we loss the thickness, but we handle this by developing method for retrieving it by subtracting first and second  index from every scan and save values in dictionary using patient and scan ids as keys to the thickness value for the specific scan.

  # getting tickness from each scan in data
  #-----------------------------------------
  def get_tickness():
      series_tick = {}
  
      for patient in os.listdir(d_pth):
          patient_path = os.path.join(d_pth, patient)
          for scan in os.listdir(patient_path):
              scan_path = os.path.join(d_pth, patient, scan)
  
              frames_list = sorted([int(prefix.split('.')[0]) for prefix in os.listdir(scan_path)])   
  
              tick = frames_list[1] - frames_list[0]
              series_tick[f'{patient}_{scan}'] = tick
      return series_tick


Masks conversions and combination:
Previously, we made the strategic decision to consolidate all masks corresponding to each scan into a unified NIfTI file. Following this consolidation, we transformed the NIfTI file into JPEG frames. This approach was adopted to ensure compatibility with our dataset, which has been converted to JPEG format. By aligning the data formats, specifically converting both masks and dataset to JPEG, we aim to streamline the training process for our models, fostering greater efficiency and ease of application.

(artwork represent combining nifti files and slicing it to convert as jpeg)

Code description:
	Initially, our process entails traversing through the directory structure to access the individual paths for segmentation data corresponding to each patient's scans. 

	upon establishing the foundational groundwork, we create a base container, meticulously designed to accommodate the amalgamation of all these segmented masks. Employing the fundamental principles of logical operations, specifically employing the logical OR operation element-wise, we seamlessly merge these individual masks into a comprehensive whole.


	Once this consolidation phase is complete, we embark on a crucial iterative process. This involves systematically looping through the combined NIfTI mask, meticulously slicing it to capture pertinent regions of interest.

	Here's where the essence of our methodology truly shines. With precision and finesse, we extract these delineated regions and transform them into a format conducive for further analysis and visualization. Leveraging the transformative power of image manipulation, we meticulously save each slice as a JPEG file.


	These newly minted JPEG representations serve as invaluable artifacts, encapsulating crucial insights derived from the segmentation process. As they find their home in a specified pathway, meticulously chosen for organizational clarity, they stand as a testament to our commitment to excellence in data management and analysis.








important note: proudly all codes provided on this documentation are “Manually Designed”

def convert_masks_to_jpeg():
    #=========PATIENTS==================
    for patient in os.listdir(d_pth):

        patient_path = os.path.join(d_pth,patient)
        
        new_patient_path = os.path.join(jpg_masks_pth, patient) #for new path
        #--make-patient-dir------------------------------------------------
        if not os.path.exists(new_patient_path): os.mkdir(new_patient_path)
        #------------------------------------------------------------------

        #===========SCANS====================
        for scan in os.listdir(patient_path):

            scan_path = os.path.join(patient_path, scan)

            mask_paths = os.path.join(nii_masks_pth, patient, scan) #blood twist
            masks_lists = os.listdir(mask_paths)
            
            prefixes = sorted([int(prefix.split('.')[0]) for prefix in os.listdir(scan_path)])
            prefixes = [ ''.join([str(p),'.jpeg']) for p in prefixes]

            new_scan_path = os.path.join(new_patient_path, scan) #for new path
            #--make-scan-dir:--------------------------------------------
            if not os.path.exists(new_scan_path): os.mkdir(new_scan_path)
            #------------------------------------------------------------ 

            # ***init base mask:***
            base_mask_path = os.path.join(nii_masks_pth, patient, scan, masks_lists[0]) # blood twist
            base_mask = process_mask(base_mask_path, patient, scan)



            # combinatation loop

            #====MASKS=============
            for mask in masks_lists:

                mask_path = os.path.join(nii_masks_pth, patient, scan, mask) # the blood twist
                modified_mask = process_mask(mask_path, patient, scan)
                print("DEBUG: mask:", mask)
                base_mask = np.logical_or(base_mask, modified_mask)
            
            print("DEBUG: masks combined : ", base_mask.shape)
            # Convert combined nii to jpeg frames 

            print("DEBUG: base_mask.shape[0] = ", base_mask.shape[0])
            print("DEBUG: len(prefixes) = ", len(prefixes))
            for i in range( base_mask.shape[0] ):

                frame_path_name = os.path.join(new_scan_path, prefixes[i])
                base_mask_uint8 = (base_mask * 255).astype(np.uint8)
                print("DEBUG: new path", frame_path_name)
                cv2.imwrite(frame_path_name, base_mask_uint8[i,:,:])
            
            #empty prefix list
            prefixes = []


Time consumed for combination and conversion: 10 hours
Output 
It is pretty to see masks finally made and compare it with the other side for the data which will be applied on












Data Preparation (3)
Applying masks on data:
With the preprocessing of masks now complete, the next step is to apply them to the corresponding data. As per our standard procedure, we've developed a dedicated code for this task. This code operates by concurrently reading the image and mask data for each scan. Subsequently, it utilizes logical AND operations elementwise to extract regions from the images where the binary mask equals one or evaluates to true. These extracted regions are then saved for further analysis or visualization, thereby facilitating the seamless integration of masks with the original data.

important note: proudly all codes provided on this documentation are “Manually Designed”


l= os.listdir
j=os.path.join

def main():
    
    md_pth = 'C:\\codingWorkspace\\masked_data'
    d_pth = 'C:\\codingWorkspace\\reduced_256_tickness_5'
    jpg_masks_pth = 'C:\\codingWorkspace\\segmentations'

    for p in tqdm(os.listdir(d_pth), desc="Patients Progress"):

        p_pth = os.path.join(d_pth, p)

        new_p_pth = os.path.join(md_pth, p) # for new path (patient)
        # --make-patient-dir------------------------------------------------
        if not os.path.exists(new_p_pth):
            os.mkdir(new_p_pth)
        # ------------------------------------------------------------------

        for s in os.listdir(p_pth):

            s_pth = os.path.join(d_pth, p, s)

            new_sc_pth = os.path.join(md_pth, p, s) # for new path (scan)
            # --make-scan-dir:--------------------------------------------
            if not os.path.exists(new_sc_pth):
                os.mkdir(new_sc_pth)
            # ------------------------------------------------------------ 

            for f in os.listdir(s_pth):
                frame_pth = os.path.join(d_pth, p, s, f)
                mask_pth = os.path.join(jpg_masks_pth, p, s, f)

                image = cv2.imread(frame_pth)
                mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                result = cv2.bitwise_and(image, image, mask=mask)

                result_path = os.path.join(md_pth, p, s, f)
                cv2.imwrite(result_path, result)

if __name__ == '__main__':
    main()

Time consumed for combination and conversion: 1.5 hour
Note based observations:
Applying masks on frames scans is good but causing some noise that may affect the model training, so, we decide to remove it using morphological processing

important note: proudly all codes provided on this documentation are “Manually Designed”

l = os.listdir
exist = os.path.exists
j = os.path.join
morph = cv2.morphologyEx

def main():
    md_pth = 'C:\\codingWorkspace\\masked_data'
    mfd_pth = 'C:\\codingWorkspace\\masked_filtered_data'

    k = np.ones((2,2),np.uint8)

    for p in tqdm(l(md_pth)):

        p_newpth = j(mfd_pth, p)
        if not exist(p_newpth): os.mkdir(p_newpth)

        for s in l(j(md_pth, p)):

            s_newpth = j(mfd_pth, p ,s)
            if not exist(s_newpth): os.makedirs(s_newpth)

            for f in l(j(md_pth, p, s)):

                f_pth = j(md_pth, p ,s, f)
                img = cv2.imread(f_pth)
                filtered_img = morph(img, cv2.MORPH_OPEN, k)

                f_newpth = j(mfd_pth, p, s, f)
                cv2.imwrite(f_newpth, filtered_img)
if __name__ == '__main__':
    main()

 .
Represent the same methodology on reaching image. Changes only on:
‘cv2.morphologyEx (img, cv2.MORPH_OPEN, k)’
Output:
Finally, we finished preparation on data and here is the final result for the second approach  with backup for each alteration made





Extract Region of Interest:
Based on our observations and analysis, we've devised a strategy to identify the regions of interest (ROI) within our dataset, as illustrated in the diagram below. The ROI spans from the spleen to the bowel, representing the abdominal region of interest. Once we apply masks to the scans, residual black frames remain in scans where no discernible information exists, hindering the learning process and diminishing the utility of these frames as input for our model architecture. To address this issue, we'll employ a function previously utilized for calculating black pixels within the scans, applying the same threshold to identify and subsequently remove these frames. By leveraging the indices of the first and last frames within the ROI, we'll curate the data for further processing by our model designed for extravasation detection. This approach ensures that we focus our model's attention on the relevant anatomical structures while optimizing the quality of input data for subsequent analysis..











we use the code below to remove to extract region of interest. this method provide backup.
important note: proudly all codes provided on this documentation are “Manually Designed”

l = os.listdir
j = os.path.join
exist = os.path.exists

def calculate_black_percentage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    black_pixels = np.sum(img == 0)
    total_pixels = img.shape[0] * img.shape[1]
    return (black_pixels / total_pixels) * 100, img

def main():

    mfd_pth = 'C:\\codingWorkspace\\masked_filtered_data' 
    nBmfd_pth = 'C:\\codingWorkspace\\nonBlack_MFdata'
    
    for p in tqdm(l(mfd_pth)):

        p_newpth = j(nBmfd_pth, p)
        if not exist(p_newpth): os.mkdir(p_newpth)

        for s in l(j(mfd_pth, p)):

            s_newpth = j(nBmfd_pth, p ,s)
            if not exist(s_newpth): os.makedirs(s_newpth)

            for f in l(j(mfd_pth, p, s)):

                f_pth = j(mfd_pth, p ,s, f)
                b_pct, img = calculate_black_percentage(f_pth)

                if b_pct <= 96.0:
                    f_newpth = j(nBmfd_pth, p, s, f)
                    cv2.imwrite(f_newpth, img)

if __name__ == '__main__':
    main()


Output:
For example we can see we successfully extract region of interest in the figure which for patient id 19 from  304 to 644 indices
 
Developed Data retrieval as input
After getting ROI  it is time to develop retrieval of data from the specified region and take more than four frames from each scan . we alter the previous function to be more general and behave with whatever you put from required number of frame to retrieve from scan, but scan should have at least number of frames equal to the required one.

Code Description
This method builds upon our previous approach, where we successfully extracted the region of interest (ROI) from the dataset. Once we've obtained the ROI, the need for upper and lower bounds diminishes, as their primary purpose was to identify the ROI, which we've already accomplished. What remains within each scan are the frames corresponding to the ROI.

To streamline the process, we set the lower bound index to zero and the upper bound index to the index of the last frame. This adjustment ensures that our focus remains solely on the frames relevant to the ROI, eliminating any extraneous data points.

Next, we determine the spacing step by dividing the total number of frames specified by the 'division' variable. This step is crucial for iterating through the frames effectively. We then loop through the frames based on this calculated division until we reach the index of the last frame. In cases where we don't reach the last frame, we'll retain only the frames obtained earlier during the extraction of the ROI. This iterative process ensures that we capture and retain the necessary frames within the ROI while optimizing computational efficiency.  
important note: proudly all codes provided on this documentation are “Manually Designed”
def select_elements_with_spacing(input_list, divsion):
      
    spacing = len(input_list) // divsion
    if spacing == 0 :
        spacing = 1

    selected_indices = [spacing * i for i in range(0,divsion-1)]
    selected_indices.append(len(input_list)-1)

    selected_elements = [input_list[index] for index in selected_indices]
    
    return selected_elements


Note Based Explorations:
	We decide to use 10 frames as retrieving data from scan, so we look at the percentage of scans which has less than 10 frames. So, we used the code
 d_pth = 'C:\\codingWorkspace\\masked_filtered_data'

 counter = 0
 total = 0
 for p in l(d_pth):
     for s in l(j(d_pth, p)):
         num_frames = len(l(j(d_pth, p, s)))
         if num_frames < 10:
             counter +=1
         total +=1
         
 print((counter/total)*100)
fortunately, all scan above ten fames (0% of scans has number of frame under 10)
Performance (3):

unfortunately, segmentation getting lower performance than before

	Accuracy	AUC-ROC	F1-Score	Precision	Recall
	Train	Valid	Train	Valid	Train	Valid	Train	Valid	Train	Valid
B	98.13%	98.51%	84.31%	88.84%	20.65%	24.35%	16.48%	28.65%	39.87%	26.35%
E	91.63%	90.42%	76.66%	80.01%	26.66%	31.63%	32.62%	43.04%	25.05%	28.90%
L	90.47%	89.88%	70.44%	73.68%	86.32%	85.54%	90.47%	89.88%	85.26%	85.12%
K	94.12%	93.60%	69.73%	72.16%	91.61%	90.70%	94.12%	93.60%	90.19%	88.45%
S	90.48%	90.49%	71.65%	74.05%	86.83%	86.86%	90.48%	90.49%	85.08%	84.44%

Note based exploration:
	The first model in our approach didn’t get the high performance we expect so now there is no need for continue in ensemble approach.

	After thinking it is reasonable not to change the performance because we only remove little tissues. And may  this little tissue maybe important for learning process.

Third Approach

After deliberation, we opt to utilize the Total Segmentator for extracting Regions of Interest (ROI) without the need for applying masks. Our approach involves determining the indices for the first and last slices corresponding to the region of interest in each scan. Subsequently, we train the model on the entire scan data, but confine the training process within the extracted ROI.




















Performance:
Thankfully, we can see improving in performance  compared to previous performance ‘s approach
Note: model still affected with unbalancing 

	Accuracy	AUC-ROC	f1-SCORE	RECALL	PRECISION
	train	valid	train	valid	train	valid	train	valid	train	valid
B	98.23%	98.56%	88.19%	90.49%	32.98%	35.54%	28.24%	44.17%	48.73%	42.08%
E	92.42%	91.41%	85.28%	87.92%	40.33%	43.11%	51.95%	60.94%	34.14%	35.64%
L	90.76%	90.33%	77.41%	80.63%	87.32%	86.95%	90.75%	90.33%	86.65%	86.27%
K	94.14	93.86%	74.68%	77.82%	91.71%	91.24%	94.13%	93.86%	91.41%	90.38%
S	90.64%	90.84%	75.27%	77.74%	87.04%	87.37%	90.64%	90.84%	86.19%	86.41%

We have some time to explore new Approach which is 3D one

Fourth Approach:

In this approach we will work on the scan as a block of 3d object to train on model made for 3d processing. But before we do this we need to handle data shape to be from [batch,10, 256,256] to 
[batch, 1, 10, 256, 256] which refer to [batch, channel, depth, height, width] and transformation step changed it position in code to match changes in shape

We can see changes in altered code below for retieving data:

    
     def __getitem__(self, idx):
        
        # sample 10 image instances
        dicom_images = select_elements_with_spacing(self.img_paths[idx], input_img_group)
        patient_id = dicom_images[0].split('\\')[-3]
        images = []
        
        for d in dicom_images:
            image = preprocess_jpeg(d)
            images.append(image)
            
        images = np.stack(images)
        image = torch.tensor(images, dtype = torch.float).unsqueeze(dim = 1)
  
        image = self.transform(image).squeeze(dim = 1)

        # the modst valuable code 
        image = image.unsqueeze(0)

        
        label = self.df[self.df.patient_id == int(patient_id)].values[0][1:-1]
        

the rest of code is the same as previous code
3D Model Architecture:

Our new architecture begins with a  3D convolutional layer, adeptly processing data across width, height, and depth dimensions within the input shape [batch, 1, 10, 256, 256], resulting in an output shape of [batch, 1, 8, 254, 254]. This refined representation then flows seamlessly into the integration of the cutting-edge pretrained model, '3D-ResNet', poised for exploration within our framework.

Note: used batch equal 5 due to high extensive training computation which take 2 hours for only 10 frames in each 3d object

 
Model Code
Code for specifying the model architecture
important note: proudly all codes provided on this documentation are “Manually Designed”

import torch
import torch.nn as nn
import torchvision

class 3DCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv3d = nn.Conv3d(1, 3, kernel_size=3)
        self.backbone = torchvision.models.video.r3d_18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove output layer
   
        # heads
        self.bowel_head = nn.Linear(in_features, 1)  # Binary 
        self.extravasation_head = nn.Linear(in_features, 1)  # Binary 
        self.kidney_head = nn.Linear(in_features, 3)  # Multi-class 
        self.liver_head = nn.Linear(in_features, 3)  # Multi-class 
        self.spleen_head = nn.Linear(in_features, 3)  # Multi-class 
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.backbone(x)
        # heads
        bowel_out = self.bowel_head(x)
        extravasation_out = self.extravasation_head(x)
        kidney_out = self.kidney_head(x)
        liver_out = self.liver_head(x)
        spleen_out = self.spleen_head(x)
        
        return bowel_out, extravasation_out, kidney_out, liver_out, spleen_out


Performance

	Accuracy	AUC-ROC	F1-Score	Precision	Recall
	train	valid	train	valid	train	valid	train	valid	train	valid
B	98.24%	98.38%	85.14%	89.35%	25.79%	18.02%	23.48%	23.25%	30.45%	19.19%
E	90.87%	88.23%	78.56%	79.51%	31.39%	30.91%	40.89%	49.03%	25.93%	23.52%
L	90.40%	89.31%	73.32%	75.89%	86.79%	85.68%	90.40%	89.31%	86.56%	83.67%
K	94.17%	93.29%	70.94%	71.17%	91.51%	90.57%	94.17%	93.29%	89.70%	88.38%
S	90.50%	90.09%	70.17%	71.94%	86.66%	86.24%	90.50%	90.09%	85.73%	84.29%

despite not getting better performance than ROI but I see it is promising and more robust I think in the future if we increase numbers of  frames it will give better results but for our short time we will end with third approach. 

Note: I tried to use the 3d version of efficientnet-b0 but faces some problems related to applying filter inside the model (5x5x5) bigger than the volume itself which became (4, 34, 34), so due to time limit we ignore using it 











Final Approach

third approach is the final one, with performance:

 we opt to utilize the Total Segmentator for extracting Regions of Interest (ROI) without the need for applying masks. Our approach involves determining the indices for the first and last slices corresponding to the region of interest in each scan. Subsequently, we train the model on the entire  scan data, but confine the training process within the extracted ROI.




	Accuracy	AUC-ROC	f1-SCORE	RECALL	PRECISION
	train	valid	train	valid	train	valid	train	valid	train	valid
B	98.23%	98.56%	88.19%	90.49%	32.98%	35.54%	28.24%	44.17%	48.73%	42.08%
E	92.42%	91.41%	85.28%	87.92%	40.33%	43.11%	51.95%	60.94%	34.14%	35.64%
L	90.76%	90.33%	77.41%	80.63%	87.32%	86.95%	90.75%	90.33%	86.65%	86.27%
K	94.14	93.86%	74.68%	77.82%	91.71%	91.24%	94.13%	93.86%	91.41%	90.38%
S	90.64%	90.84%	75.27%	77.74%	87.04%	87.37%	90.64%	90.84%	86.19%	86.41%


Thanks for reading
