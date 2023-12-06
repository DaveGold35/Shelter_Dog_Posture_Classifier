# Shelter_Dog_Posture_Classifier
 This repository holds the experimental implementation of a data processing and model training and testing pipline focused on making images with disrupted foregrounds useable and less noisy to achive better information extraction.

**Folder Structure Design**

When currating datasets to be augmented and trained using these notebooks, inherant datastructures were assumed and implemented in procedures.
For example:
    Video_to_frame.ipynb uses an _input path_ variable that needs to referance a folder of videos in the .mp4 format and an _output path_ variable for a destination folder 
    (which it would create if it doesnt exist)

Each notebook uses a similar input/output path variables, but importantly, it is best to sort your data for training based on posture after the first step/during the second step. 

Datasets used in trial notebooks can be found at:
    https://huggingface.co/datasets/davegold/ShelterDogsPostureClassification .
Trial4 dataset uses duplicate of trial3 data.
    
**Notebook use case & ordering:**

 1. Video_to_frame.ipynb                 :    to break videos down into frame samples
 2. dog_detection_cropping.ipynb         :    use a Resnet50 model to detect dogs, and crop frames to them with padding
 3. Sobel_inpainting.ipynb               :    take cropped images, and inpaint the bars in-front of dog images
 4. Custom_datasets.ipynb                :    split dataset into training, validation, and testing sets
 5. training V2.ipynb                    :    set parameters and train a Resnet 18 model on dataset
 6. testing.ipynb                        :    evaluate trained models using a confusion matrix

<img width="800" alt="Screenshot 2023-12-06 at 9 58 08 AM" src="https://github.com/DaveGold35/Shelter_Dog_Posture_Classifier/assets/139391616/bfda3d77-9dce-419c-bace-fd6f2fe3be0e">

**Notes:**

Notebooks are designed to be run using Python3.9

Main changes need to be made to **path variables**.

dog_detection_cropping.ipynb:
    when cropping to dog using function:
            'find_largest_dog(boxes)'
    there is an issue where the set tested when choosing the 'largest box' is not associated exclusivly with dog objects.  

    The following line addition should fix the issue
    
    def find_largest_box(boxes, labels): 
        #labels added as input
        largest_box = 0
        for i in range(len(boxes)):
            if labels[i] == 18: #number associated with dogs
                  box = boxes[i].detach().numpy()
                  if (box[2]-box[0])*(box[3]-box[1]) > largest_box:
                      largest_box = (box[2]-box[0])*(box[3]-box[1])
                      largest_box_coord = box

        return largest_box_coord

    function apply(img, boxes, labels, masks):
        This funciton is created to be able to understand and display what object box is being cropped to and used in order to debug or see a midstep in the process.  
        In order to use and see you must un-comment lines 15-18 in apply procedure cell.
Sobel_inpainting:
    If you use the procedures in this notebook without useing the pipeline function or 'iter_pip' function, you may run into a error where inpainting only occurs in the top 40x40 box of an image.  This is due to input variable in the _measure_between_lines()_ procedure that's default values are set to 40 for logic checks.  This leads to missing inpainting due to the output of _measure_between_lines()_ returning a variable 'points_with_gaps' that returns a list of gaps for filling in the _fill_between_lines()_ procedure passed as the variable 'set'.

