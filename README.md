# Shelter_Dog_Posture_Classifier
 This repository holds the experimental implementation of a data processing and model training and testing pipline focused on making images with disrupted foregrounds useable and less noisy to achive better information extraction.

Notes:

dog_detection_cropping.ipynb:
    when cropping to dog using function:
            'find_largest_dog(boxes)'
    there is an issue where the set tested when choosing the 'largest box' is not associated exclusivly with dog objects.  
=======
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