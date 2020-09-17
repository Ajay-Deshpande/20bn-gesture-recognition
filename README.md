# 20bn-gesture-recognition

The dataset for this project has been downloaded from https://20bn.com/datasets/jester/v1 website. The data consists of 27 classes with around 2.5k images per class. The project considers only 8 classes to accomodate for the available resources. The dataset provides csv's (present inside annotations) which contains a mapping between folder_name and gesture class. Each folder contains a sequence of 12-68 images.
The trainer script consists of a data generator which inherits the keras data generator to provide data to the model in the required format. The generator yields data to the model in the shape of (batch_size,frames_count,image_height,image_width). The model architecture is defined in the trainer script. After training the model is stored in two files - the architecture in a json file and the weights in a h5 file.
The predictor script opens the webcam and passes the specified number frames to get a prediction from the model. The prediction is shown on the interface.
The accuracy check scripts creates a heatmap of the confusion matrix and it can be used to verify how good the model is.
