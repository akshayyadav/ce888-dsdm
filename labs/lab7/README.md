# CE888_2019

For Lab 7: 

Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

This repository shows how we can use transfer learning in keras with the example of training a 4 class classification model using VGG-16 pre-trained weights.The vgg-16 is the CNN models trained on more than a million images of 1000 different categories.

Please see the two examples 
1. TransferLearning_Example1.ipynb	(VGG-16 model as feature extractor)
2. TransferLearning_Exercise2.ipynb (VGG-16 model as fine tuning)

In "TransferLearning_Example1.ipynb", we are taking a new dataset and it is similar to original dataset. Since the data is average size, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a classifier on the CNN codes for 4 classes only.


In "TransferLearning_Example2.ipynb", we are taking same dataset as above. Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.

Now, please download any images dataset, with 2 or more classes apply the same methods such as feature extractor and fine tuning.
