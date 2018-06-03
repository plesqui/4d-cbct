# 4d-cbct
A deep convolutional neural network model (based on the 'U-Net') to enhance the image quality of 4-D Cone Beam CT

## Objective
In this project, inspired by the SPARE Challenge (http://sydney.edu.au/medicine/image-x/research/SPARE-Challenge.php), we are investigating the performance of deep learning models to improve the quality of 4-dimensional cone-beam CT images. In particular, we have implemented a deep-convolutional neural network based on the 'U-Net' architecture (Ronneberger et al 2015).

## The Model
![U-Net](https://github.com/plesqui/4d-cbct/blob/master/U-Net-architecture.png?raw=true "U-Net")

The figure above shows the architecture of the original 2-D U-Net that was implemented for image segmentation tasks (https://arxiv.org/abs/1505.04597). Our model contains the following modifications:

- We have replaced the Maxpooling layers by 2-D Convolutional layers.
- We have replaced the up-convolution layers by re-size (using nearest neighbours) + 2-D convolutions. This modification is intended to prevent the network from exibiting artifacts typical of deconvolutional layers. A very nice description of this problem can be found here: https://distill.pub/2016/deconv-checkerboard/. 
- We have replaced the ReLu activations by leaky ReLus.
