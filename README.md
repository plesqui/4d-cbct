# 4d-cbct
A deep convolutional neural network model (based on the 'U-Net') to enhance the image quality of 4-D Cone Beam CT

## Objective
In this project, inspired by the SPARE Challenge (http://sydney.edu.au/medicine/image-x/research/SPARE-Challenge.php), we are investigating the performance of deep learning models to improve the quality of 4-dimensional cone-beam CT images. In particular, we have implemented a deep-convolutional neural network based on the 'U-Net' architecture (Ronneberger et al 2015). The model presented here corresponds to our first prototype. 


## The Model
![U-Net](https://github.com/plesqui/4d-cbct/blob/master/U-Net-architecture.png?raw=true "U-Net")

The figure above shows the architecture of the original 2-D U-Net that was implemented for image segmentation tasks (https://arxiv.org/abs/1505.04597). Our model contains the following modifications:

- We have replaced the Maxpooling layers by 2-D Convolutional layers.
- We have replaced the up-convolution layers by re-size (using nearest neighbours) + 2-D convolutions. This modification is intended to prevent the network from exibiting artifacts typical of deconvolutional layers. A very nice description of this problem can be found here: https://distill.pub/2016/deconv-checkerboard/. 
- We have replaced the ReLu activations by leaky ReLus.
- Our input/output corresponds to 448 x 448 cbct axial slices. 

## The Data
The data was provided by the SPARE Challenge. The SPARE challenge is led by Dr Andy Shieh and Prof Paul Keall at the ACRF Image X Institute, The University of Sydney. Collaborators who have contributed to the datasets include A/Prof Xun Jia, Miss Yesenia Gonzalez, and Mr Bin Li from the University of Texas Southwestern Medical Center, and Dr Simon Rit from the Creatis Medical Imaging Research Center.

The data consisted of 4-Dimensional cone-beam CT images of 12 patients acquired in 1 minute (sparse input data, suffering from high levels of noise and artifacts), and the corresponding high-quality images (complete output data). These data will be released to the public by the organizers of the challenge in the future.

# Preliminary Results
![U-Net](https://github.com/plesqui/4d-cbct/blob/master/preliminary.JPG?raw=true "U-Net")

The figure above illustrates the performance of our prototype on images from the validation set. The top-row displays three cone-beam CT slices reconstructed from 1-minute scans (input data). The middle row shows the improvements made by our model (predictions). The bottom row shows the ground-truth (high-quality images).

# Current and future work
Currently, we are working on the following research questions:
1) What is the optimum loss-fuction? To this end, we are evaluating the performance of our model with the mean-squared error, the absolute error and the structural similarity index metric as loss-functions.

2) Our prototype was built to improve the quality of the reconstructed images. Can we build a deep learning model that improves the measured projection data instead (i.e., the sinograms)? How does the performance of such model compares to the performance of our current prototype?
