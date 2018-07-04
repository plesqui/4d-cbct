# 4d-cbct
A deep convolutional neural network model (based on the 'U-Net') to enhance the image quality of 4-D Cone Beam CT

## Objective
In this project, inspired by the SPARE Challenge (http://sydney.edu.au/medicine/image-x/research/SPARE-Challenge.php), we are investigating the performance of deep learning models to improve the quality of 4-dimensional cone-beam CT images. In particular, we have implemented a deep-convolutional neural network based on the 'U-Net' architecture (Ronneberger et al 2015). The model presented here corresponds to our first prototype. 


## The Model
![U-Net](https://github.com/plesqui/4d-cbct/blob/master/U-Net-architecture.png?raw=true "U-Net")

The figure above shows the architecture of the original 2-D U-Net that was implemented for image segmentation tasks (https://arxiv.org/abs/1505.04597). Our model contains the following modifications:

- We have replaced the Maxpooling layers by 2-D Convolutional layers.
- We have replaced the up-convolution layers by re-size (using nearest neighbours) + 2-D convolutions. This modification is intended to prevent the network from exibiting artifacts typical of deconvolutional layers. A very nice description of this problem can be found here: https://distill.pub/2016/deconv-checkerboard/. 
- Our input/output corresponds to 448 x 448 cbct axial slices. 

## The Data
The data was provided by the SPARE Challenge. The SPARE challenge is led by Dr Andy Shieh and Prof Paul Keall at the ACRF Image X Institute, The University of Sydney. Collaborators who have contributed to the datasets include A/Prof Xun Jia, Miss Yesenia Gonzalez, and Mr Bin Li from the University of Texas Southwestern Medical Center, and Dr Simon Rit from the Creatis Medical Imaging Research Center.

The data consisted of 4-Dimensional cone-beam CT images of 12 patients acquired in 1 minute (sparse input data, suffering from high levels of noise and artifacts), and the corresponding high-quality images (complete output data). These data will be released to the public by the organizers of the challenge in the future.

## Preliminary Results (Prototype model)
![U-Net](https://github.com/plesqui/4d-cbct/blob/master/preliminary.JPG?raw=true "U-Net")

The figure above illustrates the performance of our prototype on images from the validation set. The top-row displays three cone-beam CT slices reconstructed from 1-minute scans (input data). The middle row shows the improvements made by our model (predictions). The bottom row shows the ground-truth (high-quality images).

# Quantitative assessment of the prototype performance
In deep learning applications to enhance image data, the mean-square-error loss function (applied on a pixel-by-pixel basis) is often used. However, different groups have shown that the selection of a loss function, more relevant to the imaging-task at hand, can greatly improve the overall performance of the model. For instance, Zhao et al 2015 proposed several alternatives to the mean-square-error loss function for de-noising, super-resolution, and JPEG artifacts removal. The authors proposed a loss function which is a combination of the mean-absolute-error and the structural similarity. Read the study here: https://arxiv.org/abs/1511.08861. 

Another very recent study by Taghanaki et al 2018 showed that a simple network with the proper loss function can outperform more complex architectures (e.g. networks with skip connections) in image segmentation tasks. Read their work here: https://arxiv.org/abs/1805.02798

In light of these results, we decided to investigate the following research question:
1) Using the U-Net architecture, what is the optimum loss-fuction for denoising and artifact removal of 4-D cone-beam CT images? 
To this end, we evaluated the performance of our prototype model with the following loss functions:
- Loss A: mean-squared error
- Loss B: mean-absolute error
- Loss C: structural similarity 
- Loss D: 0.75 * (mean-square error) + 0.25 * (structural similarity)

We assessed the performance of each trained version of our prototype model by evaluating multiple metrics (mean-square error, mean-absolute error, peak signal-to-noise ratio and structural similarity) on the test dataset (i.e., images of patients that were not shown during training). In particular, we computed these metrics on both the entire image of each patient and also within the patient body only. The patient body on each image was segmented using a region growing algorithm (available on the SimpleITK library for python. The code is available in my repository). The results are shown in the four figures below. Overall, we do observe an improvement in all the image quality metrics with respect to the initial 'un-enhanced' images (referred to as 'Original' in the figures). 

![per](https://github.com/plesqui/4d-cbct/blob/master/metrics_eval1.png?raw=true "Performance assessment")

![per2](https://github.com/plesqui/4d-cbct/blob/master/metrics_eval2.png?raw=true "Performance assessment")

# Future work
Our prototype was built to improve the quality of the reconstructed images. One limitation of this approach is that the performance of the model will depend on the quality/artifacts present on the input images. Such quality of inputs also is sensitive to the method applied to reconstruct the measured projection data. To overcome this limitation, and to generalize our model as much as possible, we are investigating the following research question:

2) Can we build a deep learning model that improves the quality of the measured projection data (i.e., the sinograms)? How does the performance of such model compares to the performance of our current prototype?
