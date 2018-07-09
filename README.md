[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./writeup/german-shepherd-solo.png "VGG16 Model Figure"


#### Udacity Artificial Intelligence Nanodegree
### Term 2: Project 5
# Classifying Dog Breeds Using CNNs & PyTorch

##### &nbsp;

<img src="writeup/dog-breeds-header.jpg" width="100%" />

*Image credit: [pethelpful.com](https://pethelpful.com/dogs/Tips-on-Choosing-the-Right-Dog-Breed)*


##### &nbsp;
## Goal & Overview
Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

<img src="writeup/german-shepherd-solo.png" width="50%" alt=""  title="Sample Output" />

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

##### &nbsp;
## Approach
*coming soon*


##### &nbsp;
## Results
*coming soon*



##### &nbsp;
---

# Project Starter Code
In case you want to run this project yourself, below is the project starter code. [This repository](https://github.com/udacity/dog-project) contains all the code needed.

##### &nbsp;
## Setup Instructions
These instructions have been modified for running the project on AWS, with specific recommendations for PyTorch. Udacity's original setup instructions can be found [here](https://github.com/udacity/dog-project#project-instructions).

##### &nbsp;

1. Create an Amazon Web Services EC2 instance. I recommend [Deep Learning AMI (Ubuntu) Version 10.0 - ami-e580c79d](https://aws.amazon.com/marketplace/pp/B077GCH38C), which comes with most of the packages you'll need already pre-installed.

	• Your instance needs at least 50 GB of GPU memory to run this project in PyTorch. In my limited experience, PyTorch is much more memory intensive than Tensorflow. The smaller p2.xlarge EC2 instance with 12 GB of GPU memory worked fine for my initial implementation of the project in Keras+Tensorflow. However, this same instance kept encountering 'out of memory' errors when running the project in PyTorch. Once I switched to a p2.8xlarge instance (96 GB GPU memory), the project ran smoothly &mdash; but, keep in mind that this larger instance is much more costly, ~$7/hr vs $1/hr. You can compare the different instance types yourself [here](https://aws.amazon.com/ec2/instance-types/) (the P2 and P3 instances are found under "Accelerated Computing"). Again, if you're running the project on Tensorflow, the smaller p2.xlarge EC2 instance is fine.

	• When setting-up your instance, remember to open port 8888 for Jupyter Notebook in your [security group settings](https://www.evernote.com/l/ABdh1MljZRRFPKBZEsh0XH-oBMd28_J-yfs).

2. Start your EC2 instance via the console, then login via terminal.
```
ssh -i <path to key> ubuntu@<IPv4 Public IP address>
```

3. Activate your PyTorch environment for Python 3.6.
```
source activate pytorch_p36
```
Or, if you're using tensorflow, you'd enter `source activate tensorflow_p36`. A complete list of environments is available [here](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-conda.html#tutorial-conda-login).

4. Install OpenCV
```
conda install -c conda-forge opencv
```

5. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

6. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Then unzip the folder. Make sure it's located at `path/to/dog-project/dogImages`. If you are using a Windows machine, you may want to use [7zip](http://www.7-zip.org/) to extract the folder.
```
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
```

7. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder, makig sure it's at location `path/to/dog-project/lfw`.  
```
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
unzip lfw.zip
```

8. Open Jupyter Notebook via your AWS terminal.
```
jupyter notebook --ip=0.0.0.0 --no-browser
```
	Then, to open Jupyter in your browser, start by copying the URL provided in your terminal window. It should look something like this: `http://ip-170-35-87-127:8888/?token=eqith4949181huhfkjqdhfh1948`

	Paste this URL into your browser and replace the `http://ip-170-35-87-127` portion to the left of the colon ":" with the IP address for your EC2 instance.

9. Before running code, verify that the kernel matches your Conda environment. If you need to change the kernel, go to the drop-down menu (Kernel > Change kernel). Then, you can start running code cells in the notebook.
