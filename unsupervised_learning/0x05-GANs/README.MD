Description

This course is designed to use a novel unsupervised learning technique where a learning model generates new data sets that are judged on its authenticity of another learning model. GANs in their original inceptions can be thought of as a more advanced monte carlo hidden markov model.

Students are introduced to the concepts and are tasked with making a simple machine learning model, with random noise as a data set, while using PyTorch to train the neural network.

PyTorch is another open source deep learning library. It is an optimized library, based on the tensor representations, for deep learning using GPUs and CPUs. It is developed by Facebook’s AI Research lab. The project was also merged with Caffe, another deep learning framework originally developed at the University of California Berkeley.

It is currently as popular as TensorFlow by the amount of academic papers being published with its use. It has also gained popularity through Google searches.

The normal distribution is quintessential to neural networks and machine learning because it allows the data and the math to be simplified, or smoothed, in order to process the simulation in a more robust fashion.

A GAN can generate a normal distribution from a random sample data set. This process actually proves it’s possible to have a GAN generate fake statistical data that could be presented as something real. This is actually the first function of a GAN from its initial published academic paper.
Learning Objectives:

At the end of this project you are expected you are expected to be able to explain without the help of Google:
General

    What is PyTorch?
    What is a generator?
    What is a discriminator?
    What is the minimax loss? modified minimax loss? wasserstein loss?
    How do you train a GAN?
    What are the use cases for GANs?
    What are the short comings of GANs?
    What is Energy based learning?
    What are the gaussian and uniform distributions?
    What is statistical noise?
    What are the moments of a distribution?

Requirements
General

    All your files will be interpreted/compiled on Ubuntu (version 18.04 LTS or greater) using python3 (version 3.6 or greater)
    Your files will be executed with numpy (version 1.17+), PyTorch (version 1.8+), and Plotly.
    All your files should end with a new line
    The first line of all your files should be exactly #!/usr/bin/env python3
    A README.md file, at the root of the folder of the project, is mandatory
    Your code should use the pycodestyle style (version 2.4)
    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
    Unless otherwise noted, you are not allowed to import any module except the libraries from torch, import numpy as np, as needed
    All your files must be executable

More Info
Installing Latest version of Pytorch

Install for Cuda

Note: Should be used if you have a supported Nvidia or AMD GPU

student@holbie-ml:~/pytorch$ pip3 install torch torchvision torchaudio --user

Install for CPU

student@holbie-ml:~/pytorch$ pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --user

More information on how to install PyTorch can be found here: https://pytorch.org/get-started/locally/
Installing Numpy 1.17.0

Install Numpy: student@holbie-ml:~/pytorch$ pip3 install numpy==1.17.0 --user

More information on how to install Plotly can be found here: https://numpy.org/install/
Installing Plotly 5.5.0

Install Plotly: student@holbie-ml:~/pytorch$ pip3 install plotly==5.5.0 --user

More information on how to install Plotly can be found here: https://plotly.com/python/getting-started/