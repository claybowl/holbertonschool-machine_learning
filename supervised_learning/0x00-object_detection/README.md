0x00. Object Detection
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 5
 Project will start Apr 7, 2023 12:00 AM, must end by Apr 25, 2023 12:00 AM
 was released at Apr 7, 2023 12:00 AM
 Manual QA review must be done (request it when you are done with the project)
 An auto review will be launched at the deadline

 Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is OpenCV and how do you use it?
What is object detection?
What is the Sliding Windows algorithm?
What is a single-shot detector?
What is the YOLO algorithm?
What is IOU and how do you calculate it?
What is non-max suppression?
What are anchor boxes?
What is mAP and how do you calculate it?
Requirements
Python Scripts
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
The length of your files will be tested using wc


Download and Use OpenCV 4.1.x
alexa@ubuntu-xenial:~$ pip install --user opencv-python
alexa@ubuntu-xenial:~$ python3
>>> import cv2
>>> cv2.__version__
'4.1.0'
