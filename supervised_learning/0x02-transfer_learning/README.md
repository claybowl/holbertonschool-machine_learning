# 0x02. Transfer Learning

By: Alexa Orrico, Software Engineer at Holberton School
Weight: 5
Project will start Mar 24, 2023 12:00 AM, must end by Apr 6, 2023 12:00 AM
Manual QA review must be done (request it when you are done with the project)

## Resources

Read or watch:

- [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
- [Transfer Learning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Transfer learning & fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning#fine-tuning)

Definitions to skim:

- [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)

References:

- [Keras Applications](https://keras.io/api/applications/)
- [Keras Datasets](https://keras.io/api/datasets/)
- [tf.keras.layers.Lambda](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda)
- [tf.keras.backend.resize_images](https://www.tensorflow.org/api_docs/python/tf/keras/backend/resize_images)
- [A Survey on Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

- What is a transfer learning?
- What is fine-tuning?
- What is a frozen layer? How and why do you freeze a layer?
- How to use transfer learning with Keras applications

## Requirements

### General

- Allowed editors: vi, vim, emacs
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
- Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.4)
- All your modules should have documentation (python3 -c 'print(**import**("my_module").**doc**)')
- All your classes should have documentation (python3 -c 'print(**import**("my_module").MyClass.**doc**)')
- All your functions (inside and outside a class) should have documentation (python3 -c 'print(**import**("my_module").my_function.**doc**)' and python3 -c 'print(**import**("my_module").MyClass.my_function.**doc**)')
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as K`
- All your files must be executable
- The length of your files will be tested using wc
