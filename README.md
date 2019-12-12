# Image Classifier 
Data scientist nanodegree second project(Deep Learning)

### Table of Contents
1. [Installation](https://github.com/rawan231/Image-Classifier#Installation)
2. [Project Inroduction](https://github.com/rawan231/Image-Classifier#Project-Introduction)
3. [File Descriptions](https://github.com/rawan231/Image-Classifier#File-Descriptions)
4. [Run](https://github.com/rawan231/Image-Classifier#Run)
5. [License](https://github.com/rawan231/Image-Classifier#License)

### Installation

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Jupyter-Notbook](https://jupyter.org/install.html)
- [PyTorch](https://pytorch.org/)

### Project Introduction

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
In this project, I have trained an image classifier to recognize different species of flowers (102 categories). Then, I wrote a Python program that can run from the command line to classify images.


### File Descriptions
- `Image Classifier Project.ipynb` is the Jupyter notebook where I implemented the image classifier.
- `Image Classifier Project.html` the notebook saved as html file.
- `cat_to_name.json` contains dictionary that matches between flower categories numbers and names.
- `train.py` is the Python program that train the image classifier on flowers data, and `train_functions.py` contains functions implementations.
- `predict.py` is the Python program that predict new flower's category using the trained classifier, and `predict_functions.py' contains functions implementations.
- `utils.py` contains helper functions for the training process.


### Run
* Jupyter notebook
In a terminal or command window, navigate to the top-level project directory `Image-Classifier`,  and run the following command:
```
jupyter notebook Image Classifier Project.ipynb
```

* Python program   
* Train the classifier
In a terminal or command window, navigate to the top-level project directory `Image-Classifier`,  and run the following command:
```
Python train.py
```
* Classifiy an image 
In a terminal or command window, navigate to the top-level project directory `Image-Classifier`,  and run the following command:
```
Python predict.py
```

### License
This project is licensed under the MIT License - see the LICENSE file for details


