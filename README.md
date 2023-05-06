# Happy Whale Competition: CS529 Project 3

#### By Michael Adams and Jack Ringer

In this project we implement various methodologies to try and classify whales
based on their fluke. This code was used for this Kaggle competition:
https://www.kaggle.com/competitions/cs429529-humpback-whale-identification-challenge/overview

## Project Guide

All code files used can be found in the src/ directory.

An overview of what is implemented in each of these files is given as follows:

* bounding_boxes.py: retrieving and visualizing bounding boxes
* calculate_image_stats.py: calculate means / stds of images
* data_exploration.py: misc data plotting/visualization
* evaluation.py: code used to generate submission files for Kaggle
* non_ml_predictions.py: generating predictions using non-ML methods
* train.py: code to train a classifier model
* train_arcface.py: code to train an arcface model
* train_siamese.py: code to train a siamese network

The src/models subdirectory contains PyTorch implementations of the different
models we tested. The src/utils subdirectory
contains various utility functions, including augmentations and the whale
dataset setup. For more information see report.

## Setup

All necessary packages to run the code in this project can be installed with:

pip install -r requirements.txt