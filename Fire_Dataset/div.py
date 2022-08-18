import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

random.seed(108)

# Read JPEGImages and Annotations
JPEGImages = [os.path.join('JPEGImages', x) for x in os.listdir('JPEGImages')]
Annotations = [os.path.join('Annotations', x) for x in os.listdir('Annotations') if x[-3:] == "txt"]

JPEGImages.sort()
Annotations.sort()

return_code = os.system('mkdir JPEGImages/train JPEGImages/val JPEGImages/test Annotations/train Annotations/val Annotations/test')
print("return code:", return_code)

# Split the dataset into train-valid-test splits 
train_JPEGImages, val_JPEGImages, train_Annotations, val_Annotations = train_test_split(JPEGImages, Annotations, test_size = 0.2, random_state = 1)
val_JPEGImages, test_JPEGImages, val_Annotations, test_Annotations = train_test_split(val_JPEGImages, val_Annotations, test_size = 0.5, random_state = 1)


#Utility function to move JPEGImages
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_JPEGImages, 'JPEGImages/train')
move_files_to_folder(val_JPEGImages, 'JPEGImages/val/')
move_files_to_folder(test_JPEGImages, 'JPEGImages/test/')
move_files_to_folder(train_Annotations, 'Annotations/train/')
move_files_to_folder(val_Annotations, 'Annotations/val/')
move_files_to_folder(test_Annotations, 'Annotations/test/')

