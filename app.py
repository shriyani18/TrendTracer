import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#for normalize the result we uses linear algebra
from numpy.linalg import norm
import os
from tqdm import tqdm  #shows for loops progress
import pickle

# creating resnet model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False  # here we use the model

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  #we have add our own top layer by removing top layer of image
])

#print(model.summary())  #printing the summary of model


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  #loading the image
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)  # vector containing extracted features

    return normalized_result


print(os.listdir('images'))  #name of all 44000 images-++

#we want complete path name of all the images and put it in list
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))  #it joins file path from images directory

    #prints length
print(len(filenames))
#gives path of first 5 images
print(filenames[0:5])

#feature list contains
# [[2048 features for each image],
# [],
# []....,
# [44kimages]]
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

print(np.array(feature_list).shape)

#extract features in file and write in binary mode
pickle.dump(feature_list, open('embedings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))  #names of the files
