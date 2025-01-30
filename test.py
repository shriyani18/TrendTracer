import pickle
import tensorflow
import sklearn
import numpy as np
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
#for comaparing features of new image with other
from sklearn.neighbors import NearestNeighbors
import cv2 #for displaying images


#loading feature list and convert it into array
feature_list=np.array(pickle.load(open('embedings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))


model=ResNet50(weights='imagenet', include_top=False, input_shape= (224, 224, 3))
model.trainable = False  # here we use the model

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#calculating features of sample image
img=image.load_img('samples/10001.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img= preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result=result / norm(result)# vector containing extracted features

#comparing features
neighbors=NearestNeighbors(n_neighbors=4,algorithm='brute',metric='euclidean')#algorithm used is brute first search
neighbors.fit(feature_list)

from PIL import Image
import cv2

#gives indices of nearest image
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

# for displaying images
for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    temp_img = cv2.resize(temp_img, (510, 510))
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(temp_img)
    img.show()
























