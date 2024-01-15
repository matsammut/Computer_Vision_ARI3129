import tensorflow as tf
import os

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

import matplotlib.pyplot as plt

resnet_model = ResNet50(weights='imagenet')
vgg_model = VGG16()
resnet_score = 0
vgg_score = 0
# cwd = os.getcwd()
# data_dir = os.path.join(cwd, 'dataset')
# images = []
# print(data_dir)

directory = 'dataset'
imgNum = 1
for filename in os.listdir(directory):

    print ("Image ", imgNum)
    imgNum+=1
    f = os.path.join(directory, filename)

    if os.path.isfile(f):

        img = image.load_img(f, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        #resnet
        resnet_pred = resnet_model.predict(x)
        resnet_dec = decode_predictions(resnet_pred, top=1)[0][0][1]
        print('Predicted by Resnet:', decode_predictions(resnet_pred, top=1))
        if(resnet_dec == "crane"):
            resnet_score += 1

        # vgg
        vgg_pred = vgg_model.predict(x)
        vgg_dec = decode_predictions(vgg_pred)
        vgg_dec = vgg_dec[0][0]
        print('Predicted by VGG: %s (%.2f%%)' % (vgg_dec[1], vgg_dec[2]*100))
        if(vgg_dec[1] == "crane"):
            vgg_score += 1

        print ("\n\n")



print ("Resnet Score: ",resnet_score)
print ("VGG Score: ", vgg_score)


# Precisicion  (MAP)
resnet_map = resnet_score/imgNum
vgg_map = vgg_score/imgNum

resnet_prec = round(resnet_map*100)
vgg_prec = round(vgg_map*100)
print ("Resnet Precision: ",resnet_map, "(", resnet_prec, "%)")
print ("VGG Precision: ", vgg_map, "(", round(vgg_map*100), "%)")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
cnn = ['Resnet50', 'VGG16']
scores = [resnet_prec, vgg_prec]
ax.bar(cnn, scores)
plt.show()