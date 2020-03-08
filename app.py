# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:57:40 2020

@author: pavansubhash_t
"""
import os
import sys
import cv2
from flask import Flask, redirect, url_for, request, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from tensorflow.keras import backend as K
from PIL import Image
import joblib
K.set_image_data_format('channels_first')

classes = ['Red','Grey','Green','Black','Beige','White','Wine','Purple','Blue','Orange','Brown','Pink','Turquoise','Yellow','Magenta','Golden','Plains And Textures','Stripes And Checks','Floral','Abstract','Small Motifs','Scroll','Geometric','Damask','Animal Print']

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(Activation('sigmoid'))

app = Flask(__name__)

model.load_weights('Multi_Label_Classification.hdf5')
best_threshold = joblib.load('best_th.pkl')

@app.route('/predict_ld', methods=['POST'])
def predict_ld():
    if request.method == 'POST':
        file =request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(filename)
        file.save(filepath)
        img = cv2.imread(filepath)
        os.remove(filepath)
        img = cv2.resize(img,(100,100))
        img = img.transpose((2,0,1))
        img = img.astype('float32')
        img = img/255
        img = np.expand_dims(img,axis=0)
        pred = model.predict_proba(img)
        y1_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])
        result = [classes[i] for i in range(25) if y1_pred[i]==1 ]
        
        return jsonify({"predictions":result})
    
    return None

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0'), app)
    http_server.serve_forever()        
