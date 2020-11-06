import socket
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import time

model = tf.keras.models.load_model('HAR_model')

UDP_IP = '192.168.1.123'
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, # Internet
          socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

fileob = open('pickled_label_encoder.obj','rb')
label_encoder = pickle.load(fileob)

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes

    decoded_data = np.frombuffer(data,dtype='f2').reshape((1,40,6))
    predicted_label = model.predict([decoded_data[:,:,0],decoded_data[:,:,1],decoded_data[:,:,2],
    decoded_data[:,:,3],decoded_data[:,:,4],decoded_data[:,:,5]])
    label = label_encoder.inverse_transform([np.argmax(predicted_label)])[0]
    print(label)
