from keras.models import Model,load_model
from keras.layers import Dense,Input
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def autoencoder_model(input_feat):
    encoding_dim = 100
    input_img = Input(shape=(512,))
    #build model
    #encoder layers
    encoded_1 = Dense(300,activation='relu',)(input_img)
    #encoded_2 = Dense(100,activation='relu')(encoded_1)
    #encoded_3 = Dense(50,activation='relu')(encoded_2)
    encoder_output = Dense(encoding_dim)(encoded_1)

    #decoder layers
    #decoded_1 = Dense(50, activation='relu')(encoder_output)
    #decoded_2 = Dense(100, activation='relu')(decoded_1)
    decoded_3 = Dense(300, activation='relu')(encoder_output)
    decoded = Dense(512, activation='tanh')(decoded_3)
    #construct the  autoencoder model
    autoencoder = Model(input= input_img,output = decoded)

    #compile
    autoencoder.compile(optimizer='adam',loss='mse')

    #construct the encoder
    encoder = Model(input = input_img,output= encoder_output)
    autoencoder.fit(input_feat,input_feat,
                    batch_size=256,
                    nb_epoch=20,
                    shuffle=True)
    encoder.save('autoencoder_model.h5')
    return input_feat

def use_ae_reduction(input_data):
    model = load_model('autoencoder_model.h5')
    return model.predict(input_data)
    #output_feat = encoder.predict(input_feat)
    #return output_feat

