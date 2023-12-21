#####################################################################################################################
# U-Net based salt body detection
# Reference of the original program :
#       Moon, H.,..., Jun, H.*, 2020, Comparison of convolutional neural networks for dividing seismic sequences,
#       Journal of the Korean Society of Mineral and Energy Resources Engineers., 57, 541-553. 
#       doi: 10.32390/ksmer.2020.57.6.541
#####################################################################################################################
from keras_unet_collection import models as kmodels
#from kears_unet_collection import utils as kutils

import matplotlib.pyplot as plt

import sys, gc, os
import glob

#from tensorflow.keras import backend as K
#from tensorflow.python import keras
from keras import models, backend
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, \
    UpSampling2D, BatchNormalization, Concatenate, Activation, Conv2DTranspose, \
    Add, GlobalAveragePooling2D,Dense,Reshape,Multiply

import re
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler

from tensorflow.keras import utils
import sys

import random

from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

#data read status bar
from tqdm import tqdm 
import tensorflow as tf
# backend.set_image_data_format('channels_first')


##convlution blocks
def bn_ac_conv(x, n_f, stride):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_f, (3, 3), strides=stride, padding = 'same')(x)
    return x

def bn_ac_conv_dilate(x, n_f, stride):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = Conv2D(n_f, (3, 3), strides=stride, padding = 'same')(x)
    x2 = Conv2D(n_f, (3, 3), strides=stride, dilation_rate=(3,3), padding = 'same')(x)
    x3 = Conv2D(n_f, (3, 3), strides=stride, dilation_rate=(5,5), padding = 'same')(x)
    x = Concatenate(axis=3)([x1, x2, x3])
    x = Conv2D(n_f, (1, 1), strides=stride, padding = 'same')(x)
    del x1, x2, x3
    return x


def bn_ac_conv1d(x, n_f, stride):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    squeeze = GlobalAveragePooling2D()(x)
    if x.shape[-1] < n_f:

        if stride == (1,1):
            excitation = Dense(n_f//4,activation='relu')(squeeze)
            excitation = Dense(n_f,activation='sigmoid')(excitation)
            excitation = Reshape((1,1,n_f))(excitation)
            print(f"excitation : {excitation.shape[-1]}")
        else:
            excitation = Dense(n_f//4,activation='relu')(squeeze)
            excitation = Dense(n_f//2,activation='sigmoid')(excitation)
            excitation = Reshape((1,1,n_f//2))(excitation)
            print(f"excitation : {excitation.shape[-1]}")
          
        scaled_feature_map = Multiply()([x,excitation])

        x = Conv2D(n_f, (1,1),strides=stride,padding='valid')(scaled_feature_map)
        return x
    else:
        if stride ==  (1,1):
            excitation = Dense(n_f//4,activation='relu')(squeeze)
            excitation = Dense(n_f*2,activation='sigmoid')(excitation)
            excitation = Reshape((1,1,n_f*2))(excitation)
            print(f"excitation : {excitation}")
        else:
            excitation = Dense(n_f//4,activation='relu')(squeeze)
            excitation = Dense(n_f,activation='sigmoid')(excitation)
            excitation = Reshape((1,1,n_f))(excitation)
            print(f"excitation : {excitation}")
        scaled = Multiply()([x,excitation])
        x = Conv2D(n_f,(1,1),strides = stride, padding='valid')(scaled)
        return x
        


def first_block(x, n_f):
    shortcut = x
    #x = bn_ac_conv(x, n_f, (1, 1))
    x = bn_ac_conv_dilate(x, n_f, (1, 1))
    x = Dropout(0.05)(x)
    x = bn_ac_conv(x, n_f, (1, 1))
    shortcut = bn_ac_conv1d(shortcut, n_f, (1, 1))
    x = Add()([x, shortcut])
    return x

def simple_block_dil(x, n_f):
    shortcut = x
    x = bn_ac_conv_dilate(x, n_f, (1, 1))
    x = Dropout(0.05)(x)
    x = bn_ac_conv(x, n_f, (1, 1))
    x = Add()([x, shortcut])
    return x

def simple_block(x, n_f):
    shortcut = x
    #x = Dropout(0.05)(x)
    x = bn_ac_conv(x, n_f, (1, 1))
    x = Add()([x, shortcut])
    return x

def down_block(x, n_f):
    #x = Conv2D(n_f, (3, 3), padding='same', dilation_rate=(10,4))(x)
    shortcut = x
    x = bn_ac_conv(x, n_f, (2, 2))
    x = Dropout(0.05)(x)
    x = bn_ac_conv(x, n_f, (1, 1))
    shortcut = bn_ac_conv1d(shortcut, n_f, (2, 2))
    x = Add()([x, shortcut])
    return x

def up_block(x, e, n_f):
    #x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(n_f, (3, 3), strides=(2,2), padding='same')(x)
    x = Concatenate(axis=3)([x, e])
    shortcut = x
    x = bn_ac_conv(x, n_f, (1, 1))
    #x = Dropout(0.05)(x)
    x = bn_ac_conv(x, n_f, (1, 1))
    shortcut = bn_ac_conv1d(shortcut, n_f, (1, 1))
    x = Add()([x, shortcut])
    return x

def conv_unet(x, n_f, mp_flag=True):
    x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def deconv_unet(x, e, n_f):
    ic = 3
    #x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(n_f, (3, 3), strides=(2,2), padding='same')(x)
    x = Concatenate(axis=ic)([x, e])
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def deconv_ae(x, e, n_f):
    ic = 3
    x = Conv2DTranspose(n_f, (3, 3), strides=(2,2), padding='same')(x)
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_f, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


##Network Architecture: R-Unet
def URES34NET(org_shape, iteration, n_ch, donly):
    i = 0
    # Input
    original = Input(shape=org_shape)

    # Encoding_layer1
    for ii in range(iteration[0]):
        if ii == 0:
            c1 = first_block(original, 16)
        else:
            c1 = simple_block_dil(c1, 16)
    concate1 = c1

    # Encoding_layer2
    for ii in range(iteration[1]):
        if ii == 0:
            c2 = down_block(c1, 32)
        else:
            c2 = simple_block_dil(c2, 32)
    concate2 = c2

    # Encoding_layer3
    for ii in range(iteration[2]):
        if ii == 0:
            c3 = down_block(c2, 64)
        else:
            c3 = simple_block_dil(c3, 64)
    concate3 = c3

    # Encoding_layer4
    for ii in range(iteration[3]):
        if ii == 0:
            c4 = down_block(c3, 128)
        else:
            c4 = simple_block_dil(c4, 128)
    concate4 = c4

    if donly:
        print("Down: RUnet34, Up: Unet")
        x = deconv_unet(c4, concate3, 64)
        x = deconv_unet(x, concate2, 32)
        x = deconv_unet(x, concate1, 16)
    
        decoded = Conv2D(n_ch, (3, 3), activation='sigmoid', padding='same')(x)

    else:
        print("Down: RUnet34, Up: RUnet34")
        # Decoding_layer1
        for ii in range(iteration[2]):
            if ii == 0:
                u1 = up_block(c4, concate3, 64)
                print(u1.shape)
            else:
                u1 = simple_block(u1, 64)
    
        # Decoding_layer2
        for ii in range(iteration[1]):
            if ii == 0:
                u2 = up_block(u1, concate2, 32)
            else:
                u2 = simple_block(u2, 32)
    
        # Decoding_layer3
        for ii in range(iteration[0]):
            if ii == 0:
                u3 = up_block(u2, concate1, 16)
            else:
                u3 = simple_block(u3, 16)
    
        decoded = Conv2D(n_ch, (3, 3), activation='softmax', padding='same')(u3)

    model = models.Model(inputs=original, outputs=decoded)

    return model

#AE
def AE(org_shape, n_ch):
    print("Down: Unet, Up: Unet")
    # Input
    original = Input(shape=org_shape)

    # Encoding
    c1 = conv_unet(original, 16, mp_flag=False)
    c2 = conv_unet(c1, 32)
    c3 = conv_unet(c2, 64)

    # Encoder
    encoded = conv_unet(c3, 128)

    # Decoding(original)
    x = deconv_ae(encoded, c3, 64)
    x = deconv_ae(x, c2, 32)
    x = deconv_ae(x, c1, 16)

    decoded = Conv2D(n_ch, (3, 3), activation='softmax', padding='same')(x)
    model = models.Model(inputs=original, outputs=decoded)

    return model


##Network Architecture: Unet
def UNET(org_shape, n_ch):
    print("Down: Unet, Up: Unet")
    # Input
    original = Input(shape=org_shape)

    # Encoding
    c1 = conv_unet(original, 16, mp_flag=False)
    c2 = conv_unet(c1, 32)
    c3 = conv_unet(c2, 64)

    # Encoder
    encoded = conv_unet(c3, 128)

    # Decoding(original)
    x = deconv_unet(encoded, c3, 64)
    x = deconv_unet(x, c2, 32)
    x = deconv_unet(x, c1, 16)

    decoded = Conv2D(n_ch, (3, 3), activation='softmax', padding='same')(x)
    model = models.Model(inputs=original, outputs=decoded)

    return model

###########################
# load data
###########################
def input_files(file_name):
    fin = open(file_name,"rb")
    patch = np.fromfile(fin,dtype='float32')
    fin.close()
    return patch


##load data and make training data array(binary file)
def load_train_data(data_dir,n1,n2,n_ch,datatype):
    ntrain = 0
    for idata in range(len(data_dir)):
        if idata == 0:
            file_list1 = glob.glob(data_dir[idata]+'image*.bin') #get image file name 
            file_list2 = glob.glob(data_dir[idata]+'mask*.bin')  #get mask file name 
            #print(file_list1)
            print(len(file_list1))

        else:
            file_list3 = glob.glob(data_dir[idata]+'image*.bin') #get image file name 
            file_list4 = glob.glob(data_dir[idata]+'mask*.bin')  #get mask file name 

            file_list1.extend(file_list3)
            file_list2.extend(file_list4)
            #print(file_list1)
            print(len(file_list1))



    print('number of train data:',len(file_list1))
    file_list1 = sorted(file_list1)
    file_list2 = sorted(file_list2)

    if len(file_list1) != len(file_list2):
       print("ERROR: The number of shot and mask file is different")
       sys.exit(1)
   
    x_train = []
    y_train = []
    for ii in tqdm(range(len(file_list1))):
        patch1 = input_files(file_list1[ii])
        patch1 = patch1.reshape(-1,n1)
    
        patch2 = input_files(file_list2[ii])
        patch2 = patch2.reshape(-1,n1)

        x_train.append(patch1)
        y_train.append(patch2)

        ntrain = ntrain + 1

        del patch1
        del patch2
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.reshape(-1,n2,n1,1)
    y_train = y_train.reshape(-1,n2,n1,1)
    y_train = utils.to_categorical(y_train, num_classes=n_ch)
    print("shape of train data:", x_train.shape,y_train.shape)
    print("ntrain: %d" %ntrain)

    gc.collect()
    return x_train, y_train


##read training data
class DATA_read():
    def __init__(self, in_ch, data_dir, test_dir, n1, n2):
        print(data_dir,"n1:",n1,"n2:",n2,"in_ch:",in_ch)
        n_ch = 2
        in_ch = 1
        x_train, y_train = load_train_data(data_dir,n1,n2,n_ch,'train')
        x_test, y_test = load_train_data(test_dir,n1,n2,n_ch,'test')

        input_shape = (n2, n1, in_ch)

        self.input_shape = input_shape
        self.x_train_in, self.x_train_out = x_train, y_train
        self.x_test_in, self.x_test_out = x_test, y_test
        self.n_ch = n_ch
        self.in_ch = in_ch

##find last updated model to resume training
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

###########################
# UNET verification
###########################
import matplotlib.pyplot as plt


###########################
# UNET test
###########################
import numpy as np
from sklearn.preprocessing import minmax_scale

##Apply trained model to test data, write test results
def write_results(data, model):
    x_test_in = data.x_test_in
    x_test_out = data.x_test_out
    decoded_imgs_org = model.predict(x_test_in)
    decoded_imgs = decoded_imgs_org

#    x_test_in = x_test_in[..., 0]

    n = 10

    print("Validate")
    print("n_test:%d"%(x_test_in.shape[0]))
    os.system("mkdir -p test_out")

    for jj in range(x_test_in.shape[0]):
        xxx = np.float32(x_test_in[jj,:,:,0])
        yyy = decoded_imgs[jj,:,:,:]
        yyy_arg = np.float32(np.argmax(yyy,axis=2))
        zzz = x_test_out[jj,:,:,:]
        zzz_arg = np.float32(np.argmax(zzz,axis=2))

        fout1 = open("./test_out/x_test_in_%04d.bin"%jj,"wb")
        fout2 = open("./test_out/x_pred_out_%04d.bin"%jj,"wb")
        fout3 = open("./test_out/x_test_out_%04d.bin"%jj,"wb")

        xxx.tofile(fout1)
        yyy_arg.tofile(fout2)
        zzz_arg.tofile(fout3)
        fout1.close()
        fout2.close()
        fout3.close()

def main(in_ch=1, epochs=2, batch_size=32, fig=True):
    ###########################
    # learn
    ###########################

    n1 = 96
    n2 = 96
    ndata = 1
    ntest = 1

    #data location (folder)
    data1 = "/data/knu2023/2023/0.DATA/train/"
    test1 = "/data/knu2023/2023/0.DATA/test/"
    data_dir = [data1]
    test_dir = [test1]

    data = DATA_read(in_ch=in_ch,data_dir=data_dir,test_dir=test_dir,n1=n1,n2=n2)
    print(data.input_shape, data.x_train_in.shape, data.x_train_out.shape)

    train_data_out = 0
    if(train_data_out):
        for jj in range(data.x_train_in.shape[0]):
            yyy = np.float32(data.x_train_out[jj,:,:,0])
            xxx = np.float32(data.x_train_in[jj,:,:,0])
    
            fout1 = open("./test_out/x_train_out_%04d.bin"%jj,"wb")
            fout2 = open("./test_out/x_train_in_%04d.bin"%jj,"wb")
    
            yyy.tofile(fout1)
            xxx.tofile(fout2)
            fout1.close()
            fout2.close()

    data.input_shape=(None,None,in_ch)
    iteration = [1,1,1,1]

    if args.structure == 'runet':
        donly = 0 #donly=0: use ures34 up and down
        model = URES34NET(data.input_shape, iteration, data.n_ch, donly)
    elif args.structure == 'ae':
        model = AE(data.input_shape, data.n_ch)
    elif args.structure == 'unet':
        model = UNET(data.input_shape, data.n_ch)
    elif args.structure == 'coll_unet':
        model = kmodels.att_unet_2d((96, 96, 1), filter_num=[16, 32, 64], n_labels=2, 
                           stack_num_down=2, stack_num_up=2, activation='ReLU', 
                           atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone=None, weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='attunet')
    elif args.structure == 'coll_unet_plus':
        model = kmodels.unet_plus_2d(data.input_shape, filter_num=[16, 32, 64, 128], n_labels=2, 
                           stack_num_down=2, stack_num_up=2, activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone=None, weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='xnet')

    model.summary()

    save_dir = os.path.join('models',args.model)
    os.system("mkdir -p models")
    os.system("mkdir -p %s" %save_dir)

    #load the last model
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' %initial_epoch)
        model = models.load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)

    #complie model
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')

    #save model, write logs
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=1)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')

    #training
    history = model.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs,initial_epoch=initial_epoch,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=0.25,
                       callbacks=[checkpointer,csv_logger])

    #write test results
    if args.write:
        write_results(data, model)


if __name__ == '__main__':
    import argparse
    from distutils import util

    parser = argparse.ArgumentParser(description='UNET for Sparker Seismic data Muting')
    parser.add_argument('-input_channels', type=int, default=1, help='input channels (default: 1)')
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs (default: 10)')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size (default: 50)')
    parser.add_argument('-model', type=str, default='geun_SE_runet', help='model name to save (default: Unet)')
    parser.add_argument('-structure', type=str, default='runet', help='CNN structure (ae, unet, runet, coll_unet_plus)')
    parser.add_argument('-write', type=lambda x: bool(util.strtobool(x)), default=True, help='flag to show figures (default: True)')

    args = parser.parse_args()

    print("Aargs:", args)

    print("write result?:",args.write)
    main(args.input_channels, args.epochs, args.batch_size, args.write)
#####################################################################################################################
