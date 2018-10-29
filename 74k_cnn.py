import time 
import numpy as np 
import pandas as pd from sklearn.cross_validation 
import train_test_split from sklearn.preprocessing 
import LabelBinarizer, LabelEncoder, MinMaxScaler from keras.regularizers 
import l2 from keras.models 
import Sequential from keras.layers.core 
import Dense, Activation, Dropout, Flatten from keras.layers.convolutional 
import Convolution2D, MaxPooling2D from keras.layers.advanced_activations 
import LeakyReLU, PReLU from keras.utils 
import np_utils, generic_utils from keras.optimizers 
import SGD from random 
import randint, uniform 
import seaborn as sns from matplotlib 
import pyplot from skimage.io 
import imshow from skimage 
import transform, filters, exposure 
PIXELS = 64 
imageSize = PIXELS * PIXELS 
num_features = imageSize 
label_enc = LabelBinarizer() 
BATCHSIZE = 128 
def fast_warp(img, tf, output_shape, mode='constant'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode) 
def batch_iterator(data, y, batchsize, model):
    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS), dtype = 'float32')
        dorotate = randint(-10,10)
        trans_1 = randint(-10,10)
        trans_2 = randint(-10,10)
        zoom = uniform(1, 1.3)
        shear_deg = uniform(-25, 25)
        center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)
        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))
        tform = tform_center + tform_aug + tform_uncenter
        for j in range(X_batch.shape[0]):
            X_batch_aug[j][0] = fast_warp(X_batch[j][0], tform,
                                          output_shape = (PIXELS, PIXELS))
        indices_sobel = np.random.choice(int(X_batch_aug.shape[0]), int(X_batch_aug.shape[0] / 4), replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)
        indices_invert = np.random.choice(int(X_batch_aug.shape[0]), int(X_batch_aug.shape[0] / 2), replace = 
False)
        for l in indices_invert:
            img = X_batch_aug[l][0]
            X_batch_aug[l][0] = np.absolute(img - np.amax(img))
        loss.append(model.train_on_batch(X_batch_aug, y_batch))
    return np.mean(loss) 
def load_data_cv(train_path):
    print('Read data')
    training = np.load(train_path)
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    training_inputs = training[:,0:num_features].astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(training_inputs, training_targets)
    print ('train size:', x_train.shape[0], 'eval size:', x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], 1, PIXELS, PIXELS)
    x_test = x_test.reshape(x_test.shape[0], 1, PIXELS, PIXELS)
    return x_train, x_test, y_train, y_test 
def load_data_test(train_path, test_path):
    print('Read data')
    training = np.load(train_path)
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    training_inputs = training[:,0:num_features].astype('float32')
    testing_inputs = np.load(test_path).astype('float32')
    training_inputs = training_inputs.reshape(training_inputs.shape[0], 1, PIXELS, PIXELS)
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 1, PIXELS, PIXELS)
    return training_inputs, training_targets, testing_inputs 
def build_model():
    #using VGG
    
    print('Model is getting created ...')
    model = Sequential()
    model.add(Convolution2D(128,(3,2), input_shape=(1,PIXELS, PIXELS), activation = 'relu', 
data_format='channels_first'))
    model.add(Convolution2D(128,(3,2), activation = 'relu') )
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))
    model.add(Convolution2D(256,(3,2), activation = 'relu'))
    model.add(Convolution2D(256,(3,2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))
    model.add(Convolution2D(512,3,strides=(3,2), activation = 'relu'))
    model.add(Convolution2D(512,3,strides=(3,2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(62))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    print('Model is created !')
    return model 
def main():
    x_train, x_test, y_train, y_test = load_data_cv('./train_64.npy')
    model = build_model()
    print("Starting training ...")
    train_loss = []
    valid_loss = []
    valid_acc = []
    try:
        for i in range(1,275):
            print(i)
            if i == 250:
                model.optimizer.lr.set_value(0.003)
            if i == 275:
                model.optimizer.lr.set_value(0.0003)
            start = time.time()
            loss = batch_iterator(x_train, y_train, BATCHSIZE, model)
            train_loss.append(loss)
            valid_avg = model.evaluate(x_test, y_test, metrics=['accuracy'], verbose = 0)
            valid_loss.append(valid_avg[0])
            valid_acc.append(valid_avg[1])
            end = time.time() - start
            print ('iter:', i, '| Tloss:', np.round(loss, decimals = 3))
    except KeyboardInterrupt:
        pass
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)
    sns.set_style("whitegrid")
    pyplot.plot(train_loss, linewidth = 3, label = 'train loss')
    pyplot.plot(valid_loss, linewidth = 3, label = 'valid loss')
    pyplot.legend(loc = 2)
    pyplot.ylim([0,4.5])
    pyplot.twinx()
    pyplot.plot(valid_acc, linewidth = 3, label = 'valid accuracy', color = 'r')
    pyplot.grid()
    pyplot.ylim([0,1])
    pyplot.legend(loc = 1)
    pyplot.savefig('./training_plot.png')
    
if __name__ == '__main__':
    main()
