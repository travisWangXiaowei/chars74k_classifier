import numpy as np
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import hog


def getimgpaths(datapath):
    paths = []
    labels = []
    for dir in os.listdir(datapath):
        try:
            for filename in os.listdir(os.path.join(datapath, dir)):
                paths.append(os.path.join(datapath, dir, filename))
                label = dir[7:]
                if label[0] == '0' :
                    label = label[1];

                labels.append(int(label))
        except:
            pass

    return paths,labels


def load_image(filename):
    use_HOG = True
    path_mask = filename.replace('bmp', 'msk');
    img = cv2.imread(filename)
    mask = cv2.imread(path_mask,cv2.IMREAD_UNCHANGED);

    img_roi = cv2.bitwise_and(img,img, mask = mask)
    gray_image = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_image, tuple([32,32]))
    if use_HOG:
        HOG_img = pre_process_HOG(img)
        print(filename)
        return HOG_img
    else:
        return img

def pre_process_HOG(img):
    fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
             cells_per_block=(1, 1), visualize=False)
    print(fd.shape)
    return fd


def load_data(filepath):
    print("-------start pre-processing----------")
    list = []
    imgpaths, labels = getimgpaths(filepath)
    [list.append(load_image(img)/255) for img in imgpaths]
    X = np.stack(list, axis=0)
    y = np.asarray(labels)
    print("-------pre-processing done----------")
    return X, y


def pre_process_PCA(X, n_components):
    pca = PCA(n_components)
    pca.fit(X)
    X = pca.transform(X)
    return X


X, y = load_data('./data/bmp')



# nsamples, nx, ny = X.shape
#
#
# X = X.reshape((nsamples,nx*ny))
# X = pre_process_PCA(X, 200)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4
#                     ) #0.545415

mlp = MLPClassifier(hidden_layer_sizes=(600,250), max_iter=1000, alpha=5e-4,
                    solver='adam', verbose=True, tol=1e-4)
mlp.fit(X_train, y_train)
print("Train set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))







