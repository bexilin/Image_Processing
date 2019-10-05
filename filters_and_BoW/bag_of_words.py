import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


folder = './images/'
categories = ['tractor', 'coral reef', 'banana', 'fountain']


def image_patches(im, patch_size=(16,16)):
    # Returns image patches from the image  
    # Input- im: H x W x C
    #        patch_size: U X V 
    # Output- patches: 16 x 768 
    xnumber = int(np.shape(im)[0]/16)
    ynumber = int(np.shape(im)[1]/16)
    R,G,B = im[:,:,0],im[:,:,1],im[:,:,2]
    R_all = np.array([R[16*i:16*(i+1),16*j:16*(j+1)] for i in range(xnumber) for j in range(ynumber)])
    G_all = np.array([G[16*i:16*(i+1),16*j:16*(j+1)] for i in range(xnumber) for j in range(ynumber)])
    B_all = np.array([B[16*i:16*(i+1),16*j:16*(j+1)] for i in range(xnumber) for j in range(ynumber)])
    R_all, G_all, B_all = np.reshape(R_all,(16,16*16)), np.reshape(G_all,(16,16*16)), np.reshape(B_all,(16,16*16))
    patches = np.hstack((R_all,G_all,B_all))
    return patches


def build_codebook(X_Train, num_clusters=15):
    # Returns a KMeans object fit to the dataset  
    # Input- X_train: (3*N/4 * M) x P
    #        num_clusters: scalar 
    # Output- KMeans: object
    codebook = KMeans(n_clusters=num_clusters).fit(X_Train)
    return codebook


def normalize_and_split(X, y):
    # Returns the normalized, split dataset in patches
    # N = num of samples
    # M = num of patches per image
    # P = size of patch flattened to a vector
    # Input- X: N x M x P 
    #        y: N x 1
    # Output- X_train: (3*N/4 * M) x P
    #         X_test: (N/4 * M) x 1
    #         y_train: 3*N/4 x 1
    #         y_test: N/4 x 1
    X = np.asarray(X)
    patch = X[0] # Need to fetch the patch size.. should be (16, 768)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=42)

    # Shape it so it is of size (Num of Patches x Size of Patch
    X_train = X_train.reshape(-1, patch.shape[1])
    X_test = X_test.reshape(-1, patch.shape[1])

    # After building X, it must be normalized.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_classifier(X_hist_train, X_hist_test, y_train, y_test):
    # Trains a classifier and evaluates it
    # Input- X_hist_train: 1500 x 15 
    #        X_hist_test: 500 x 15
    #        y_train: 1500 x 1
    #        y_test: 500 x 1
    # Output- clf: classifier object 
    #         score: scalar 
    clf = SVC()

    clf.fit(X_hist_train, y_train)
    score = clf.score(X_hist_test, y_test)
    print("Validation Performance: {}".format(str(score)))
    return clf, score

def build_histogram(labels):
    labels = np.reshape(labels,(int(len(labels)/16),16))
    histogram = np.zeros((np.shape(labels)[0],15))
    for i in range(np.shape(labels)[0]):
        histogram[i,:]=np.histogram(labels[i,:],bins=15,range=(-0.5,14.5))[0]
    return histogram


def main():
    X = []
    y = []
    paths = []

    # Iterate over the images in `images/` 
    paths += [('./images/banana/'+f,'banana') for f in os.listdir('./images/banana/')]
    paths += [('./images/coral_reef/'+f,'coral_reef') for f in os.listdir('./images/coral_reef/')]
    paths += [('./images/fountain/'+f,'fountain') for f in os.listdir('./images/fountain/')]
    paths += [('./images/tractor/'+f,'tractor') for f in os.listdir('./images/tractor/')]

    # Extract the patches from each of them and them to X and the class labels to y
    for p in paths:
        image = imread(p[0])
        if len(np.shape(image)) < 3:
            continue
        patches = image_patches(image)
        X.append(patches)
        y.append(p[1])

    # Normalize and split image patches
    X_train, X_test, y_train, y_test = normalize_and_split(X, y)

    # Build the codebook with X_train
    codebook = build_codebook(X_train)
    labels_train = codebook.predict(X_train)

    # Produce the labels for each of the test samples
    labels_test = codebook.predict(X_test)

    # Reshape to appropriate sizes 
    # X_train = X_train.reshape(1500, 16, 768)
    # X_test = X_train.reshape(500, 16, 768)
    X_train = X_train.reshape(len(y_train), 16, 768)
    X_test = X_test.reshape(len(y_test), 16, 768)

    # Build histogram
    X_hist_train = build_histogram(labels_train)
    X_hist_test = build_histogram(labels_test)
    
    # Train the classifier
    train_classifier(X_hist_train, X_hist_test, y_train, y_test)


if __name__ == "__main__":
    main()
