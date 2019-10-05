import os
import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import math


def read_img(path):
    return cv2.imread(path, 0)


def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")
    
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W

    if np.shape(kernel)[0] == 1:
        kernel = np.vstack((np.zeros((1,np.shape(kernel)[1])),kernel,np.zeros((1,np.shape(kernel)[1]))))
    if np.shape(kernel)[1] == 1:
        kernel = np.hstack((np.zeros((np.shape(kernel)[0],1)),kernel,np.zeros((np.shape(kernel)[0],1))))
    image_2=signal.convolve2d(image,kernel,mode='same')
    return image_2

def gaussian_2d(x,y,sigma):
    return 1/(2*math.pi*sigma*sigma)*math.exp(-(x*x+y*y)/(2*sigma*sigma))

def harris_corner_detector(image, x_offset=2, y_offset=2, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return an heatmap image where every pixel is the harris
    # corner detector score for that pixel.
    # OR, do this with gradients (think Sobel operator) and
    # the structure tensor. 
    # Input- image: H x W
    #        x_offset: a scalar
    #        y_offset: a scalar
    #        window_size: a scalar tuple M, N 
    # Output- results: a image of size H x W

    ## Compute Ix, Iy using sobel filter
    kx = 0.5*np.array([[1,0,-1]])
    ky = 0.5*np.array([[1],[0],[-1]])
    Ix, Iy = convolve(image,kx), convolve(image,ky)
    
    ## Compute guassian filter of window_size
    k_g = np.zeros(window_size)
    sigma = 1
    for i in [(x,y)for x in range(-x_offset,x_offset+1) for y in range(-y_offset,y_offset+1)]:
        k_g[i[0]+x_offset,i[1]+y_offset] = gaussian_2d(i[0],i[1],sigma)
    
    ## compute component of M
    Ix_2_w, Iy_2_w, Ix_Iy_w = convolve(Ix*Ix,k_g), convolve(Iy*Iy,k_g),convolve(Ix*Iy,k_g)
    Ix_2_w, Iy_2_w, Ix_Iy_w = np.reshape(Ix_2_w,-1),np.reshape(Iy_2_w,-1),np.reshape(Ix_Iy_w,-1)
    
    ## compute M array
    M = np.array([[[Ix_2_w[i],Ix_Iy_w[i]],[Ix_Iy_w[i],Iy_2_w[i]]] for i in range(np.shape(Ix_2_w)[0])])
    
    ## compute R array
    alpha = 0.04
    R = np.linalg.det(M) - alpha*np.trace(M,axis1=1,axis2=2)*np.trace(M,axis1=1,axis2=2)
    R = np.reshape(R,np.shape(image))
    
    #_, R = cv2.threshold(R, 2000 , 255, cv2.THRESH_BINARY)
    
    plt.figure()
    plt.imshow(R, cmap = plt.cm.hot)
    plt.colorbar()
    plt.show()
    
    return R


def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Feature Detection #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    harris_corner_image = harris_corner_detector(img)
    save_img(harris_corner_image, "./feature_detection/q1.png")


if __name__ == "__main__":
    main()
