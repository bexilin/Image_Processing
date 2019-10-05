import scipy
from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
import math


def read_img(path):
    return cv2.imread(path, 0)


def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")


def display_img(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


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


def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    kx = np.array([[1,0,-1]])  # 1 x 3
    ky = np.array([[1],[0],[-1]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(np.square(Ix)+np.square(Iy))

    return grad_magnitude, Ix, Iy


def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx, Gy = convolve(image,kx), convolve(image,ky)
    grad_magnitude = np.sqrt(np.square(Gx)+np.square(Gy))

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    output = []
    for i in angles:
        k = math.cos(i)*kx + math.sin(i)*ky
        output.append(convolve(image,k))
    
    return output


def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N

    xnumber = math.floor(np.shape(image)[0]/16)
    ynumber = math.floor(np.shape(image)[1]/16)
    output_origin = np.array([image[16*i:16*(i+1),16*j:16*(j+1)] for i in range(xnumber) for j in range(ynumber)])
    output_shape = np.shape(output_origin)
    output_vectors = np.reshape(output_origin,(output_shape[0],16*16))
    output_vectors = np.matrix(output_vectors)
    output_vectors_normalized = (output_vectors-np.mean(output_vectors,axis=1))/np.std(output_vectors,axis=1)
    output_vectors_normalized = np.array(output_vectors_normalized)
    output = np.reshape(output_vectors_normalized,output_shape)
    output = list(output)
    
    """
    output = []
    for patch in output_origin:
        patch_v = np.reshape(patch, -1)
        patch_v = (patch_v-np.mean(patch_v))/np.std(patch_v)
        patch_m = np.reshape(patch_v,np.shape(patch))
        output.append(patch_m)
    """
    return output

def gaussian_2d(x,y,sigma):
    return 1/(2*math.pi*sigma*sigma)*math.exp(-(x*x+y*y)/(2*sigma*sigma))

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    patches = image_patches(img)
     
    for i in range(3):
        chosen_id = np.random.randint(0,len(patches)-1)
        chosen_patch = patches[chosen_id]
        filename = "./image_patches/q1_patch_" + str(chosen_id+1) + ".png"
        save_img(chosen_patch, filename)
    # save_img(chosen_patches, "./image_patches/q1_patch.png")

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    kernel_gaussian = np.zeros((3,3))
    sigma = 0.8493
    for i in [(x,y)for x in range(-1,2) for y in range(-1,2)]:
        kernel_gaussian[i[0]+1,i[1]+1] = gaussian_2d(i[0],i[1],sigma)

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")


    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()
