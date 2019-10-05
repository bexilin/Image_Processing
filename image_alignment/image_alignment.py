from skimage import io 
from skimage.transform import resize
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def task_1(origin,filename):
    # stack R, G, B image to make a color image 
    o_size=np.shape(origin)
    length=math.floor(o_size[0]/3.)
    B=origin[:length,:].copy()
    G=origin[length:2*length,:].copy()
    R=origin[2*length:3*length,:].copy()
    RGB=np.dstack((R,G,B))
    newname=filename+'_task1.jpg'
    io.imsave(newname,RGB)
    return RGB

    
def cropborder(img,crop):
    # crop the extra white or black border of the image
    if (img[3,3]<=20):
        _, t = cv2.threshold(img, 20,255, cv2.THRESH_BINARY)
    else:
        _, t = cv2.threshold(img, 240,255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    x,y,w,h = cv2.boundingRect(boundary)
    
    if crop is 0:
        crop = img.copy()
        cv2.drawContours(crop,[boundary],-1,150,2)
    else:
        crop = img[y:y+h,x:x+w]

    return crop

def task_2_1(image,xrange,yrange,filename,save):
    # finding offset for R, G, B image alignment through dot product of intensity
    R,G,B = image[:,:,0], image[:,:,1], image[:,:,2]
    R_v = np.reshape(R,-1)
    R_v = R_v/np.sum(R_v)
    for i in range(-xrange,xrange+1):
        for j in range(-yrange,yrange+1):
            B_r,G_r=np.roll(B,(i,j),axis=(0,1)),np.roll(G,(i,j),axis=(0,1))
            B_r_v,G_r_v = np.reshape(B_r,-1),np.reshape(G_r,-1)
            B_r_v,G_r_v = B_r_v/np.sum(B_r_v),G_r_v/np.sum(G_r_v)
            dot_BR = np.dot(B_r_v,R_v)
            dot_GR = np.dot(G_r_v,R_v)
            if (i,j) == (-xrange,-yrange):
                best_BR, best_GR = dot_BR, dot_GR
                best_B_offset, best_G_offset = (i,j), (i,j)
            else:
                if dot_BR > best_BR:
                    best_BR = dot_BR
                    best_B_offset = (i,j)
                if dot_GR > best_GR:
                    best_GR = dot_GR
                    best_G_offset = (i,j)
    
    B_best, G_best = np.roll(B,best_B_offset,axis=(0,1)),np.roll(G,best_G_offset,axis=(0,1))
    RGB=np.dstack((R,G_best,B_best))
    if save == 1:
        newname=filename+'_task2.jpg'
        io.imsave(newname,RGB)
    elif save == 2:
        newname=filename+'_task3.jpg'
        io.imsave(newname,RGB)
    return 0

def task_2_2(image,xrange,yrange,filename,save):
    # finding offset for R, G, B image alignment through dot product of gradient
    R,G,B = image[:,:,0], image[:,:,1], image[:,:,2]
    B_g,G_g,R_g = cv2.Sobel(B,-1,1,1), cv2.Sobel(G,-1,1,1), cv2.Sobel(R,-1,1,1)
    B_g_m,G_g_m,R_g_m=np.mean(B_g),np.mean(G_g),np.mean(R_g)
    R_g_r=np.reshape(R_g,-1)-R_g_m
    for i in range(-xrange,xrange+1):
        for j in range(-yrange,yrange+1):
            B_g_r,G_g_r=np.roll(B_g,(i,j),axis=(0,1)),np.roll(G_g,(i,j),axis=(0,1))
            B_g_r1,G_g_r1=np.reshape(B_g_r,-1),np.reshape(G_g_r,-1)
            dot_BR = np.dot((B_g_r1-B_g_m),R_g_r)
            dot_GR = np.dot((G_g_r1-G_g_m),R_g_r)
            if (i,j) == (-xrange,-yrange):
                best_BR, best_GR = dot_BR, dot_GR
                best_B_offset, best_G_offset = (i,j), (i,j)
            else:
                if dot_BR > best_BR:
                    best_BR = dot_BR
                    best_B_offset = (i,j)
                if dot_GR > best_GR:
                    best_GR = dot_GR
                    best_G_offset = (i,j)
    
    B_best, G_best = np.roll(B,best_B_offset,axis=(0,1)),np.roll(G,best_G_offset,axis=(0,1))
    RGB=np.dstack((R,G_best,B_best))
    if save == 1:
        newname=filename+'_task2.jpg'
        io.imsave(newname,RGB)
    elif save == 2:
        newname=filename+'_task3.jpg'
        io.imsave(newname,RGB)
    return best_B_offset, best_G_offset

def task_3(image,xrange,yrange,filename):
    # finding best offset for R, G, B image alignment in two step, first downsample
    # image to half width and height, finding best offset within given range, then
    # use this alignment as initial, do the job again on the full size image
    image_1 = resize(image,tuple(int(0.5*x) for x in np.shape(image)[:2]))
    B_1, G_1 = task_2_2(image_1,int(0.5*xrange),int(0.5*yrange),0,0)
    R,G,B = image[:,:,0], image[:,:,1], image[:,:,2]
    G_1, B_1 = tuple(int(2*x) for x in np.shape(G_1)),tuple(int(2*x) for x in np.shape(B_1))
    G_best_2, B_best_2 = np.roll(G,G_1,axis=(0,1)),np.roll(B,B_1,axis=(0,1))
    image_2 = np.dstack((R,G_best_2,B_best_2))
    B_2, G_2 = task_2_2(image_2,int(0.5*xrange),int(0.5*yrange),filename,2)  
    return 0

if __name__ == "__main__":
    # please specify an image to do the following tasks
    filename = None
    origin=io.imread(filename+'.jpg')
    RGB=task_1(origin,filename)
    R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
    B_crop, G_crop, R_crop=cropborder(B,1),cropborder(G,1),cropborder(R,1)
    xmin = min(np.shape(B_crop)[0],np.shape(G_crop)[0],np.shape(R_crop)[0])
    ymin = min(np.shape(B_crop)[1],np.shape(G_crop)[1],np.shape(R_crop)[1])  
    RGB_crop = np.dstack((R_crop[0:xmin,0:ymin],G_crop[0:xmin,0:ymin],B_crop[0:xmin,0:ymin]))
    
    task_2_2(RGB_crop,15,15,filename,1)
    
    task_3(RGB_crop,30,30,filename)
