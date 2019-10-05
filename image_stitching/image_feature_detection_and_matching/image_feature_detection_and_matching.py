import os
import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import math
import copy
import time


def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")


def compute_sift(filename):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints,descriptors=sift.detectAndCompute(img_gray,None)
    
    return keypoints,descriptors

def main():
    # specify two images for matching
    img_1_name = 'bbb_left'
    img_2_name = 'bbb_right'
    ##### compute SIFT keypoints and descriptors #####
    img_1,img_2 = cv2.imread('./'+img_1_name+'.jpg'), cv2.imread('./'+img_2_name+'.jpg')
    img_1_g,img_2_g = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(2000)
    img_1_kp,img_1_d = sift.detectAndCompute(img_1_g,None)
    img_2_kp,img_2_d = sift.detectAndCompute(img_2_g,None)
    
    ##### feature detection directory #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")
    
    ##### draw keypoints on images #####
    img_1_draw_kp=cv2.drawKeypoints(img_1,img_1_kp,None)
    img_2_draw_kp=cv2.drawKeypoints(img_2,img_2_kp,None)
    save_img(img_1_draw_kp, "./feature_detection/"+img_1_name+".png")
    save_img(img_2_draw_kp, "./feature_detection/"+img_2_name+".png")
    
    ##### compute distances between descriptors from two images and select pairs
    img_1_d,img_2_d = np.matrix(img_1_d),np.matrix(img_2_d)
    img_1_d_n = (img_1_d-np.mean(img_1_d,axis=1))/np.std(img_1_d,axis=1)
    img_2_d_n = (img_2_d-np.mean(img_2_d,axis=1))/np.std(img_2_d,axis=1)
    
    thresh = 0.4
    pairs_d = np.zeros((np.shape(img_1_d_n)[0],np.shape(img_2_d_n)[0]))
    pairs = []
    
    start=time.time()
    for i in range(np.shape(img_1_d_n)[0]):
        print(i)
        pairs_d[i,:] = np.linalg.norm(img_2_d_n-img_1_d_n[i,:],ord=2,axis=1)
        pairs_sorted = np.argsort(pairs_d[i,:])
        if pairs_d[i,pairs_sorted[0]]/pairs_d[i,pairs_sorted[1]]<=thresh:
            pairs.append([img_1_kp[i].pt[0],img_1_kp[i].pt[1],img_2_kp[pairs_sorted[0]].pt[0],img_2_kp[pairs_sorted[0]].pt[1]])
    end=time.time()
    print(end-start)
    
    pairs = np.transpose(np.array(pairs))

    ##### ransac for finding best homography #####
    bestCount = 0
    threshold = 2
    loopnumber = 2000
    for loop in range(loopnumber):
        index=np.random.randint(np.shape(pairs)[1],size = 4)
        A = np.zeros((8,9))
        
        for i in range(len(index)):
            x = np.hstack((pairs[:2,index[i]],1))
            xd_x = pairs[2,index[i]]*x
            yd_x = pairs[3,index[i]]*x
            A[2*i,:] = np.hstack((-x,np.zeros(3),xd_x))
            A[2*i+1,:] = np.hstack((np.zeros(3),-x,yd_x))
            
        _,s,vh = np.linalg.svd(A)
        T = np.matrix(np.reshape(vh[np.argmin(s),:],(3,3)))
        x_all = np.matrix(np.vstack((pairs[:2,:],np.ones((1,np.shape(pairs)[1])))))
        xd_all = T*x_all
        xd_all = xd_all[:2,:]/xd_all[2,:]
        error = np.linalg.norm(xd_all-pairs[2:,:],ord=2,axis=0)
        inliers = sum(error < threshold)
        if inliers > bestCount:
            bestT = T/T[2,2]
            bestCount = inliers
            bestError = error < threshold
            bestMean = np.mean(error[bestError])
    
    ##### draw matched features #####
    img_compare_xrange = max(np.shape(img_1_draw_kp)[0],np.shape(img_2_draw_kp)[0])
    img_compare_yrange = np.shape(img_1_draw_kp)[1]+np.shape(img_2_draw_kp)[1]
    img_compare = 255*np.ones((img_compare_xrange,img_compare_yrange,3))
    img_compare[:np.shape(img_1_draw_kp)[0],:np.shape(img_1_draw_kp)[1],:]=copy.deepcopy(img_1_draw_kp)
    img_compare[:np.shape(img_2_draw_kp)[0],np.shape(img_1_draw_kp)[1]:,:]=copy.deepcopy(img_2_draw_kp)
    img_pairs=pairs.copy()
    img_pairs[2,:]=pairs[2,:].copy()+np.shape(img_1_draw_kp)[1]
    plt.imshow(cv2.cvtColor(img_compare.astype(np.uint8), cv2.COLOR_BGR2RGB), cmap = 'gray')
    for i in range(np.shape(pairs)[1]):
        if bestError[i]:
            x_plot = [img_pairs[0,i],img_pairs[2,i]]
            y_plot = [img_pairs[1,i],img_pairs[3,i]]
            plt.plot(x_plot,y_plot,lineWidth=0.2)
            #plt.plot(img_pairs[0,i],img_pairs[1,i],marker='o')
            #plt.plot(img_pairs[2,i],img_pairs[3,i],marker='o')
            
    plt.savefig("./feature_detection/"+img_1_name+"&"+img_2_name+"_compare.png")
    
    ##### warp images #####
    h,w = np.shape(img_compare)[0], np.shape(img_compare)[1]
    h_2,w_2 = np.shape(img_2)[0], np.shape(img_2)[1]
    img_warp_size = tuple((math.ceil(2*w),math.ceil(2*h)))
    img_1_offset = np.matrix([[1,0,math.ceil(0.5*w)],[0,1,math.ceil(0.5*h)],[0,0,1]])
    img_warp_1 = cv2.warpPerspective(img_1,img_1_offset*bestT,img_warp_size)
    img_warp_2 = np.zeros(np.shape(img_warp_1))
    img_warp_2[math.ceil(0.5*h):math.ceil(0.5*h)+h_2, math.ceil(0.5*w):math.ceil(0.5*w)+w_2,:] = copy.deepcopy(img_2)
    img_warp_1_g = cv2.cvtColor(img_warp_1,cv2.COLOR_BGR2GRAY)
    img_warp_2_g = cv2.cvtColor(img_warp_2.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    index_1 = (img_warp_1_g>0) * ~(img_warp_2_g>0)
    index_2 = ~(img_warp_1_g>0) * (img_warp_2_g>0)
    index_overlap = (img_warp_1_g>0) * (img_warp_2_g>0)
    img_warp_combine = np.zeros(np.shape(img_warp_1))
    img_warp_combine[index_1,:] = img_warp_1[index_1,:]
    img_warp_combine[index_2,:] = img_warp_2[index_2,:]
    img_warp_combine[index_overlap,:] = img_warp_1[index_overlap,:]
    img_warp_combine_g = cv2.cvtColor(img_warp_combine.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(img_warp_combine_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    x,y,wrange,hrange = cv2.boundingRect(boundary)
    img_warp_combine = img_warp_combine[y:y+hrange,x:x+wrange]
    
    
    """
    for hrange in range(np.shape(img_2)[0]):
        for wrange in range(np.shape(img_2)[1]):
            if np.linalg.norm(img_warp[math.ceil(0.5*h)+hrange,math.ceil(0.5*w)+wrange,:]-np.zeros(3))>0.1:
                img_warp[math.ceil(0.5*h)+hrange,math.ceil(0.5*w)+wrange,i]=0.5*(
                        img_warp[math.ceil(0.5*h)+hrange,math.ceil(0.5*w)+wrange,i]
                        +copy.deepcopy(img_2[hrange,wrange,i]))
            else:
                img_warp[math.ceil(0.5*h)+hrange,math.ceil(0.5*w)+wrange,i]=copy.deepcopy(img_2[hrange,wrange,i])
            
    """
    save_img(img_warp_combine, "./feature_detection/"+img_1_name+"&"+img_2_name+"_warp.png")
    
    return 0
        
    ##### harris corner #####
    #harris_corner_image = harris_corner_detector(img_2)
    #harris_corner_image = cv2.cornerHarris(img_1,5,3,0.04)
    #h = np.reshape(harris_corner_image,-1)
    #h = h/max(h)*255
    #h = np.reshape(h,np.shape(harris_corner_image))
    #harris_corner_image = harris_corner_detector(img_1_g)
    #save_img(harris_corner_image, "./feature_detection/img_1_2.png")
    
    ##### BFmatcher #####
    #bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    #matches = bf.match(img_1_d,img_2_d)
    #matches = sorted(matches, key = lambda x:x.distance)
    #img = cv2.drawMatches(img_1,img_1_kp,img_2,img_2_kp,matches[:400],None,flags=2)
    #plt.imshow(img),plt.show
    #save_img(img,"./feature_detection/BFmatcher.png")

if __name__ == "__main__":
    main()