import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import math
import copy
import glob
import time

def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")


def extract_all_features(directory,feature_num):
    # extract features from all images
    img_list = []
    feature_cum = [0]
    img_index = 0
    for img_name in glob.glob(directory+'/*.jpg'):
        img = cv2.imread(img_name)
        img_list.append(img)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(feature_num)
        kp,des=sift.detectAndCompute(img_gray,None)
        kp_xy = np.array([[x.pt[0],x.pt[1]] for x in kp])
        des = np.matrix(des)
        des_n = (des-np.mean(des,axis=1))/np.std(des,axis=1)
        label = img_index*np.ones((len(kp),1))
        if img_index == 0:
            all_kp = kp_xy
            all_des = des_n
            all_label = label
            feature_cum.append(len(kp))
        else:
            all_kp = np.vstack((all_kp,kp_xy))
            all_des = np.vstack((all_des,des_n))
            all_label = np.vstack((all_label,label))
            feature_cum.append(feature_cum[img_index]+len(kp))
        img_index += 1
    
    return all_kp, all_des, all_label, feature_cum, img_list


def pairs_and_homography(des_1,des_2,kp_1,kp_2):
    # get matched pairs
    pairs = []
    thresh = 0.4
    for i in range(np.shape(des_1)[0]):
        pairs_d = np.linalg.norm(des_2-des_1[i,:],ord=2,axis=1)
        pairs_sorted = np.argsort(pairs_d)
        if pairs_d[pairs_sorted[0]]/pairs_d[pairs_sorted[1]]<=thresh:
            pairs.append([kp_1[i,0],kp_1[i,1],kp_2[pairs_sorted[0],0],
                          kp_2[pairs_sorted[0],1]])
    pairs = np.transpose(np.array(pairs))
    if len(pairs) == 0:
        return None, None
    elif np.shape(pairs)[1] < 30:
        return None, None
    
    # compute best homgraphy and inliers number
    bestCount = 0
    threshold = 2
    loopnumber = 5000
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
            
    return bestT,bestCount


def warp_imges(img_1,img_2,T):
    # warp two images given homography
    h = max(np.shape(img_1)[0],np.shape(img_2)[0])
    w = np.shape(img_1)[1]+np.shape(img_2)[1]
    h_2,w_2 = np.shape(img_2)[0], np.shape(img_2)[1]
    img_warp_size = tuple((math.ceil(1.5*w),math.ceil(1.5*h)))
    img_1_offset = np.matrix([[1,0,math.ceil(0.3*w)],[0,1,math.ceil(0.3*h)],[0,0,1]])
    img_warp_1 = cv2.warpPerspective(img_1,img_1_offset*T,img_warp_size)
    img_warp_2 = np.zeros(np.shape(img_warp_1))
    img_warp_2[math.ceil(0.3*h):math.ceil(0.3*h)+h_2, math.ceil(0.3*w):math.ceil(0.3*w)+w_2,:] = copy.deepcopy(img_2)
    img_warp_1_g = cv2.cvtColor(img_warp_1,cv2.COLOR_BGR2GRAY)
    img_warp_2_g = cv2.cvtColor(img_warp_2.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    index_1 = (img_warp_1_g>0) * ~(img_warp_2_g>0)
    index_2 = ~(img_warp_1_g>0) * (img_warp_2_g>0)
    index_overlap = (img_warp_1_g>0) * (img_warp_2_g>0)
    img_warp_combine = np.zeros(np.shape(img_warp_1))
    img_warp_combine[index_1,:] = img_warp_1[index_1,:]
    img_warp_combine[index_2,:] = img_warp_2[index_2,:]
    img_warp_combine[index_overlap,:] = img_warp_1[index_overlap,:]
    """
    img_warp_combine_g = cv2.cvtColor(img_warp_combine.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(img_warp_combine_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    x,y,wrange,hrange = cv2.boundingRect(boundary)
    img_warp_combine = img_warp_combine[y:y+hrange,x:x+wrange]
    """
    return img_warp_combine, img_1_offset
    
    
class graph:
    # graph structure for finding images that belong to the same panorama
    def __init__(self, img_num):
        self.info = [[] for i in range(img_num)]
        self.neighbours = [0 for i in range(img_num)]
    
    def add_edge(self,i,j,weight):
        if np.shape(self.info[i])[0] == 0:
            self.info[i] = np.array([[j],[weight]])
        else: self.info[i] = np.hstack((self.info[i],np.array([[j],[weight]])))
        if np.shape(self.info[j])[0] == 0:
            self.info[j] = np.array([[i],[weight]])
        else: self.info[j] = np.hstack((self.info[j],np.array([[i],[weight]])))
    
    def complete(self):
        for i in range(len(self.info)):
            if np.shape(self.info[i])[0] == 0:
                continue
            sort_weight = np.argsort(self.info[i][1,:])
            self.info[i] = self.info[i][:,np.flip(sort_weight)]
            self.neighbours[i] = sum(self.info[i][1,:])
    
    def remove_vertex(self,i):
        vertex = list(self.info[i][0,:].copy())
        self.neighbours[i] = -1 
        return vertex
        
    def get_begin_vertex(self):
        i = np.argmax(self.neighbours)
        if np.shape(self.info[i])[0] == 0:
            self.neighbours[i] = -1
            return i, []
        vertex = list(self.info[i][0,:].copy())
        self.neighbours[i] = -1
        return i, vertex
    
    def have_edge(self,i):
        return (self.exist(i) and self.neighbours[i] != 0)
    
    def exist(self,i):
        return self.neighbours[i] != -1
    
    def empty(self):
        return max(self.neighbours) == -1
    

def main():
    # specify a diectory that contain images for building panorama 
    directory = 'wizarding_world'
    feature_num = 3000
    
    # extract features from all images 
    all_kp,all_des,all_label,feature_cum,img_list = extract_all_features(
            directory,feature_num)
    all_des,all_label = all_des.astype(np.float32),all_label.astype(np.float32)
    
    # use KNN model to find top N matches for every features (The first one is themselves)
    model=cv2.ml.KNearest_create()
    model.train(all_des,cv2.ml.ROW_SAMPLE,all_label)
    knn = 3
    
    # initialize a graph that records the relationship between images
    img_graph = graph(len(img_list))
    all_relation = np.zeros((len(img_list),len(img_list)))
    all_homography = np.zeros((len(img_list),len(img_list),3,3))
    all_inliers = np.zeros((len(img_list),len(img_list)))
    
    
    print('start to find matched images:\n')
    for i in range(len(img_list)):
        
        # find best matches for all features in current image
        _,_,neighbours,_ = model.findNearest(all_des[
                feature_cum[i]:feature_cum[i+1],:],knn+1)
        
        # summarize number of matches between current image and others
        match_num = [sum(sum(neighbours==j)) for j in range(len(img_list))]
        match_index = list(np.argsort(match_num))
        match_index.reverse()
        
        # select N images with most matches with current image as candidates
        candidate = match_index[1:knn+1]
        for j in candidate:
            if all_relation[i,j] == 1:
                continue
            des_1,des_2 = all_des[feature_cum[i]:feature_cum[i+1],:], all_des[
                    feature_cum[j]:feature_cum[j+1],:]
            kp_1,kp_2 = all_kp[feature_cum[i]:feature_cum[i+1],:], all_kp[
                    feature_cum[j]:feature_cum[j+1],:]
            homography, inliers = pairs_and_homography(des_1,des_2,kp_1,kp_2)
            if inliers == None or inliers < 15:
                continue
            all_relation[i,j] = all_relation[j,i] = 1
            all_homography[i,j] = homography
            all_homography[j,i] = np.linalg.inv(homography)
            all_inliers[i,j] = all_inliers[j,i] = inliers
            img_graph.add_edge(i,j,inliers)
            print('a matched image pair is found:\n','image ',i+1,' <--> ',
                  'image ',j+1)
    
    img_graph.complete()
    
    count = 1
    while not img_graph.empty():
        print('making panorama ',count)
        index, current = img_graph.get_begin_vertex()
        
        # if an image has no relationship with others, it does not belong to a 
        # panorama
        if len(current) == 0:
            print('not a panorama')
            continue
        
        panorama = img_list[index]
        warp_offset = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        
        # when there are still images belonging to panorama that has not been
        # warp, warp it to the unfinished panorama
        while len(current) != 0:
            if img_graph.have_edge(current[0]):
                vertex = img_graph.remove_vertex(current[0])
                for i in vertex :
                    if i in current:
                        continue
                    if img_graph.exist(i):
                        current.append(i)
                        T = np.matrix(all_homography[current[0],index])*np.matrix(
                                all_homography[i,current[0]])
                        all_homography[i,index] = T/T[2,2]
            
            warp_img = img_list[current[0]]
            panorama,new_offset = warp_imges(warp_img,panorama,warp_offset*np.matrix(
                    all_homography[current[0],index]))
            warp_offset = new_offset*warp_offset
            current.pop(0)
        
        # crop the panorama to suitable size
        panorama_g = cv2.cvtColor(panorama.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(panorama_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        x,y,wrange,hrange = cv2.boundingRect(boundary)
        panorama = panorama[y:y+hrange,x:x+wrange]
        save_img(panorama,directory+'/panorama'+str(count)+'.png')
        count = count + 1
    
    
if __name__ == "__main__":
    main()