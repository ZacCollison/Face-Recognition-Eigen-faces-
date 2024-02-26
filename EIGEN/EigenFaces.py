import glob
import numpy as np
import cv2 
from PIL import Image, ImageOps
import os
from matplotlib import pyplot as plt


def image_cropper(filename): #1365x1365 --> dimensions for matrix A --> using Haar cascades algorithm to extract/crop faces
    # Read the input image 
    img = cv2.imread(f'crop/{filename}') 
    
    # Convert into grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Load the cascade 
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml') 
    
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, 1.1, 6) 
    
    # Draw rectangle around the faces and crop the faces 
    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
        faces = img[y:y + h, x:x + w]
        faces.resize((400,400))
        cv2.imshow("face",faces) 
        cv2.imwrite(f'Library/{filename.split(".")[0]}.jpg', faces) 

def crop_delete():
    images_in_crop = os.listdir("crop")
    try:
        for i in range(len(images_in_crop)):
            os.remove(f"crop/{images_in_crop[i]}")
    except:
        print("Error deleting images in crop")

def form_matrix():
    
    images = os.listdir("Library")
    images= [f for f in glob.glob('Library/*.png')]

    A = np.zeros((200**2,1)) #defaulted to 200x200 images --> square photos on iphone --> resized to 200x200
    for i in range(len(images)):
        img = Image.open(f'{images[i]}')
        img.thumbnail((200,200))

        #Convert to greyscale so only using 2 dimensions instead of 3 for RGB
        grey = ImageOps.grayscale(img)
        new =  grey.resize((200,200))
        print(new.size)
        array = np.array(new).reshape((200**2,1))
        A = np.concatenate([A,array],axis=1)

    final = np.delete(A,0,1) #remove column of 0 in first column
    return final

def average_face(original):
    avg = np.sum(original,axis=1)
    avg_result = avg/original.shape[1]
    return avg_result

def demean(result,average):

    A = np.zeros((200**2,1))
    for i in range(result.shape[1]):
        imageVector = result[:,i]
        demeaned = imageVector - average 

        A = np.concatenate([A,demeaned.reshape((200**2,1))],axis=1)
    Final_A = np.delete(A,0,1)
    return Final_A

def detection(weights,eigenfaces,average):
    images = os.listdir("Test")
    img = Image.open(f"Test/{images[0]}")
    grey = ImageOps.grayscale(img)
    new =  grey.resize((200,200))
    test = np.array(new).reshape((200**2,1))
    

    ### DEMEAN testing image

    demeaned_test = test[:,0] - average 



    test_weight = eigenfaces.T @ demeaned_test #Project new values into eigen space
    distance = np.linalg.norm(weights - test_weight,axis=1)

    min_index = np.argmin(distance)
    error = test_weight[min_index]/weights[min_index] *100

    return min_index,distance[min_index]

def check_library(weights,eigenfaces,average,i):
    #this will run through the library and see which one has similiar outcome to testing image

    img = Image.open(f"{i}")
    grey = ImageOps.grayscale(img)
    new =  grey.resize((200,200))
    test = np.array(new).reshape((200**2,1))

    ### DEMEAN testing image

    demeaned_test = test[:,0] - average 



    test_weight = eigenfaces.T @ demeaned_test
    distance = np.linalg.norm(weights - test_weight,axis=1)
    print(distance.shape)
    min_index = np.argmin(distance)

    return min_index,distance[min_index]

# def eigenspace(demeaned_data):
#     print("Space")
#     Covariance = np.cov(demeaned_data,rowvar=False)
#     eigenvalues,eigenfaces = np.linalg.eig(Covariance)
#     sorted_eigen = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[sorted_eigen]
#     eigenfaces = eigenfaces[:,sorted_eigen]

#     totalVariance= np.sum(eigenvalues)
#     explainedVariance = np.cumsum(eigenvalues)/totalVariance
#     num_components = np.argmax(explainedVariance>=0.95)+1

#     selected_faces = eigenfaces[:, :num_components]
#     eigenspace = np.dot(demeaned_data,selected_faces)  

#     return eigenspace,selected_faces

def detectionUpdated(average,eigenfaces,oldFaces):
    print("detection")

    images = os.listdir("Test")
    img = Image.open(f"Test/{images[0]}")
    grey = ImageOps.grayscale(img)
    new =  grey.resize((200,200))
    newFace = np.array(new).reshape((200**2,1))

    demeaned_newFace = newFace[:,0] - average
    print(eigenfaces.shape)
    print(demeaned_newFace.shape)
    coefficientsNew = eigenfaces.T @ demeaned_newFace
    # coefficientsNew = np.dot(eigenfaces.T,demeaned_newFace)
    print(eigenfaces.shape)
    print(oldFaces.shape)
    coefficientsOld = (eigenfaces.T @ oldFaces).T
    # coefficientsOld = np.dot(eigenfaces.T,oldFaces.T).T

    distance = np.linalg.norm(coefficientsNew-coefficientsOld,axis=1)

    match = np.argmin(distance) #index of face put in --> gather from a csv file potentially???
    print("complete")
    return match





def main():
    ### read images in crop then sends them to be cropped--> placing them in library
    # images_for_crop = os.listdir("crop")
    # for i in range(len(images_for_crop)):
    #     image_cropper(images_for_crop[i])

    ### delete everything in crop after they have been cropped and moved to library
    # crop_delete()

    ### turn NxN images to N^2x1 dimensions and concatenate them so N^2xM matrix
    result = form_matrix()

    ###Find average face --> then subtract from traing set to form matrix A
    average = average_face(result)

    plt.imshow(average.reshape((200,200)),cmap='Greys')
    plt.savefig("Average")
    ### De-mean results to form matrix A
    A = demean(result,average)


    C = A.T @ A

    ### calculate eigen vector 
    # e = np.array([0,2*40260310.5]).T
    print("EIGEN")
    eigenValues,eigenVectors = np.linalg.eig(C) # note that columns correspond to different eigenVectors

    
    print("")
    print(max(eigenValues))

    mappedEigenVectors = A @ eigenVectors
    print(mappedEigenVectors.shape)
    for i in range(mappedEigenVectors.shape[1]):
        plt.imshow(mappedEigenVectors[:,i].reshape((200,200)),cmap='Greys')
        plt.savefig(f"EigenFaces/Eigen:{i}")


    

    #create weights --> each column represents the weights for each image
    print("WEIGHTS")
    w = mappedEigenVectors.T @ A
    print(w.shape)

    #detection

    result = detectionUpdated(average,mappedEigenVectors,A)
    print(result)
    # match,dist =  detection(w,mappedEigenVectors,average)
    # print(match)
    # images = os.listdir("Library")
    # images= [f for f in glob.glob('Library/*.png')]
    # dist_dict = {}
    # for i in range(len(images)):
    #     new_match,dist_check = check_library(w,mappedEigenVectors,average,images[i])
    #     if match == new_match:
    #         dist_dict[i] = abs(dist - dist_check)
    
    # print(dist_dict)
    # print(max(dist_dict))

main()