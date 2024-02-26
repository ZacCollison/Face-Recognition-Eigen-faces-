# Face-Recognition-Eigen-faces-
This project's aim was to develop a basic face recognition system using fundamental applications of linear algebra and coding practices to identify variances in faces. I aim to further refine the code to this project in the future.

# How to operate
Simply place training images into the "Library" folder, then place image you are testing into the "Test" folder. Then simply run the "EigenFaces.py" program and it will output the index of the image that is most similiar to your selected image. The indexes start at 0, with the top image in the folder and increment by 1.

# How does it work?
1. First the program reads the pixel values for the NxN images placed into the "Library" folder. It converts the image to black and white to optimise processing, as then it is dealing with 2 dimensional matrices instead of 3 dimensional matrices (RGB). In future, you will be able to put any size images in the library and faces will be cropped out of these images using the haarcascades algorithm.
2. Next, all the images are resized to $N^2$x1 dimensions.
3. Following this, concatenate all the image vectors together so that it forms a $N^2$xM matrix. Where "M" is the number of sample images in the library.
4. Next the average face is found from these images (This image is saved under "Average.png"), then subtracted from each image to form matrix "A" (A de-meaned matrix).
5. Next the covariance matrix is calculated by multiplying the transposed "A" matrix with iteself. This will show the variations of each image in respect to one another.
6. Then calculate the eigenvectors of the covariance matrix to depict the maxium variances between images. Choose the eigenvectors with the most variation (This is indicated by a higher eigenvalue).
7. Now multiply matrix "A" with the eigenVectors inorder to create eigenfaces, these can be seen under the EigenFaces folder in the document. These images show the most dramtic variations in faces.
8. Next the program, after to creating an eigenspace to project images onto to see their variations,it will then project the demeaned test image onto the eigenfaces and compare these values with the values of the sample faces that have been projected into the eigenspace.
9. It compares how close images are by using Euclidean distance between the values and finding the image with the lowest distance in variation.
10. Then finally returning the index of the closest matching image.

**Note**: Majority of these mathematical calculations were done by using the numpy library.
