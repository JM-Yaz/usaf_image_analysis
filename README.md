We try to perform image matching and rotation in this code.

To run the code use python main.py <path_to_the_USAF_image>.

Please note that the unrotated.bmp image is neeeded to rotate the taken image and use it as a reference for the rotations.

The general idea is to compare the taken image with the image of reference, and match the taken image to the image of reference and process it through the rest of the algorithm.

Since most image matching algorithms work better with grayscale images, we read the images as grayscale images.

We then use the ORB (Oriented FAST and Rotated BRIEF) algorithm to detect keypoints and extract descriptors from the images. 
The keypoints are the distinctive features of the images, and the descriptors are vectors that describe the characteristics of the keypoints.

Once the keypoints and descriptors have been extracted, we use the BFMatcher (Brute-Force Matcher) algorithm to match the descriptors from the reference and target images. 
This algorithm compares the descriptors of the two images and returns a list of matches, which indicate which keypoints in the two images are similar.

We then sort the matches by distance and keep only the best ones, which are the most likely to be correct matches. 
The coordinates of the matched keypoints are then extracted and used as input to the RANSAC (RANdom SAmple Consensus) algorithm, which estimates the homography matrix between the two images.

The homography matrix is a 3x3 matrix that describes the geometric transformation between the two images. 
It contains information about the translation, rotation, and scaling of the images, and it can be used to calculate the rotation angle needed to align the two images.

To calculate the rotation angle, the code uses the Rodrigues formula to convert the homography matrix into a rotation matrix. 
The rotation matrix is a 3x3 matrix that describes the rotation of the images in 3D space. 
The rotation angle can then be calculated from the elements of the rotation matrix using the arctan2 function.

Finally, we use the calculated rotation angle and the cv2.getRotationMatrix2D() and cv2.warpAffine() methods to rotate the target image and align it with the reference image. 
The rotated image is then saved and processed through the rest of the algorithm.
