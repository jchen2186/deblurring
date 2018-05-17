import numpy as np
import cv2 as cv

def deblur(y):
    """
    Returns the deblurred color plane.
    """

    # applying the formula (5) from the paper
    x = (H + lam*I).T*H*y

    return x

def gaussian_blur(img,sigma):
	#blur the image using Gaussian Blur; specify a kernel (odd positive) and sigma
	blurred = cv.GaussianBlur(img, (65,65), sigma)
	return blurred

# we're working with 255 x 255 pixel images
image_size = 100 * 100

# Identity matrix I
I = np.identity(image_size)

# H is our convolution matrix, used to blur the original image
H = np.matrix(I)
# print(H)

# transpose of H
H_T = H.T

# lambda > 0 is the control parameter    print(blurred)
# we vary this to blur the image
lam = 1
# identity is the identity matrix
identity = np.matrix(np.identity(image_size))

# x is the matrix containing the pixels of the original/unblurred image
# y is the matrix containing the pixels of the blurred image
# x = (H_T * H + lam * identity).I * H_T * y

# import image
img1 = cv.imread('img/flower1.jpg')
img1_blurred = gaussian_blur(img1,7)
cv.imwrite("img/img1_blurred7.jpg", img1_blurred)



# split image into different color planes
b, g, r = cv.split(img1)

# flatten the image
b_flat, g_flat, r_flat = b.flatten(), g.flatten(), r.flatten()

# type cast the lists so they become matrices
b_flat = np.matrix(b_flat)
g_flat = np.matrix(g_flat)
r_flat = np.matrix(r_flat)

# print(type(b_flat))

# apply function to each color


# reshape the flattened matrix so it is a square matrix again
# b_deblur = b_flat

# join the color planes to reform the image
# img = cv.merge((b_deblur, g_deblur, r_deblur))
