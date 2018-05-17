import numpy as np
import cv2 as cv

def deblur(y):
    """
    Returns the deblurred color plane.
    """

    # applying the formula (5) from the paper
    x = (H + lam*I).I*y

    return x

def gaussian_blur(img,sigma):
	#blur the image using Gaussian Blur; specify a kernel (odd positive) and sigma
	blurred = cv.GaussianBlur(img, (51,51), sigma)
	return blurred

# we're working with 255 x 255 pixel images
image_size = 255 * 255

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
img1_blurred = gaussian_blur(img1,3)
cv.imwrite("img/img1_blurred7.jpg", img1_blurred)

# split image into different color planes
b, g, r = cv.split(img1_blurred)

# flatten the image
b_flat, g_flat, r_flat = b.flatten(), g.flatten(), r.flatten()

# type cast the lists so they become matrices
b_flat = np.matrix(b_flat).T
g_flat = np.matrix(g_flat).T
r_flat = np.matrix(r_flat).T

r_deblur= deblur(r_flat)
g_deblur = deblur(g_flat)
b_deblur = deblur(b_flat)

r_deblur=np.reshape(r_deblur, (255, 255))
g_deblur=np.reshape(g_deblur, (255, 255))
b_deblur=np.reshape(b_deblur, (255, 255))

# print(type(b_flat))

# apply function to each color


# reshape the flattened matrix so it is a square matrix again
# b_deblur = b_flat

# join the color planes to reform the image
img = cv.merge((b_deblur, g_deblur, r_deblur))
cv.imwrite("img/img1_deblurred3.jpg", img)
