import numpy as np
import cv2 as cv
import sys

def deblur(y):
    """
    Returns the deblurred color plane.
    """

    # applying the formula (5) from the paper
    x = (H + lam*I).I*y

    return x

def gaussian_blur(img, kernel, sigma):
	#blur the image using Gaussian Blur; specify a kernel (odd positive) and sigma
	blurred = cv.GaussianBlur(img, (kernel,kernel), sigma)
	return blurred

def generate_H(size):
	I = np.identity(size)
	current = 1
	for row in I:
		if current < len(row):
			row[current] = 1
			current += 1
	return np.matrix(.5 * I)

def deblur_by_row(plane, side_dem, lam):
	"""deblur the color plane by rows, returns the plane deblurred """

	# x is the matrix containing the pixels of the original/unblurred image
	# y is the matrix containing the pixels of the blurred image
	# x = (H_T * H + lam * identity).I * H_T * y

	"""solve for x = (H^T*H + lam*I)^-1*H^T*y
	since we don't want to do the inverse, we will solve for (H^T*H + lam*I)x = H^T * y"""

	H = generate_H(side_dem)
	I = np.matrix(np.identity(side_dem))
	H_transposed = H.transpose()

	A = H_transposed * H + lam * I

	answer = np.empty((0,side_dem))

	for row in plane:
		y = np.matrix(row).transpose() # transpose the row
		right_side = H_transposed * y
		x = np.linalg.solve(A, right_side) #solve the system of equation to get the deblurred image

		answer = np.append(answer,x.transpose(), axis=0) #append the transpose of that to the matrix

	return answer

# ------------------- DEFAULTS -------------------
img_name = "flower1"
extension = ".jpg"

# BLUR OPTIONS (for gaussian blur)
kernel = 3
sigma = 3

# lambda > 0 is the control parameter    print(blurred)
# we vary this to debblur the image
lam = 3

# ------------------- USER INPUT -------------------
print("1. Use defaults.")
print("2. Input image name, blur options, and lambda for deblur.")

if (sys.version_info > (3, 0)):
    # python 3 code
    option = int(input("Choose option: "))

    if option == 2:
        img_name = input("Choose image (without file extension): ")
        extension = input("File image extension: ")

        # get blurring options
        kernel = int(input("Gaussian kernel: "))
        sigma = int(input("Gaussian sigma: "))

        # get lambda
        lam = float(input("Lambda: "))
else:
    # python 2 code
    option = int(raw_input("Choose option: "))

    if option == 2:
    	img_name = raw_input("Choose image (without file extension): ")
    	extension = raw_input("File image extension: ")

    	# get blurring options
    	kernel = int(raw_input("Gaussian kernel: "))
    	sigma = int(raw_input("Gaussian sigma: "))

    	# get lambda
    	lam = int(raw_input("Lambda: "))

# import image
img_path = 'img/' + img_name + extension
print("Original unblurred image:", img_path)
img = cv.imread(img_path)

# image dimensions
height, width = img.shape[:2]
image_size = height * width

# blur the image and save it
img_blurred = gaussian_blur(img,kernel,sigma)
blurred_name = img_name + "_ker" + str(kernel) + "_sig" + str(sigma)
blurred_path = 'img/' + blurred_name + extension
print("Blurred image:", blurred_path)
cv.imwrite(blurred_path, img_blurred)

# split image into different color planes
b, g, r = cv.split(img_blurred)

# deblur by rows - apply function to each color
r_deblur= deblur_by_row(r, width, lam)
g_deblur = deblur_by_row(g, width, lam)
b_deblur = deblur_by_row(b, width, lam)

# deblur again on the columns
# r_deblur = deblur_by_row(r_deblur.transpose(), x_dem,lam)
# g_deblur = deblur_by_row(g_deblur.transpose(), x_dem, lam)
# b_deblur = deblur_by_row(b_deblur.transpose(), x_dem, lam)

# join the color planes to reform the image
img = cv.merge((b_deblur, g_deblur, r_deblur))
deblurred_name = blurred_name + "_deblurred_lam" + str(lam)
deblurred_path = "img/" + deblurred_name + ".jpg"
print("Deblurred image:", deblurred_path)
cv.imwrite(deblurred_path, img)


# # ---- OLD DEBLUR ----
# # Identity matrix I
# I = np.identity(image_size)

# # H is our convolution matrix, used to blur the original image
# H = np.matrix(I)
# # print(H)

# # transpose of H
# H_T = H.T

# # identity is the identity matrix
# identity = np.matrix(np.identity(image_size))

# # flatten the image
# b_flat, g_flat, r_flat = b.flatten(), g.flatten(), r.flatten()

# # type cast the lists so they become matrices
# b_flat = np.matrix(b_flat).T
# g_flat = np.matrix(g_flat).T
# r_flat = np.matrix(r_flat).T

# # reshape the flattened matrix so it is a square matrix again
# r_deblur=np.reshape(r_deblur, (x_dem, y_dem))
# g_deblur=np.reshape(g_deblur, (x_dem, y_dem))
# b_deblur=np.reshape(b_deblur, (x_dem, y_dem))