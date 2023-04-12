import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
1) Implement a function that changes the brightness and/or contrast of a colored image.
The function should have as input:
• a colored or black and white image, with values 𝑟, 𝑔, 𝑏
• the value of the brightness change 𝛽
• the value of the contrast change 𝜅
The brightness should be changed additively, that is, the pixel with color 𝑐 = (𝑟, 𝑔, 𝑏) should
pass to 𝑐 = (𝑟 + 𝛽, 𝑔 + 𝛽, 𝑏 + 𝛽) (it is the same color, but lower saturation - it was mixed with a
little white color).
The contrast should be changed multiplicatively, that is, the pixel with color 𝑐 = (𝑟, 𝑔, 𝑏) should
pass to 𝑐 = (𝜅(𝑟 − 𝑟̅ ) + 𝑟̅ , 𝜅(𝑔 − 𝑔̅ ) + 𝑔̅ , 𝜅(𝑏 − 𝑏̅ ) + 𝑏̅ ).
Test with the image lowcontrast.png or another image of your choice.
"""

img = cv2.imread('assets/lowcontrast.png')

def edit_image(img, brightness, contrast):
    new_img = img + brightness

    for i in range(3): # works for colorful images
        avg = np.average(new_img[:, :, i])
        new_img[:, :, i] = (new_img[:, :, i] - avg)*contrast + avg

    return new_img

bright_img = edit_image(img, 70, 1)
high_contrast_img = edit_image(img, 0, 1.75)

cv2.imshow("Original", img)
cv2.waitKey(0)

cv2.imshow("Bright", bright_img)
cv2.waitKey(0)

cv2.imshow("High Contrast", high_contrast_img)
cv2.waitKey(0)

"""
2) Make the histogram of the original image, the image with altered brightness and the
image with altered contrast using the function developed in (1). Compare with the
histogram of the image with brightness and contrast altered by the function
cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
Does your function implement the same concept as the OpenCV function?
"""

img_2 = cv2.convertScaleAbs(img, alpha=1, beta=70)
cv2.imshow("cv2's bright", img_2)
cv2.waitKey(0)

img_3 = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
cv2.imshow("cv2's high contrast", img_3)
cv2.waitKey(0)

fig, axs = plt.subplots(nrows=5)
ax0, ax1, ax2, ax3, ax4 = axs
ax0.hist(img.ravel(), 256, [0, 256], color='blue', label='Original')
ax0.legend(loc='upper center')
ax1.hist(bright_img.ravel(), 256, [0, 256], color='red', label='Bright')
ax1.legend(loc='upper center')
ax2.hist(high_contrast_img.ravel(), 256, [0, 256], color='green', label='High Contrast')
ax2.legend(loc='upper center')
ax3.hist(img_2.ravel(), 256, [0, 256], color='yellow', label="cv2's bright")
ax3.legend(loc='upper center')
ax4.hist(img_3.ravel(), 256, [0, 256], color='purple', label="cv2's high contrast")
ax4.legend(loc='upper center')
ax4.tick_params(axis='y', labelcolor='red')
plt.show()

"""
While our function tries to keep the image's average color and raise
the standard deviation, cv2's function just multiplies the image's
values by the alpha parameter, which is why the image is so bright.
The beta parameter is just added to the image's values, acting like our
brightness parameter.
"""

"""
3) Write a function to perform the convolution of an image with a filter. The function
should have as input:
• a image
• the matrix corresponding to the filter
The output should be the filtered image. Test with the following filters:
• Constant 3x3
• Derivative in the horizontal 1x3 and derivative in the vertical 3x1
• Sobel filter horizontal and vertical 3x3 (how about looking at the module of the gradient too?)
• Gaussian filter with mean zero and variance of 2 pixels, truncated in a 5x5 matrix.
The images for testing are up to you, but one of the tests should be with a chessboard image.
"""

def apply_convolution(img, matrix):
    img = img/255
    new_img = np.zeros(img.shape)

    x_padding = (matrix.shape[0]-1)//2
    y_padding = (matrix.shape[1]-1)//2

    for i in range(x_padding + 1, img.shape[0] - x_padding):
        for j in range(y_padding + 1, img.shape[1] - y_padding):
            for k in range(3):
                new_img[i, j, k] = np.sum(img[i - x_padding : i + x_padding + 1,
                                              j - y_padding : j + y_padding + 1,
                                              k] * matrix) # * is element-wise multiplication

    return new_img

def gaussian_matrix(size=3, std=1, mean=(0, 0)):
    gauss_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gauss_matrix[i, j] = (1/(2*np.pi*std**2)) * np.exp(-((i - size//2 - mean[0])**2 + (j - size//2 - mean[1])**2) / (2*std**2))
    
    return gauss_matrix

img = cv2.imread('assets/chessboard.png')
cv2.imshow("Original", img)
cv2.waitKey(0)

constant_filter = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])/9 # soft blur
filtered_img = apply_convolution(img, constant_filter)
cv2.imshow("Filtered", filtered_img)
cv2.waitKey(0)

derivative_x = np.array([[1, 0, -1]])
derivative_x_img = apply_convolution(img, derivative_x)
cv2.imshow("Derivative in relation to x", derivative_x_img)
cv2.waitKey(0)

derivative_y = np.array([[1],
                         [0],
                         [-1]])
derivative_y_img = apply_convolution(img, derivative_y)
cv2.imshow("Derivative in relation to y", derivative_y_img)
cv2.waitKey(0)

sobel_x = np.array([[1, 0,-1],
                    [2, 0,-2],
                    [1, 0,-1]]) # Sobel could also be seen as [[1],[2],[1]]*([1,0,-1]*A)
sobel_x_img = apply_convolution(img, sobel_x)
cv2.imshow("Sobel in relation to x", sobel_x_img)
cv2.waitKey(0)

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])
sobel_y_img = apply_convolution(img, sobel_y)
cv2.imshow("Sobel in relation to y", sobel_y_img)
cv2.waitKey(0)

gradient_magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)
cv2.imshow("Gradient magnitude", gradient_magnitude)
cv2.waitKey(0)

gaussian_blur = gaussian_matrix(5, 2)
gaussian_blur_img = apply_convolution(img, gaussian_blur)
cv2.imshow("Gaussian blur", gaussian_blur_img)
cv2.waitKey(0)

img = cv2.imread('assets/bike.png')
cv2.imshow("Original", img)
cv2.waitKey(0)

constant_filter = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])/9 # soft blur
filtered_img = apply_convolution(img, constant_filter)
cv2.imshow("Filtered", filtered_img)
cv2.waitKey(0)

derivative_x = np.array([[1, 0, -1]])
derivative_x_img = apply_convolution(img, derivative_x)
cv2.imshow("Derivative in relation to x", derivative_x_img)
cv2.waitKey(0)

derivative_y = np.array([[1],
                         [0],
                         [-1]])
derivative_y_img = apply_convolution(img, derivative_y)
cv2.imshow("Derivative in relation to y", derivative_y_img)
cv2.waitKey(0)

sobel_x = np.array([[1, 0,-1],
                    [2, 0,-2],
                    [1, 0,-1]]) # Sobel could also be seen as [[1],[2],[1]]*([1,0,-1]*A)
sobel_x_img = apply_convolution(img, sobel_x)
cv2.imshow("Sobel in relation to x", sobel_x_img)
cv2.waitKey(0)

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])
sobel_y_img = apply_convolution(img, sobel_y)
cv2.imshow("Sobel in relation to y", sobel_y_img)
cv2.waitKey(0)

gradient_magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)
cv2.imshow("Gradient magnitude", gradient_magnitude)
cv2.waitKey(0)

gaussian_blur = gaussian_matrix(5, 2)
gaussian_blur_img = apply_convolution(img, gaussian_blur)
cv2.imshow("Gaussian blur", gaussian_blur_img)
cv2.waitKey(0)

img = cv2.imread('assets/valve.png')
cv2.imshow("Original", img)
cv2.waitKey(0)

constant_filter = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])/9 # soft blur
filtered_img = apply_convolution(img, constant_filter)
cv2.imshow("Filtered", filtered_img)
cv2.waitKey(0)

derivative_x = np.array([[1, 0, -1]])
derivative_x_img = apply_convolution(img, derivative_x)
cv2.imshow("Derivative in relation to x", derivative_x_img)
cv2.waitKey(0)

derivative_y = np.array([[1],
                         [0],
                         [-1]])
derivative_y_img = apply_convolution(img, derivative_y)
cv2.imshow("Derivative in relation to y", derivative_y_img)
cv2.waitKey(0)

sobel_x = np.array([[1, 0,-1],
                    [2, 0,-2],
                    [1, 0,-1]]) # Sobel could also be seen as [[1],[2],[1]]*([1,0,-1]*A)
sobel_x_img = apply_convolution(img, sobel_x)
cv2.imshow("Sobel in relation to x", sobel_x_img)
cv2.waitKey(0)

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])
sobel_y_img = apply_convolution(img, sobel_y)
cv2.imshow("Sobel in relation to y", sobel_y_img)
cv2.waitKey(0)

gradient_magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)
cv2.imshow("Gradient magnitude", gradient_magnitude)
cv2.waitKey(0)

gaussian_blur = gaussian_matrix(5, 0.5)
gaussian_blur_img = apply_convolution(img, gaussian_blur)
cv2.imshow("Gaussian blur", gaussian_blur_img)
cv2.waitKey(0)


"""
4) Create a reduced version of an image by eliminating half of the lines and half of the columns
in two ways:
a) Without doing a smoothing, just cutting lines and columns
b) With smoothing before eliminating lines and columns.
You can use OpenCV functions to do the smoothing.
"""
# a
huge_fruits = cv2.imread('assets/fruits.png')

fruits = huge_fruits[::2, ::2]
cv2.imshow("Original", huge_fruits)
cv2.waitKey(0)
cv2.imshow("Downsampled", fruits)
cv2.waitKey(0)
print(huge_fruits.shape, fruits.shape)

# b
downsampled_fruits = fruits[::2, ::2]

blurred_fruits = cv2.GaussianBlur(fruits, (5,5), 0)
downsampled_blurred_fruits = blurred_fruits[::2, ::2]

cv2.imshow("Original", fruits)
cv2.waitKey(0)

cv2.imshow("Downsampled", downsampled_fruits)
cv2.waitKey(0)

cv2.imshow("Blurred", blurred_fruits)
cv2.waitKey(0)

cv2.imshow("Downsampled Blurred", downsampled_blurred_fruits)
cv2.waitKey(0)