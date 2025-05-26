import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

######################################
def salt_pepper_noise(img):
    """
    Degrade image using salt-pepper noise
    """
    h, w = img.shape

    p_salt = 0.25  # probability of salt noise
    p_pepper = 0.25
    p_noise = p_salt + p_pepper

    noisy_image = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            random_p = np.random.uniform(0, 1)
            if random_p > p_noise:
                noisy_image[i, j] = img[i, j]
            elif random_p < p_salt:
                noisy_image[i, j] = 255  # if true, add salt noise, maximum value = 255, minimum value = 0
            else:
                noisy_image[i, j] = 0    # add pepper noise, maximum value = 255, minimum value = 0

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(noisy_image, cmap='gray')
    plt.title('noisy_image'), plt.xticks([]), plt.yticks([])
    plt.show()
    return noisy_image
############################################################################

############################################################################
def mean_median_filters(noisy_image):
    """
    mean filter, median filter
    """
    h, w = noisy_image.shape
    k_mean = 7
    denoised_image_mean = np.zeros_like(noisy_image)
    noisy_image_pad = np.zeros((h+k_mean-1, w+k_mean-1))
    noisy_image_pad[int((k_mean-1)/2): h+int((k_mean-1)/2), int((k_mean-1)/2): w+int((k_mean-1)/2)] = noisy_image
    for i in range(h):
        for j in range(w):
            ROI = noisy_image_pad[i:i+k_mean, j:j+k_mean]  # select the region of interest (ROI), or the region belongs to a kernel.
            denoised_value = np.mean(ROI)  # calculate the average value
            denoised_image_mean[i, j] = denoised_value


    # median filter
    k_median = 7
    denoised_image_median = np.zeros_like(noisy_image)
    noisy_image_pad = np.zeros((h+k_median-1, w+k_median-1))
    noisy_image_pad[int((k_median-1)/2): h+int((k_median-1)/2), int((k_median-1)/2): w+int((k_median-1)/2)] = noisy_image
    for i in range(h):
        for j in range(w):
            ROI = noisy_image_pad[i:i+k_median, j:j+k_median]  # select the region of interest (ROI), or the region belongs to a kernel.
            denoised_value = np.median(ROI)  # select the median value
            denoised_image_median[i, j] = denoised_value

    cv2.imwrite('mean_filter.png', np.uint8(denoised_image_mean))
    cv2.imwrite('median_filter.png', np.uint8(denoised_image_median))

    plt.subplot(131), plt.imshow(noisy_image, cmap='gray')
    plt.title('noisy image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(denoised_image_mean, cmap='gray')
    plt.title('mean filter'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(denoised_image_median, cmap='gray')
    plt.title('median filter'), plt.xticks([]), plt.yticks([])
    plt.show()
####################################################################################################

###############################################################################################
def inverse_filtering(H_shift, img_blur):
    """
    Inverse filtering
    """

    cutoff_H = 110  # cutoff radius
    h, w = img_blur.shape
    h_c = h / 2
    w_c = w / 2

    G = cv2.dft(np.float32(img_blur), flags=cv2.DFT_COMPLEX_OUTPUT)
    G_shift = np.fft.fftshift(G)

    f_shift = np.zeros_like(G_shift)
    for v in range(h):
        for u in range(w):
            distance = (v-h_c)**2 + (u-w_c)**2
            if distance <= cutoff_H*cutoff_H:
                f_shift[v, u] = G_shift[v, u] / H_shift[v, u]
            else:
                f_shift[v, u] = G_shift[v, u]

    f_ishift = np.fft.ifftshift(f_shift)
    img_restored = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_restored = np.uint8(np.maximum(np.minimum(img_restored, 255), 0))

    plt.subplot(131), plt.imshow(img_blur, cmap='gray')
    plt.title('observed image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_restored, cmap='gray')
    plt.title('restored image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.imread('hku.png', 0), cmap='gray')
    plt.title('reference image'), plt.xticks([]), plt.yticks([])
    plt.show()

#################################################################################################


if __name__ == '__main__':
    img = np.float32(cv2.imread('hku.png', 0))
    noisy_image = salt_pepper_noise(img)

    mean_median_filters(noisy_image)

    H_shift = np.load('gaussian_low_pass.npy')  # low-pass gaussian filter (frequency domain), center is 1.
    img_blur = cv2.imread('blurry_hku.png', 0)

    inverse_filtering(H_shift, img_blur)
