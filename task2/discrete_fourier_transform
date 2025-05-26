import numpy as np
import cv2
from matplotlib import pyplot as plt


###################################################################
def DFT(img):
    """
    2d DFT using cv2.dft()
    """
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)   # 2D DFT on img
    dft_shift = np.fft.fftshift(dft)   # shift the low-frequency parts into the centre

    # visualize the spectrum and phase angle
    spectrum = np.sqrt(dft_shift[:, :, 0] ** 2 + dft_shift[:, :, 1] ** 2)
    spectrum = 20 * np.log(spectrum + 1)  # change the scale for better visualization, using 20*np.log()
    phase = np.arctan2(dft_shift[:, :, 1], dft_shift[:, :, 0])  # calculate phase, output range (-pi, pi)
    phase = phase / np.pi * 180  # Convert phase angle to [- 180, 180]

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(spectrum, cmap='gray')
    plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(phase, cmap='gray')
    plt.title('Phase'), plt.xticks([]), plt.yticks([])
    plt.show()

    return dft_shift

##################################################################


##################################################################
def high_low_pass_filter(dft_shift):
    """
    ideal high-pass and low-pass filter
    """
    cutoff = 30

    h, w = img.shape
    h_c = h // 2
    w_c = w // 2
    low_pass_filter = np.zeros((h, w, 2))  # the shape of low-pass filter

    # create low-pass filter
    for v in range(h):
        for u in range(w):
            distance = (v - h_c) ** 2 + (u - w_c) ** 2
            if distance <= cutoff ** 2:
                low_pass_filter[v, u, :] = 1  # assign a value to low pass filter

    # use the low-pass filter to filter out high frequencies.
    low_pass_frequency = dft_shift * low_pass_filter  # apply low-pass filter to the 'hku.png' image in the frequency domain.
    f_ishift = np.fft.ifftshift(low_pass_frequency)  # shift the low-frequency parts to corners.
    low_pass_back = cv2.idft(f_ishift)  # transform the new frequency map back to the spatial domain, using cv2.idft().
    low_pass_back = cv2.magnitude(low_pass_back[:, :, 0], low_pass_back[:, :, 1])

    # use the high-pass filter to filter out low frequencies
    high_pass_filter = np.ones((h, w, 2))  # create a high-pass filter
    for v in range(h):
        for u in range(w):
            distance = (v - h_c) ** 2 + (u - w_c) ** 2
            if distance <= cutoff ** 2:
                high_pass_filter[v, u, :] = 0  # set low frequencies to 0

    high_pass_frequency = dft_shift * high_pass_filter  # apply high-pass filter to the 'hku.png' image in the frequency domain.
    f_ishift = np.fft.ifftshift(high_pass_frequency)  # shift the low-frequency parts to corners.
    high_pass_back = cv2.idft(f_ishift)  # transform the new frequency map back to the spatial domain, using cv2.idft().
    high_pass_back = cv2.magnitude(high_pass_back[:, :, 0], high_pass_back[:, :, 1])

    low_pass_back = np.uint8(np.minimum(low_pass_back, 255))
    high_pass_back = np.uint8(np.maximum(np.minimum(high_pass_back, 255), 0))
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(low_pass_back, cmap='gray')
    plt.title('low_pass_image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(high_pass_back, cmap='gray')
    plt.title('high_pass_image'), plt.xticks([]), plt.yticks([])
    plt.show()

##################################################################


if __name__ == "__main__":
    img = cv2.imread('hku.png', 0)
    dft_shift = DFT(img)
    high_low_pass_filter(dft_shift)
