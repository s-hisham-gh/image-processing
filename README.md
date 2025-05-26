The following image processing techniques are implemented in **task1** of this repository:

1. Bilinear Interpolation
   - Script: `bilinear_interpolation.py`
   - Description: This script performs bilinear interpolation to resize an image while maintaining smooth transitions between pixels.
   - Input: `test1.png`, `test2.jpg`, `test3.jpg`
   - Output: Resized images with smoother edges.

2. Histogram Equalization
   - Script: `hist_equalization.py`
   - Description: This script enhances the contrast of an image by applying histogram equalization, which redistributes pixel intensities to improve visibility.
   - Input: `test1.png`, `test2.jpg`, `test3.jpg`
   - Output: Enhanced images with improved contrast.

3. Laplacian Sharpening
   - Script: `laplace_sharpening.py`
   - Description: This script sharpens images using the Laplacian filter, which emphasizes edges and fine details.
   - Input: `test1.png`, `test2.jpg`, `test3.jpg`
   - Output: Sharpened images with enhanced edges.


**How to use**:
   ```bash
   git clone https://github.com/s-hisham-gh/image-processing.git 
   cd image-processing/task1

The following image processing techniques are implemented in **task2** of this repository:

1. **Discrete Fourier Transform (DFT)**
   - Script: `discrete_fourier_transform.py`
   - Description: This script performs the Discrete Fourier Transform on an image to analyze its frequency components.
   - Input: `hku.png`
   - Output: Frequency domain representation of the image.

2. **JPEG Compression**
   - Script: `jpeg_compression.py`
   - Description: This script demonstrates JPEG compression by reducing the image file size while maintaining visual quality.
   - Input: `hku.png`
   - Output: Compressed version of the image.

3. **Pepper-Salt Noise Restoration**
   - Script: `pepper_salt_restoration.py`
   - Description: This script removes pepper-salt noise from an image using median filtering or other restoration techniques.
   - Input: `blurry_hku.png`
   - Output: Restored version of the noisy image.

**How to use**:
   ```bash
   git clone https://github.com/s-hisham-gh/image-processing.git 
   cd image-processing/task2
