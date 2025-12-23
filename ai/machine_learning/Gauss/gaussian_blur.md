Gaussian blur is commonly used in **drone imagery** to reduce noise, smooth textures, and suppress high-frequency details that can interfere with downstream processing like detection, mapping, or stitching.

## What Gaussian Blur Does

Gaussian blur convolves the image with a Gaussian kernel, where nearby pixels contribute more than distant ones. This results in smooth, natural-looking blurring without sharp artifacts.

## Why Gaussian Blur Is Useful for Drone Images

1. **Noise reduction**
   Drone cameras often introduce sensor noise due to vibration, wind, low light, or high ISO. Gaussian blur smooths this noise.

2. **Preprocessing for computer vision**
   Improves performance of:

   * Edge detection (Canny, Sobel)
   * Object detection (vehicles, people, buildings)
   * Feature matching (SIFT, ORB)
   * Image segmentation

3. **Improved orthomosaic & stitching**
   Reduces minor texture inconsistencies between overlapping frames.

4. **Privacy masking**
   Used to blur faces, license plates, or sensitive locations in aerial imagery.

5. **Anti-aliasing before downsampling**
   Prevents aliasing when resizing large drone images.

## Key Parameters

### Kernel Size (ksize)

* Must be **odd** (e.g., 3×3, 5×5, 11×11)
* Larger kernel → stronger blur

### Sigma (σ)

* Controls blur intensity
* Higher σ → smoother image
* If σ = 0, OpenCV computes it from kernel size

**Typical drone image values**

* Light smoothing: `ksize=3–5`, `σ=0.5–1.0`
* Moderate smoothing: `ksize=7–11`, `σ=1.5–3.0`
* Heavy blur (privacy): `ksize=21+`, `σ=5+`

## OpenCV Example (Python)

```python
import cv2

img = cv2.imread("drone_image.jpg")
blurred = cv2.GaussianBlur(img, (7, 7), 1.5)

cv2.imwrite("blurred.jpg", blurred)
```

## OpenCV Example (Java)

```java
Mat src = Imgcodecs.imread("drone_image.jpg");
Mat dst = new Mat();

Imgproc.GaussianBlur(src, dst, new Size(7, 7), 1.5);
Imgcodecs.imwrite("blurred.jpg", dst);
```

## When NOT to Use Gaussian Blur

* When preserving sharp edges is critical (use **bilateral filter** instead)
* Before fine-grained object recognition where detail matters
* On already low-resolution drone footage

## Alternatives for Drone Images

* **Median Blur** – better for salt-and-pepper noise
* **Bilateral Filter** – smooths while preserving edges
* **Non-Local Means** – high-quality denoising (slower)
* **Anisotropic diffusion** – terrain and satellite-style imagery

## Best Practice for Drone Pipelines

1. Stabilize image (if needed)
2. Apply **light Gaussian blur**
3. Run feature detection / ML inference
4. Avoid blurring raw data used for photogrammetry depth estimation

If you want, I can recommend **exact blur parameters** based on:

* Altitude
* Sensor resolution
* Use case (mapping, surveillance, object detection)
* Day vs night flight
