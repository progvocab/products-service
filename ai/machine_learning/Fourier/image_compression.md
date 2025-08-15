Here’s a practical, end-to-end guide to compressing a **grayscale image** using the (2-D) Fourier transform, plus a compact NumPy example you can run as-is.

# Why Fourier helps

Most natural images have most of their “energy” in **low frequencies** (smooth variations) and relatively little in **high frequencies** (fine detail and noise). In the **frequency domain**, you can aggressively reduce or discard many small-magnitude coefficients with limited visual impact, then reconstruct the image from the remaining ones.

> Note: JPEG uses the **DCT** (a cousin of the Fourier transform) on 8×8 blocks because it packs energy into fewer real coefficients and reduces boundary artifacts. Since you asked for Fourier, we’ll use the 2-D FFT here; the workflow is nearly identical.

---

## Compression pipeline (FFT version)

1. **Prepare**

   * Convert to grayscale array `X` of shape `(H, W)`, dtype `float32` or `float64`. Optionally scale to `[0,1]`.

2. **Forward transform**

   * Compute the centered 2-D FFT:
     `F = np.fft.fftshift(np.fft.fft2(X))`.

3. **Coefficient selection (sparsification)**
   Choose one strategy (common ones below):

   * **Keep top-K by magnitude**: preserve the K largest `|F(u,v)|`, zero the rest.
   * **Energy threshold**: keep the smallest set whose squared magnitudes sum to ≥ a fraction `α` (e.g., 99%) of total energy.
   * **Low-pass mask**: keep a centered square/circle in frequency space (simple, fast; good results).

4. **(Optional) Quantization**

   * Round retained coefficients to fewer bits (e.g., uniform or µ-law). This is what really drives bit-rate down before entropy coding.

5. **Entropy coding (optional, but helpful)**

   * Store (index, quantized value) pairs for non-zeros with a lossless coder (e.g., run-length + Huffman/Arithmetic).

6. **Reconstruction**

   * Place the retained (de-quantized) coefficients back into an all-zeros `F_hat` of size `(H, W)`.
   * Invert: `X_hat = np.real(np.fft.ifft2(np.fft.ifftshift(F_hat)))`.
   * Clip to valid range and cast to `uint8` if needed.

7. **Measure**

   * **Compression ratio** = original bits / compressed bits.
   * **PSNR** (dB) for quality:
     `PSNR = 20*log10(MAX_I) − 10*log10(MSE)`, where `MAX_I` = 1.0 (if normalized) or 255 and `MSE = mean((X − X_hat)^2)`.

---

## Compact NumPy example (keep top-p% coefficients)

```python
import numpy as np
from imageio import imread, imwrite

# 1) Load grayscale image (H x W)
X = imread("lena.png", pilmode="L").astype(np.float64)  # or any grayscale image
Xn = X / 255.0

# 2) Forward FFT (centered)
F = np.fft.fftshift(np.fft.fft2(Xn))

# 3) Keep top-p% coefficients by magnitude
p = 5.0  # percent to keep; try 1, 2, 5, 10
mag = np.abs(F).ravel()
K = int(np.ceil((p/100.0) * mag.size))
thresh = np.partition(mag, -K)[-K]  # K-th largest magnitude
mask = (np.abs(F) >= thresh)
F_hat = F * mask  # zero out small coefficients

# 4) (Optional) quantize retained coeffs (simple uniform example)
# qstep = 1e-3
# Fq = np.sign(F_hat) * np.floor(np.abs(F_hat)/qstep + 0.5) * qstep
# F_hat = Fq

# 5) Inverse FFT
X_hat = np.real(np.fft.ifft2(np.fft.ifftshift(F_hat)))
X_hat = np.clip(X_hat, 0, 1)

# 6) Quality metrics
mse = np.mean((Xn - X_hat)**2)
psnr = 20*np.log10(1.0) - 10*np.log10(mse + 1e-12)

print(f"Kept {p}% non-zeros -> sparsity {(mask.sum()/mask.size)*100:.2f}%")
print(f"MSE: {mse:.6f}  PSNR: {psnr:.2f} dB")

# 7) Save reconstructed image
imwrite("recon.png", (X_hat*255).astype(np.uint8))
```

**How to choose `p` / threshold**

* Start with **1–5%** kept coefficients. Many images still look good at 2–3%.
* For scientific/line art, you may need higher p (edges are high-frequency).

---

## Design choices & tips

* **Top-K vs low-pass mask**

  * *Top-K* gives the best use of a coefficient budget (keeps the most “important” ones).
  * *Low-pass mask* is cheaper to encode (just store radius/size), but can blur edges more.

* **DCT vs FFT**

  * DCT (esp. block DCT) yields real coefficients and better **energy compaction** with fewer boundary artifacts; it’s preferred for compression (e.g., JPEG).
  * FFT is convenient for whole-image global filtering and for educational clarity.

* **Blocking vs global**

  * **Global FFT** can cause ringing (Gibbs) near sharp edges when many highs are removed.
  * **Block transforms** (e.g., 8×8 DCT/FFT) localize artifacts and are easier to code sparsely, but may show blockiness at low bitrates (mitigate with deblocking filters or overlapped blocks).

* **Quantization is key**

  * Simply zeroing coefficients reduces *sparsity* but not necessarily *bitrate*. To actually compress, quantize retained values (fewer distinct symbols), then entropy-code.

* **Complexity**

  * FFT is $O(HW \log(HW))$; selecting top-K with `np.partition` is $O(N)$ average, where $N=HW$.

* **Artifacts & mitigation**

  * **Ringing**: retain a bit more high-freq, or use windowing before FFT, or switch to block DCT.
  * **Blurring**: increase `p` or use edge-aware masks (e.g., keep more high-freq where gradients are large).

---

## If you want JPEG-style (block DCT) quickly

* Replace the global FFT with an **8×8 DCT** per block, quantize with a perceptual table, zig-zag scan, run-length + Huffman. NumPy’s `scipy.fftpack.dct` or `scipy.fft.dctn` makes this straightforward.

---

If you share a sample image and a target file size/ratio, I can tailor the thresholding/quantization to hit your exact bitrate and show the PSNR/SSIM you’ll get.
