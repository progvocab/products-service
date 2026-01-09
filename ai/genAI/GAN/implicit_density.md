**Implicit density in GAN** means that the model **does not explicitly compute or represent a probability density function**, but instead **learns to generate samples that follow the data distribution**.

---

## Intuition

* GAN learns by **sampling**, not by writing down ( p(x) )
* You can generate new data points
* But you **cannot directly evaluate likelihood** of a given sample

---

## Explicit vs Implicit Density

| Model     | Density type           | Can compute (p(x))? |
| --------- | ---------------------- | ------------------- |
| GMM       | Explicit               | ✅ Yes               |
| VAE       | Explicit (approximate) | ✅ Yes               |
| **GAN**   | **Implicit**           | ❌ No                |
| Diffusion | Explicit               | ✅ Yes               |

---

## How GAN learns implicit density

* Generator maps random noise (z) → data (x)
* Discriminator tells **real vs fake**
* No probability formula is ever learned

---

## Why this matters

**Pros**

* Can model very complex distributions
* Produces sharp, realistic samples

**Cons**

* No likelihood estimation
* Hard to evaluate and train

---

## One-line interview answer

> **In GANs, implicit density means the model learns a data distribution only through sample generation, without explicitly modeling or computing the probability density function.**

If you want, I can explain:

* Why GANs can’t compute log-likelihood
* Implicit vs explicit density with diagrams
* GAN vs VAE vs Diffusion comparison
