An **upsample layer** is a layer that **increases the spatial resolution** (height and width) of feature maps, not a filter in the convolution sense.

* It **does not learn patterns** like a filter; it **resizes** feature maps.
* Common methods: **nearest-neighbor**, **bilinear interpolation**, or **transposed convolution**.
* Used in **decoders**, **segmentation**, **autoencoders**, and **GAN generators**.

**One-liner:**

> Upsampling increases feature map size, while filters extract features.
