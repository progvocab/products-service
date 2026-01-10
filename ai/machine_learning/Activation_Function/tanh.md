The **tanh (hyperbolic tangent) activation function** squashes input values into the range **−1 to +1**, making outputs **zero-centered**.

Mathematically:
[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
]

It is **smooth and differentiable**, which helps gradient-based learning, but it can suffer from **vanishing gradients** for large input values.

**Why used in GANs:** Generator outputs are often normalized to **[−1, +1]**, so `tanh` matches the image data range naturally.
