Flash Attention is an algorithm that computes the **same attention output** as standard Transformer attention, but **much faster and with much lower GPU memory usage**. The key insight is that the bottleneck isn't the arithmetic—it's moving data between GPU memory (HBM) and fast on-chip memory (SRAM).

## The standard attention algorithm




  Query matrix:
  
$$ (Q \in \mathbb{R}^{N \times d})  $$

 Key matrix: 
 
 $$   (K \in \mathbb{R}^{N \times d})  $$

  Value matrix:

 $$   (V \in \mathbb{R}^{N \times d})   $$



Standard attention computes:


 $$   [
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
]   $$   


This involves three steps:

1. Compute the score matrix

 $$ 
[
S = QK^T
]
 $$ 

 
Shape: 

 $$  (N \times N)  $$ 

2. Apply softmax

 $$  [
P = \text{softmax}(S)
]  $$ 

3. Multiply by values

 $$ [
O = PV
]  $$ 

### The problem

For sequence length (N):

* (Q,K,V): (O(Nd))
* Score matrix (S): **(O(N^2))**
* Probability matrix (P): **(O(N^2))**

The large (N^2) matrices dominate memory.

For example:

* (N=16,384)
* FP16

One attention matrix is

 $$  [
16384^2 \times 2 \approx 512\text{ MB}
]  $$ 

Two such matrices exceed **1 GB**.

Even worse, these matrices are repeatedly written to and read from GPU global memory.

 

### Why GPUs become slow

Modern GPUs have

* Huge compute throughput
* Relatively limited memory bandwidth

Typical numbers:

* HBM bandwidth: ~1–3 TB/s
* SRAM bandwidth: >20 TB/s

Moving data from HBM is much slower than doing floating point operations.

Standard attention does:

```
HBM
 ↓
Load Q,K
 ↓
Compute QKᵀ
 ↓
Write S to HBM
 ↓
Read S
 ↓
Softmax
 ↓
Write P
 ↓
Read P
 ↓
Multiply by V
```

There are many expensive memory transfers.
 

### Flash Attention idea

Instead of computing the whole (N \times N) matrix:

**Process attention block by block.**

Example:

Split into blocks of 128 rows.

Instead of

```
Q × Kᵀ
```

compute

```
Q_block × K_blockᵀ
```

inside SRAM.

After using the block:

* never store it
* discard it
* move to next block
 
## Tiling

Imagine

```
Q
████████████

K
████████████
```

Instead of

```
Entire Q × Entire K
```

Flash Attention computes

```
┌────┬────┬────┐
│■■■■│    │    │
├────┼────┼────┤
│    │■■■■│    │
├────┼────┼────┤
│    │    │■■■■│
└────┴────┴────┘
```

Each tile fits into SRAM.

 

### But softmax needs the whole row...

This is the clever part.

Normally,

 $$ 
[
\text{softmax}(x_i)=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
]
 $$ 
 

You need every element before computing the denominator.

Flash Attention instead performs **online softmax**.

For each row it keeps only

* current maximum
* running normalization factor
* running weighted output

Suppose scores arrive in blocks.

For each new block:

Update

### Running maximum

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})
$$


Then rescale previous sums.

Maintain

 $$ [
l=\sum e^{x_i-m}
]  $$ 

This gives the exact softmax without storing the whole row.

Memory per row becomes constant.
 

# Online softmax example

Scores:

```
[2,1,5,3]
```

Split into

```
[2,1]
[5,3]
```

First block:

max = 2

running sum

```
e^(0)+e^(-1)
```

Second block:

new max = 5

Rescale old sum

```
old_sum × e^(2-5)
```

Then add

```
e^(0)+e^(-2)
```

Exactly the same result as ordinary softmax.

No large matrix stored.

 

### Computing the output

Flash Attention never stores

 $$ [
P=\text{softmax}(QK^T)
]  $$ 

Instead it immediately computes

 $$ [
PV
]  $$ 

inside the block.

Workflow:

```
Load Q block
Load K block
Load V block

↓

Compute scores

↓

Online softmax

↓

Multiply with V

↓

Accumulate output

↓

Discard scores
```

Nothing intermediate is written back.

 

### Complexity

### Standard

Memory

 $$ [
O(N^2)
]  $$ 

Compute

 $$ [
O(N^2d)
] $$ 

 

### Flash Attention

Memory

 $$  [
O(Nd)
]  $$ 

Compute

Still

 $$ [
O(N^2d)
]  $$ 

Notice:

The algorithm performs essentially the **same amount of arithmetic**, but dramatically reduces memory traffic. Since GPUs are often memory-bandwidth limited, this yields large speedups.

 

### Why it's faster despite the same FLOPs

Suppose arithmetic costs

```
100 units
```

Memory movement costs

```
500 units
```

Standard attention

```
100 compute
500 memory

=600
```

Flash Attention

```
100 compute
100 memory

=200
```

Same math.

Much less waiting on memory.

 

### GPU execution

Each GPU thread block:

```
Load Q tile

for each K tile:

    load K
    load V

    compute scores

    online softmax

    accumulate output

Write final output
```

Every tile stays in SRAM while being processed.

 

### Why it's called "Flash"

The name comes from the idea of making attention **I/O-aware**:

* Minimize reads/writes to slow GPU memory.
* Keep data in fast on-chip memory as long as possible.
* Fuse multiple operations (matrix multiplication, softmax, and value multiplication) into a single kernel, avoiding intermediate memory writes.

 

### FlashAttention versions

* **FlashAttention (v1):** Introduced block-wise attention with online softmax, greatly reducing memory usage while preserving exact results.
* **FlashAttention-2:** Improved GPU work partitioning and parallelism, increasing throughput on modern GPUs.
* **FlashAttention-3:** Further optimized for newer GPU architectures (such as NVIDIA Hopper), improving utilization and supporting lower-precision formats like FP8.

### In one sentence

Flash Attention computes **exact Transformer attention** by processing small blocks that fit in fast GPU memory, using an **online softmax** to avoid materializing the full (N \times N) attention matrix. This reduces memory usage from **(O(N^2)) to (O(Nd))** while keeping the computational complexity the same, resulting in significantly faster execution on modern GPUs.
