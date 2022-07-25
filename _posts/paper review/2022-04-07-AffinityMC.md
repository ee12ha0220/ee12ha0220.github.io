---
title: Interactive Monte Carlo Denoising using Affinity of Neural Features
layout: post # do not change
current: post # do not change
cover: ../assets/images/AffinityMC_cover.png
use_cover: false
navigation: true
# date: #'2022-04-07 19:00:00'
tags: paper-review
class: post-template # do not change
subclass: post # do not change
author: saha # do not change
use_math: true # do not change
---

# introduction

### Limitations of previous work
- State of art MC denoisers use **large kernel-predicting neural networks** (Bako et al. 2017, Gharbi et al. 2019, Kettunen et al. 2019, Vogels et al. 2018, Xu et al. 2019), which have **large computational cost** for interactive applications.  
- **Faster denoisers** using hand-designed filters(Mara et al. 2017), or a compact neural network(Chaitanya et al. 2017, Meng et al. 2020) **sacrifices quality for performance**. 
- Interactive denoisers have trouble in maintaining **temporal stability** in certain scenes like soft shadows, complex global illumination, glossy reflections, refractive materials and etc..
    - Denoising artifacts are often amplified in video animation(flickering), therefore temporal-stability is important. 

### Objective of this paper
- In this paper, they propose a new denoiser for low-sample, interactive ray-tracing applications that directly operates on the path-traced samples. 
    - A light-weight neural network that summarizes **high-dimensional per-sample information** into **low-dimensional per-pixel feature vectors**. 
    - A novel **pairwise affinity** over these features, which is used to **weight the contributions of neighboring per-pixel radiance values** in a local weighted average filtering step. 
    - A new temporal aggregation mechanism which uses pairwise affinity to improve temporal stability. 

# Denoising with learned pairwise affinity

### Overall pipeline
<img src="/assets/images/AffinityMC_1.png" width="100%" height="100%"> 

### Input path-traced sample features
- **Working with samples** shows **improved denoising** compared to using pixels, but has a **computational overhead**. 
    - In this paper, they addressed this problem by using **per-sample information in computing weights**, but the filters **operate in integrated pixel radiance**. 
- During rendering, they store a 18-dim feature $\mathbf{r}_{xyst}$ for each sample. 
    - (x,y) : pixel coordinates
    - s : sample indices within a pixel
    - t : frame indices
    - contents : 
        - **Radiance**(split into diffuse and specular components)
        - **Geometric features**(normal[3], depth[1])
        - **Material information**(roughness[1], albedo[3])
        - **Binary variables** : 
            - **emissive** — indicates whether the path sampled hits emissive surface
            - **metallic** — differentiates between dielectric and conductors 
            - **transmissive** — distinguishes between reflections and refractions
            - **specular-bounce** — which is ‘true’ if first vertex on the camera path is a specular interaction

### Mapping samples to per-pixel features
- In this paper, they embed samples using a shallow FCN(uses leaky ReLU), then reduce the embeddings to per-pixel by averaging over the sample dimension

$$
\mathbf{e}_{xyt} = \frac{1}{S} \sum_{s=1}^S FC(\mathbf{r}_{xyst})
$$

### Spatio-temporal feature propagation

- The per-pixel embeddings are processed using a lightweight U-net. 
- Both current and previous frames are fed to the network. 
- Output of the UNet is given by : 
$$
(\mathbf{f}^k_{xyt}, a^k_{xyt}, c^k_{xyt}), b^k_{xyt}, \lambda^k_{xyt} = UNet(\mathbf{e}_{xyt}, \mathcal{W}_t\mathbf{\bar{e}}_{xy,t-1})
$$

- Spatial kernels are computed by calculating distances between affinity features $\mathbf{f}^k_{xyt}$, scaled by the bandwidth parameters $a^k_{xyt}$. 
- $c^k_{xyt}$ is the center weight of the kernel. 
- $b^k_{xyt}$ is a bandwidth parameter modulating the feature affinity between successive frames in a temporal kernel. 
- $\lambda^k_{xyt}$ is the parameter of an exponential moving average filter that accumulates the pixel embeddings and noisy radiance temporally. 
- $\mathcal{W}_t$ is a warping operator that reprojects frame t-1 to frame t with nearest neighbor interpolation using the geometric flow at the primary intersection point computed by the path tracer. 
- $\mathbf{\bar{e}}_{xy,t-1}$ is a temporal accumulation of the pixel embeddings defined by : 
$$
\begin{cases}
\mathbf{\bar{e}}_{xy0} = \mathbf{e}_{xy0}, \\
\mathbf{\bar{e}}_{xyt} = (1-\lambda_{xyt})\mathbf{e}_{xyt} + \lambda_{xyt}\mathcal{W}_t\mathbf{\bar{e}}_{xy,t-1}.
\end{cases}
$$
    - This helps make temporally consistent predictions, compared to simply passing the previous frame's embeddings
    - By setting $\lambda_{xyt}$ to 0 the warped embeddings can be removed, if they are inaccurate. 

### Spatial kernels from pairwise affinity
- The weight of the spatial filtering kernel is defined as below : 

$$
w^k_{xyuvt} = 
\begin{cases}
c^k_{xyt} & \mbox{if } x = u \mbox{ and } y = v, \\
exp(-a^k_{xyt}\lVert\mathbf{f}^k_{xyt} - \mathbf{f}^k_{uvt}\rVert^2_2) & otherwise.
\end{cases}
$$

- Setting $c^k_{xyt} = 1$ makes the center pixel **contribute fully to the output**, and setting $c^k_{xyt} = 0$ makes the network to **suppress the bright outliers**. 
    - These **bright outliers** often appear in **low-sample renderings when high-energy, low-probability paths are sampled**. 

### Temporally-stable kernel-based denoising
- Prior to filtering, the noisy radiance $\mathbf{L}_{xyt}$ is accumulated over time.
    - This improves overall temporal stability : 
    $$
    \begin{cases}
    \mathbf{\bar{L}}_{xy0} = \mathbf{L}_{xy0}, \\
    \mathbf{\bar{L}}_{xyt} = (1-\lambda_{xyt})\mathbf{L}_{xyt} + \lambda_{xyt}\mathcal{W}_t\mathbf{\bar{L}}_{xyt}.
    \end{cases}
    $$

- The first $K - 1$ kernels are sequentially applied as following where $\mathbf{L} ^{(0)} _{xyt} = \mathbf{\bar{L}} _{xyt}$ : 

$$
\mathbf{L}^{(k)}_{xyt} = 
\frac{\sum_{u,v}w^k_{xyuv}\mathbf{L}^{(k-1)}_{uvt}}
    {\epsilon + \sum_{u,v}w^k_{xyuv}}
$$

- Then a temporal kernel is obtained. 
    - This kernel measures the affinity between the features of the current and previous frame(after warping). 
    - The equation is as following : 
    $$
    \omega_{xyuvt} = exp(-b_{xyt} \lVert\mathbf{f}^K_{xyt} - \mathcal{W}_t\mathbf{f}^K_{uv,t-1}\rVert^2_2)
    $$
    - $w^k_{xyuvt}$ are the spatial kernels, and $\omega_{xyuvt}$ is the temporal kernel. 

- The final denoised image is obtained using the temporal kernel and the last spatial kernel : 

$$
\mathbf{O}_{xyt} = 
\frac
    {\sum_{u,v} w^K_{xyuv}\mathbf{L}^{(K-1)}_{uvt} + 
        \sum_{u^{'},v^{'}}\omega_{xyu^{'}v^{'}}\mathcal{W}_t \mathbf{O}_{u^{'}v^{'},t-1}}
    {\epsilon + \sum_{u,v}w^K_{xyuv} + \sum_{u^{'},v^{'}}\omega_{xyu^{'}v^{'}}}
$$

### Comparison to kernel-predicting networks

- Kernel-predicting methods require deeper and larger networks to fully benefit from larger kernels. 
    - The number of complexity of **pairwise interactions** between pixels **increases with kernel size**.
- This method does not require this because the per-pixel features are predicted with a **closed-form affinity**, rather than full-rank kernels. 
- Also, kernel size can be dynamically changed at runtime, without retraining.  

### Relation to the neural bilateral grid(Meng et al. 2020)

- Meng et al.[2020] used a bilateral grid [Gharbi et al.2017] for denoising, approximating a bilateral filter. 
- They used a 3D grid
    - 2D screen-space coordinates
    - A learned scalar parameter which would correspond to the range filter in a traditional bilateral filter [Tomasi and Manduchi
1998]
- This is similar to using feature vector with dimension 1 (in this paper they used 8), and they say this leads to oversmoothing. 


# Dataset and training procedure

### Dataset
- Used scene generator from Gharbi et al. [2019]

### Losses
- They aim to minimize reconstruction loss, temporal stability loss, and a regularization on the affinity of parameters. 
$$
\mathcal{L} = \mathcal{L}_{recons} + 0.25 \cdot \mathcal{L}_{temporal} + 10^{-5} \cdot \mathcal{L}_{reg}
$$

    - Symmetric Mean Absolute Percentage Error(SMAPE) was used for the losses. 

    $$
    SMAPE(A, B) = 
    \frac{1}{3}\Bbb{E}_{xyt} \frac{\lVert A_{xyt}-B_{xyt}\rVert_1}{\lVert A_{xyt}\rVert_1+\lVert B_{xyt}\rVert_1+\epsilon},\, \lVert.\rVert_1 : L_1 \mbox{ norm}\\
    \mathcal{L}_{recons} = SMAPE(\mathbf{O}, \mathbf{O^{\star}}) ,\,
    \mathcal{L}_{temporal} = SMAPE(\partial_t\mathbf{O}, \partial_t\mathbf{O^{\star}})
    $$
    
    - Regularization is an $L_2$ penalty over the kernel's bandwidth parameters

    $$
    \mathcal{L}_{reg} = \sum_k\Bbb{E}_{xyt}\lVert a^k_{xyt}\rVert^2_2 + 
    \Bbb{E}_{xyt}\lVert b_{xyt}\rVert^2_2
    $$

# Results
- Refer to [original paper](https://dl.acm.org/doi/10.1145/3450626.3459793)

