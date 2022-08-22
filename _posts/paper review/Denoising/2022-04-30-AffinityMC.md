---
title: Interactive Monte Carlo Denoising using Affinity of Neural Features
# cover: ../assets/images/AffinityMC_cover.png
date: '2022-04-30'
categories: [Paper review, MC denoising]
# tags: [paper-review]
author: saha
math: true # do not change
mermaid: true
pin : false
---

# Abstract

High-quality denoising of Monte Carlo low-sample renderings remains a critical challenge for practical interactive ray tracing. We present a new learning-based denoiser that achieves state-of-the-art quality and runs at interactive rates. Our model processes individual path-traced samples with a lightweight neural network to extract per-pixel feature vectors. The rest of our pipeline operates in pixel space. We define a novel pairwise affinity over the features in a pixel neighborhood, from which we assemble dilated spatial kernels to filter the noisy radiance. Our denoiser is temporally stable thanks to two mechanisms. First, we keep a running average of the noisy radiance and intermediate features, using a per-pixel recursive filter with learned weights. Second, we use a small temporal kernel based on the pairwise affinity between features of consecutive frames. Our experiments show our new affinities lead to higher quality outputs than techniques with comparable computational costs, and better high-frequency details than kernel-predicting approaches. Our model matches or outperfoms sota offline denoisers in the low-sample count regime (2–8 samples per pixel), and runs at interactive frame rates at 1080p resolution.


# introduction

대부분의 sota MC denoiser들은 **large kernel-predicting network** 을 사용한다. Off-line 상황에서는 이들의 computational cost가 큰 문제가 되지 않지만, interactive application에서 사용하기에는 무거운 감이 없지않아 있다. 그래서 빠른 denoiser들은 아직도 hand-crafted filter이나 compact neural network를 통해 quality를 희생해서 performance를 챙긴다. 추가적으로, video animation의 경우에는 denoising artifact들이 극대화되어 flickering같은 문제가 발생할 가능성도 있다. 

본 논문에서는 이를 해결하고자 lightweight neural network를 사용해 per-sample information을 low-dimensional per pixel feature vector로 바꿔서 학습을 진행한다. 여기에서 feature간의 novel pairwise affinity라는 개념을 사용해 neighboring per-pixel radiance value들의 weight를 정해준다. 

# Denoising with learned pairwise affinity

Interactive application에서는 낮은 spp로 rendering된 image를 사용하기 때문에, temporal stability를 유지하는 것이 정말 힘들다. 본 논문은 recursive filter을 사용해 frame간에 sample information의 평균을 구하고, denoising kernel을 frame들에 쌍으로 적용해서 이를 해결하고자 한다. 전체 framework은 아래에서 확인할 수 있다. 

<img src="/assets/images/AffinityMC_1.png" width="100%" height="100%">*전체 framework*

<!-- 전체 과정은 다음과 같다. 먼저 small pointwise network으로 per-sample embedding $\mathbf{e}_{xyt}$를 만들어준다. 이때는 각 sample에 대한 정보를 사용하게 되는데, 이 단계에서만 이를 사용하고 이후에는 모두 pixel space에서 이루어진다. 만들어진 embedding에 temporal recursive filter을 적용시켜 frame간에 information을 propagate 시키고, sample embedding을 각 pixel에 있는 sample들에 대해 평균을 내준다. 이렇게 만들어진 per-pixel embedding에 convolution을 적용시켜 filtering kernel을 만들고, 이는 noisy radiance에 적용되어 denoised image를 생성하게 된다.  -->

## Input path-traced sample features

Sample에 대한 정보를 사용하면 pixel 정보만 사용하는 것에 비해 denoising의 성능이 좋아지지만, 그만큼의 computational overhead가 생기게 된다. 본 논문에서는 이 문제를 해결하기 위해 sample의 정보를 filtering weight를 만드는 데에는 사용하지만, filter 자체는 pixel radiance에 적용한다. Rendering 하는 과정에서 sample당 18차원의 feature vector $\mathbf{r}_{xyst}$를 저장한다. 여기에는 diffuse radiance(3), specular radiance(3), normal(3), depth(1), roughness(1), albedo(3)와 추가적인 4 binary variables 'emissive', 'metallic', 'transmissive', 'specular-bounce'가 포함되어 있다. 

## Mapping samples to per-pixel features

Sample space에서의 process를 최소화 하기 위해, per sample feature $\mathbf{r}_ {xyst}$ 는 얕은 fully connected network을 통해 per pixel embedding $\mathbf{e}_{xyt}$으로 변화하게 된다(\ref{eq1}). 

---

$$
\mathbf{e}_{xyt} = \frac{1}{S}\sum_{s=1}^{S}\text{FC}(\mathbf{r}_{xyst}) \label{eq1} \tag{1}
$$

---

## Spatio-temporal feature propagation

만들어진 per-pixel embedding $\mathbf{e}_ {xyt}$는 lightweight U-net을 통해 process된다. U-net은 현재 frame 뿐만 아니라 이전의 frame에 대한 embedding을 받아서 feature vector $\mathbf{f}^k_{xyt}$와 여러 scalar들을 생성한다(\ref{eq2}).

---

$$
\text{UNet}(\mathbf{e}_{xyt}, \mathcal{W}_t\bar{\mathbf{e}}_{xy,t-1}) = \left( \mathbf{f}^k_{xyt}, a^k_{xyt}, b^k_{xyt}\right), b^k_{xyt}, \lambda^k_{xyt} \label{eq2} \tag{2}
$$

---

 이때 k는 각 kernel을 의미하고, 총 $K$개의 dilated spatial kernel이 만들어지게 된다. Kernel들은 affinity feature $\mathbf{f}^k_{xyt}$들 사이의 distance에 scaling factor $a^k_{xyt}$를 곱해서 만들어지고, $c^k_{xyt}$는 kernel의 central weight가 된다. $b^k_{xyt}$는 인접한 frame 사이의 feature affinity를 조절하기 위한 parameter이고, $\lambda^k_{xyt}$는 인접한 frame의 기여도를 정하는 parameter이다. $\mathcal{W}_t$는 frame $t-1$를 frame $t$로 reproject하는 warping factor 이고, $\bar{\mathbf{e}} _{xyt}$는 pixel embedding들의 temporal accumulation으로, 위의 parameter을 이용해 다음과 같이 정의된다(\ref{eq3}).

---

$$
\begin{cases}
\bar{\mathbf{e}}_{xy0} = \mathbf{e} _{xy0} \\
\bar{\mathbf{e}}_{xyt} = (1-\lambda _{xyt})\mathbf{e} _{xyt} + \lambda _{xyt}\mathcal{W}_t\mathbf{e} _{xy,t-1}
\end{cases} \label{eq3} \tag{3}
$$

--- 
 
## Spatial kernels from pairwise affinity

Spatial filtering kernel은 다음과 같이 정의된다(\ref{eq4}).

---

$$
w^k_{xyuvt} = 
\begin{cases}
c^k{xyt} &\text{if } x=u \text{ and } y=v, \\
\text{exp}\left( -a^k_{xyt}||\mathbf{f}^k_{xyt} - \mathbf{f}^k_{uvt}||^2_2 \right) &\text{otherwise.}
\end{cases} \label{eq4} \tag{4}
$$

---

Kernel의 center weight를 다른 parameter로 정해주는 이유는 outlier handling에 도움이 되기 때문이다. 만약 어떠한 pixel의 값이 MC noise 측면에서 outlier이라고 판단되면, $c^k_{xyt}$를 0과 가깝게 만들어서 그 pixel이 영향을 억제시킬 수 있다. 

<img src="/assets/images/AffinityMC_2.png" width="100%" height="100%">*Outlier pixel의 경우, kernel의 center weight를 다른 parameter를 이용해서 정하면 이를 효과적으로 억제할 수 있다.*

## Temporally-stable kernel based denoising

Filtering을 하기에 앞서, noisy radiance는 frame에 따라 일정한 가중치를 두고 accumulate 된다. 이는 특정한 motion이 있는 부분 등에 많은 도움을 주고, 전체적인 temporal stability를 향상시킨다고 한다(\ref{eq5}). 

---

$$
\begin{cases}
\bar{\mathbf{L}}_{xy0} = \mathbf{L}_{xy0} \\
\bar{\mathbf{L}}_{xyt} = (1-\lambda_{xyt})\mathbf{L}_{xyt} + \lambda_{xyt}\mathcal{W}_t\bar{\mathbf{L}}_{xyt}
\end{cases} \label{eq5} \tag{5}
$$

---

이때 $\lambda_{xyt}$는 앞서 U-Net을 통해 얻어진 parameter이다(\ref{eq2}). 

Filtering은 아래와 같이 이루어진다(\ref{eq6}).

---

$$
\mathbf{L}^{(k)}_{xyt} = \frac{\sum_{u,v}w^k\mathbf{L}^{(k-1)}_{uvt}}{\epsilon + \sum_{u,v}w^k_{xyuv}} \label{eq6} \tag{6}
$$

---

이때 $\epsilon = 10^{-10}$ 는 0으로 나누는 것을 방지하기 위한 작은 상수이다. 

마지막 filtering step에서는 현재 frame과 바로 이전 frame의 feature간의 affinity를 측정하는 temporal kernel을 사용한다. Kernel의 weight은 다음과 같다(\ref{eq7}).

---

$$
\omega_{xyuvt} = \text{exp}\left( -b_{xyt}||\mathbf{f}^K_{xyt} - \mathcal{W}_t\mathbf{f}^K_{uv,t-1}||^2_2 \right) \label{eq7} \tag{7}
$$

---

최종 output은 다음과 같이 얻을 수 있다(\ref{eq8}).

---

$$
\mathbf{O}_{xyt} = \frac{\sum_{u,v}w^K_{xyuv}\mathbf{L}^{(K-1)}_{uvt} + \sum_{u^{'},v^{'}}\omega_{xyu^{'}v^{'}}\mathcal{W}_t\mathbf{O}_{u^{'}v^{'},t-1}}{\epsilon + \sum_{u,v}w^K_{xyuv} + \sum_{u^{'},v^{'}}\omega_{xyu^{'}v^{'}}} \label{eq8} \tag{8}
$$

---

# Training details

## Dataset

[SBMC](https://ee12ha0220.github.io/posts/SBMC/)의 scene generator을 사용해서 dataset을 만들었다. 여기에 추가적으로 camera translation, rotation에 기반한 motion을 추가해서 넣었다. 

## Loss

전체 loss function은 다음과 같다(\ref{eq9}).

---

$$
\mathcal{L} = \mathcal{L}_\text{recons} + 0.25\cdot\mathcal{L}_\text{temporal}+10^{-5}\cdot\mathcal{L}_\text{reg} \label{eq9} \tag{9}
$$

---

Reconstruction loss($\mathcal{L}_ \text{recons}$), temporal loss($\mathcal{L}_ \text{temporal}$)에는 SMAPE가 사용되었고, $\mathcal{L}_ \text{reg}$는 $L_2$ regularization loss이다. 

더 자세한 정보는 [AffinityMC 논문](https://www.mustafaisik.net/anf/)을 참조하길 바란다. 

# Result

이전의 kernel based 연구들보다 더 빠른 속도를 보였고, 성능을 조금 내려놓은 interactive 연구들보다는 속도가 느렸지만 kernel based 연구들보다 성능이 좋을 정도로 성능 면에서 차별점을 보였다. 더 자세한 정보는 [AffinityMC 논문](https://www.mustafaisik.net/anf/)을 참조하길 바란다. 

<img src="/assets/images/AffinityMC_3.png" width="100%" height="100%">*속도와 성능을 종합해봤을 때 상당히 좋은 결과를 보인다.*

<img src="/assets/images/AffinityMC_4.png" width="100%" height="100%">*성능 면에서 많은 improvement가 있었다.*

# Conclusion

We have presented a novel method for denoising Monte Carlo renderings at interactive speeds with quality on-par with off-line denoisers. We use an efficient network to aggregate relevant per-sample features into temporally-stable per-pixel features. Pairwise affinity between these features are used to predict dilated 2D kernels that are iteratively applied to the input radiance to produce the final denoised result. We show our model can spatially adjust the kernels to effectively smooth out noise and preserve fine details. We further demonstrate how to incorporate the spatially-warped content from previous frames to produce a temporally consistent result.
