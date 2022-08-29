---
title: "PIXELCNN++: IMPROVING THE PIXELCNN WITH DISCRETIZED LOGISTIC MIXTURE LIKELIHOOD AND OTHER MODIFICATIONS"
# image : ""
date: '2022-07-25'
categories: [Paper review, Image synthesis]
# tags: [tag] 
author: saha
math: true
mermaid: true
pin : false
--- 

# Abstract

PixelCNNs are a recently proposed class of powerful generative models with tractable likelihood. Here we discuss our implementation of PixelCNNs which we make available [here](https://github.com/openai/pixel-cnn). Our implementation contains a number of modifications to the original model that both simplify its structure and improve its performance. 1) We use a discretized logistic mixture likelihood on the pixels, rather than a 256-way softmax, which we find to speed up training. 2) We condition on whole pixels, rather than R/G/B sub-pixels, simplifying the model structure. 3) We use downsampling to efficiently capture structure at multiple resolutions. 4) We introduce additional short-cut connections to further speed up optimization. 5) We regularize the model using dropout. Finally, we present state-of-the-art log likelihood results on CIFAR-10 to demonstrate the usefulness of these modifications.

# Introduction

[PixelRNN 논문](https://ee12ha0220.github.io/posts/PixelRNN/)에서 소개된 PixelCNN은 image를 위한 tractable generative model이다. 이는 image $\mathbf{x}$의 모든 sub-pixel $x_i$에 대한 pdf $p(\mathbf{x}) = \prod_i p(x_i\|x_{<i})$ 로 정의된다. 본 논문(PixelCNN++)에서는 PixelCNN에 몇 가지 modification을 가해서 그 성능을 향상시켰다. 

# Modifications to PixelCNN

## Discretized logistic mixture likelihood

PixelCNN은 sub-pixel의 conditional distribution(color channel)을 256-way softmax로 modeling 한다. RGB color의 범위가 0~255이기 때문에, 이 사이의 값으로 discretize 하는 것이다. 이는 model에 flexibility를 제공해주지만, 동시에 memory 측면에서 많은 resource를 사용하게 된다. 그리고 discretize된 value $c$가 실제로 $c-1$과 $c+1$중 어디에 가까운지 알 수 없기 때문에, 이를 추가적으로 학습시켜야 해서 학습 속도가 정말 느리다. 그래서 PixelCNN++에서는 discretized pixel value의 conditional probability를 계산하는 다른 mechanism을 제시했다. Continuous distribution인 latent color intensity $\nu$가 있다고 가정하고, 이를 가장 가까운 자연수로 round하는 방법을 사용하는데, 이를 logistic distribution의 mixture으로 modeling 한다(\ref{eq1}). 

---

$$
\begin{align}
    \nu &\sim \sum_{i=1}^K\pi_i\text{logistic}(\mu_i, s_i) \\
    P(x|\pi, \mu, s) &= \sum_{i=1}^K \pi_i [\sigma((x+0.5-\mu_i)/s_i) - \sigma((x-0.5-\mu_i)/s_i)]
\end{align} \label{eq1} \tag{1}
$$

---

이때 $\sigma()$는 logistic sigmoid function이다. Edge case인 0과 255에서는 각각 $x-0.5$를 $-\infty$로, $x+0.5$를 $\infty$로 바꿨다. 

## Conditioning on whole pixels

PixelCNN에서는 RGB 3개의 값을 분리시켜서 generative model을 만든다. 이는 굉장히 general한 structure을 보장해주지만, 동시에 model을 더 복잡하게 만든다. PixelCNN++에서는 color channel간의 dependency는 deep learning을 통해 알아내야 할 정도로 복잡하지 않다고 말하며, 각 channel의 mean이 다른 channel value와 linearly depend한다고 가정해 RGB값을 한번에 modeling 했다. 

## Using downsampling

PixelCNN에서는 작은 receptive field에서 CNN을 사용해준다. 이는 local dependency를 고려하기에는 좋지만, long range structure을 modeling하기에는 적합하지 않다. 그래서 PixelCNN++에서는 input image를 downsampling해서 receptive field를 증가시켜 줬다. 하지만 이는 정보의 손실을 불러일으킬 수 있는데, short-cut connection을 추가해서 이 문제를 해결했다. 

## Network architecture

$32 \times 32$크기의 input에 대해, PixelCNN++는 5개의 ResNet layer을 6 block만큼 사용한다. 1번과 2번, 그리고 2번과 3번 block 사이에는 strided convolution을 통한 subsampling이 이루어지고, 4번과 5번, 그리고 5번과 6번 block 사이에는 transpose된 strided convolution을 통한 upsampling이 이루어진다. 이 과정에서 정보 손실을 막기위해, 1번과 6번, 2번과 5번, 그리고 3번과 4번 block 사이에 short-cut connection을 추가했다. 이는 UNet과 비슷한 구조이다. 

<img src="/assets/images/PixelCNNpp_1.png" width="80%" height="80%">*PixelCNN++의 network 구조.*

추가적으로 regularization을 위한 dropout layer을 사용했다. 

# Training details

CIFAR-10 dataset을 사용했으며, PixelCNN과 마찬가지로 NLL loss을 사용했다. 

# Results

PixelCNN을 비롯한 이전의 연구들보다 더 좋은 성능을 발휘했다. 더 자세한 사항은 [PixelCNN++ 논문](https://arxiv.org/abs/1701.05517)을 참조하길 바란다. 

<img src="/assets/images/PixelCNNpp_2.png" width="80%" height="80%">*NLL을 bits per sub-pixel로 나타낸 표이다. PixelCNN++가 가장 성능이 좋음을 알 수 있다.*

<img src="/assets/images/PixelCNNpp_3.png" width="80%" height="80%">*왼쪽 : Class conditional generated images, 오른쪽 : Real CIFAR-10 images*

<img src="/assets/images/PixelCNNpp_4.png" width="80%" height="80%">*Short-cut connection을 사용하는 것이 성능이 훨씬 좋음을 알 수 있다.*

# Conclusion

We presented PixelCNN++, a modification of PixelCNN using a discretized logistic mixture likelihood on the pixels among other modifications. We demonstrated the usefulness of these modifications with state-of-the-art results on CIFAR-10. Our code is made available [here](https://github.com/openai/pixel-cnn) and can easily be adapted for use on other data sets.