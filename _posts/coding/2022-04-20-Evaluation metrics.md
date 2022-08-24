---
title: 많이 사용하는 loss, Evaluation metrics
date: '2022-04-20'
categories: [coding]
# tags: [study]
author: saha # do not change
math: true # do not change
mermaid: true
pin : false
---

# Loss

## MSE(Mean Squared Error)

MSE는 그 이름을 보면 무엇을 의미하는지 알 수 있는데, 오차(error)의 제곱의 평균을 의미한다. 이를 수식으로 쓰면 다음과 같다. 

---

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{Y}_i - Y_i)^2
$$

---

# Evaluation metrics

## PSNR(Peak Signal-to-Noise Ratio)

PSNR은 image의 비교에서 자주 사용되는 metric이다. 한글로 번역하면 '최대 신호 대 잡음비'가 되지만, 잘 와닿는 것 같지는 않다. PSNR의 수식은 아래와 같다. 

---

$$
\text{PNSR} = 10\cdot\log_{10}\left( \frac{\text{MAX}^2}{\text{MSE}} \right)
$$

---

이때 MAX는 가능한 pixel의 최댓값으로, 일반적인 RGB image의 경우 255가 된다. 수식 상으로는 PSNR이 클수록 두 image가 서로 비슷해야 하지만, 인간이 시각적으로 느끼는 품질 차이와는 다르기 때문에 그 반대인 경우가 있을 수도 있다.

## SSIM(Structural Similarity Index Map)

SSIM은 수치적인 오차가 아니라 인간의 시각에서의 차이를 고려하기 위해 만든 metric이다. SSIM의 수식은 아래와 같다. 

---

$$
\text{SSIM}(x,y) = l(x,y)^\alpha\cdot c(x,y)^\beta\cdot s(x,y)^\gamma
$$

---

이때 l은 luminance, c는 contrast, s는 structural의 의미를 갖고 있다. 이 값들은 아래와 같이 정의된다. 

---

$$
\begin{align}
    &l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2+\mu_y^2+C_1} \\
    &c(x,y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2+\sigma_y^2+C_2} \\
    &s(x,y) = \frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}

\end{align}
$$

---
이때 $\mu_x, \mu_y, \sigma_x, \sigma_y, \sigma_{xy}$ 는 각각 image $x$, $y$의 mean, std, 그리고 cross-covariance를 의미한다. $C_1, C_2, C_3$는 특정한 상수로, 아래와 같이 정의된다. 

---

$$
\begin{align}
    &C_1 = (0.01\cdot L)^2 \\
    &C_2 = (0.03\cdot L)^2 \\
    &C_3 = C_2/2
\end{align}
$$

---

$L$은 dynamic range value로, 일반적인 RGB image에서는 255가 된다. 위의 값들을 다 대입하면 최종 SSIM 식은 아래와 같이 나온다. 

---

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

---


## LPIPS



## IS(Inception score)

IS는 generative model에서 자주 사용하는 evaluation metric으로, ImageNet에 pretrained된 model인 ***inception-v3***를 사용해서 GAN에 의해 생성된 image를 분류한다. Label $y$, image $x$에 대해 IS는 다음과 같이 정의된다. 

---

$$
\text{IS} = \text{exp}(\mathbb{E}_{x\sim p_{data}}D_{KL}(p(y|x)||p(y)))
$$

---

## FID(Frechet Inception Distance)

FID는 IS의 여러 한계를 극복하기 위해 제시된 evaluation metric이다. 이 역시 ImageNet에 pretrained된 ***inception-v3*** model을 사용하는데, 이때 마지막 layer을 제거하고 그 바로 전 layer의 activation을 사용한다. 이 activation의 mean과 covariance를 구해서 분포 사이의 distance를 이용해 구할 수 있다. 

---

$$
\text{FID} = d^2 = ||\mu_1 - \mu_2||^2 + \text{TR}(C_1+C_2-2C_1C_2)
$$

---



## CA

## PD