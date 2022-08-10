---
title: "A Survey on Deep Learning-Based Monte Carlo Denoising"
layout: post # do not change
current: post # do not change
cover: ../assets/images/Denoising_survey_cover.PNG
use_cover: false
navigation: true
date: '2022-02-23 14:23:00'
tags: paper-review
class: post-template # do not change
subclass: post # do not change
author: saha # do not change
use_math: true # do not change
---

# Introduction
- The synthesis of realistic images is one of the popular research topic in computer graphics. 
- MC integration methods are widely used, which have the **following 2 advantages**. 
    - Unified framework for rendering almost every physically-based rendering effect. 
    - Guarantee mathematical convergence to the ground truth. 
- MC integration methods, however, needs a large number of samples to achieve images of good quality, which requires immense computational power. 
    - When using a small number of samples, MC integration results suffer from estimator variance, which appears as visually distracting noise. 
- There are mainly **2 approaches** addressing this problem. 
    - **Pre-processing** : Devise more sophisticated sampling strategies to increase sampling efficiency
    - **Post-processing** : Develop a local reconstruction functions to trade mathematical convergence for visually appealing denoising(also known as **MC denoising**). 

# Deep Learning-Based MC Denoising
- Classic MC rendering estimates the color of the pixel through MC integration, the sum of the contributions from $M$ samples in the domain $\Omega$, which depends on the contribution $f(s_m)$ and the sampling probability $p(s_m)$ of the m-th sample, $s_m$. 

$$
c = \int _\Omega f(s)ds\approx \frac{1}{M} \sum_{m=1}^M \frac{f(s_m)}{p(s_m)}
$$

- This produces **high variance with low sample counts**, which leads to visible noise, and this motivates the development of MC denoising techniques to filter noisy inputs to achieve a plausible rendering quality with a reasonable time budget. 

- MC denoising is done mostly in a supervised manner, which uses $N$ example pairs $(x^1, r^1), ..., (x^N, r^N)$ and a loss function $l$ to estimate parameters $\hat{\theta}$. $X^n$ is a block of per-pixel vectors of $x^n$, which is used to reconstruct the output at $x^n$. 

$$
\hat{\theta} = min \frac{1}{N} \sum_{n=1}^N l(r^n, g(X^n;\theta))
$$

# Pixel Denoising
- This is the most basic MC denoising method, which reconstructs a single smooth image from the noisy input using some auxiliary features. 

### Parameter prediction
- Originally, MC denoising was aimed on making filters using additional scene features(shading normals, texture albedo, etc...)
- **Kalantari et al.** observed that there is a relationship between the noisy scene and filter parameters, and proposed to learn this relationship using deep learning. 
    - Based on MLP. 
    - First train the MLP on a set of noisy image of scenes with a variety of distributed effects to regress the optimal filter parameters. 
    - Then the trained network can predict the filter parameters for a new scene. 

### Kernel prediction
- Predicting parameters of a filter establishes local reconstruction of kernels for pixels in an indirect way. 