---
title: "Interactive Monte Carlo Denoising using Affinity of Neural Features"
layout: post # do not change
current: post # do not change
cover: ../assets/images/AMCD_cover.PNG
use_cover: false
navigation: true
date: '2022-02-15 16:27:00'
tags: paper-review
class: post-template # do not change
subclass: post # do not change
author: saha # do not change
use_math: true # do not change
---

# Introduction
- Producing high-quaility images from Monte Carlo low-sample renderings remains as a big challenge for interactive ray tracing. 
- Sate-of-art MC denoisers use a ***large kernel-predicting neural network***. 
    - Its computational cost is not a issue in ***off-line rendering***, since ray tracing dominates the overall latency. 
    - But in ***interactive applications***, it can be a problem. 
- In this paper, they propose a new denoiser for low-sample, interactive ray tracing applications that ***directly operates on the path-traced samples***. 
    - The model is a light-weight neural network that ***summarizes rich per-sample information into low-dimensional per-pixel feature vectors***. 
    - They define a novel pairwise affinity over these features, which is used to weight the contributions of neighboring per-pixel radiance values in a local weighted average filtering step. 
    - These affinity-based kernels works better then kernel-predicting methods, especially in reconstructing fine details or in high-frequency light transport configurations. 

