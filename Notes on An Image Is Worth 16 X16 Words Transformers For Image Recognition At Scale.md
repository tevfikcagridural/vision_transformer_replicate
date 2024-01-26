## Abstract
The approach is using pure transformer architecture on images and it's less computationally resourceful.

## Introduction
In other earlier works CNN-like architectures with self-attention already tried. But this approach **splits images into patches and pretend these patches as words/tokens in vanilla transformer architecture**. Also earlier a similar approach was tried by other teams but wasn't very successful as they tried mid-size datasets. In this work larger (14M-300M images) were used in pre-training and the results are significantly better.

## Related work
This paper is very similar to (Cordonnier et. al. 2020) the difference is this paper uses large scale pre-training. Also pixel size for patches differ.

![[model_structure.png]]
## Method
1. Convert image $x \in \mathbb{R}^{H \times W \times C}$  into patches  $x_{p} \in \mathbb{R}^{N \times (P^{2} \cdot C)}$
	- H, W: resolution
	- C: num of channels
	- P: patch resolution
	- N = HW / P^2: num of patches
2. Append *learnable* class embedding to the patches as extra dim
3. Add *learnable* position embedding to the patch and class embeddings. Changing the information within the vector
4. Feed the vector to the transformer
5. Make classifications with an Multilayer Perceptron layer.

The whole structure is made of four equations

1. $z_{0}= [x_{class}; x^1_{p}E; x^2_{p}E;...x^N_{p}E]+E_{pos}$ $E \in \mathbb{R}^{(P^{2}\cdot C)\times D}, E_{pos} \in \mathbb{R}^{(N+1)\times D}$
2. $z'_{\ell}=MSA(LN(z_{\ell-1}))+z_{\ell-1}$   $\ell=1...L$
3. $z_{\ell}=MLP(LN(z'_{\ell}))+z'_{\ell}$    $\ell= 1...L$
4. $y=LN(z^0_L)$

## Experiments
In the fine tuning parts **SGD with moment** used for fine tuning

ViT uses 2-4x less compute to attain the same performance of ResNet

## Inspecting ViT
Model learns to encode distance within the image in the similarity of position embeddings.
![[position embeddings.png]]

Also learns to give attentions on meaningful structures in the image
![[sample_attentions.png]]