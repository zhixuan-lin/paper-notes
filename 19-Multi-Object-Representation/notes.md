* Multi-Object Representation Learning with Iterative Variational Inference
* Klaus Greff, Alexander Lerchner

# Abstract

This feature learning frame work learns to segment and represent objects _jointly_.

# 1 Introduction

* What is variational inference?
* What is amortized variational inference?
* Neural expectation maximization: **spatial mixture model**
* Iterative amortized inference
* Stochastic backpropagation and approximate inference in deep generative models: VAE

Previous work on feature learning focus on learning feature from **pre-segmented** objects. However, this work also values **discovery of objects**.

# 2 Method

## 2.1 Multi-Object Representations

* Spatial broadcast decoder:

Instead of using a single flat vector as standard VAEs, we use **multi-slot** representations.

**Generative Model**. We assume that, the input images are generated as follows: first, we determine $K$ latent representations $z_k \in \R^M$. Then, the image $x$ is extracted from a spatial mixture of Gaussian parameterized by $z_k$. Each pixel is independently sampled:
$$
p(x|z) =  \prod_{i =1}^D \sum_{k = 1}^Km_{ik}\cal N(x_i |\mu_{ik}, \sigma^2_x)
$$
Under this assumption, our goal is to 

* Find the parameter for $p(z |x)$
* Find the parameter for $p(x|z)$. I.e., the mapping from $z$ to parameters of the mixture.

**Decoder Structure**. The problem is to decode $z_k$ to $m_k$ and $\mu_k$. We use **spatial broadcast network**. All slot share weights to ensure a common format.

## 2.2 Inference

