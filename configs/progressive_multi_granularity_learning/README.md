# PMG

[PMG](https://arxiv.org/abs/2003.03836v3)

## Abstract
In this work, we propose a novel framework for fine-grained visual classification to tackle these problems. In particular, we propose: (i) a progressive training strategy that effectively fuses features from different granularities, and (ii) a random jigsaw patch generator that encourages the network to learn features at specific granularities. We obtain state-of-the-art performances on several standard FGVC benchmark datasets, where the proposed method consistently outperforms existing methods or delivers competitive results.

<div align=center>
<img src="https://github.com/YangYuqi317/PRIS-CV_FGVCLib/blob/main/docs/en/configs/framework/PMG_1.png?raw=true"/>
</div>

Illustration of features learned by general methods (a and b) and our proposed method (c and d). 
(a) Traditional convolution neural networks trained with cross entropy (CE) loss tend to find the most discriminative parts. 
(b) Other state-of-the-art methods focus on how to find more discriminative parts. 
(c) Our proposed progressive training (Here we use last three stages for explanation.) gradually locates discriminative information from low stages to deep stage. And features extracted from all trained stages are concatenated together to ensure complementary relationships are fully explored, which is represented by “Stage Concat.” 
(d) With assistance of jigsaw puzzle generator the granularity of parts learned at each step are restricted inside patches.

## Introduction
Progressive training methodology was originally proposed for generative adversarial networks, where it started with low-resolution images, and then progressively increased the resolution by adding layers to the networks. Instead of learning the information from all the scales, this strategy allows the network to discover large-scale structure of the image distribution and then shift attention to increasingly ner scale details. Recently, progressive training strategy has been widely utilized for generation tasks, since it can simplify the information propagation within the network by intermediate supervision. For FGVC, the fusion of multi-granularity information is critical to the model performance. In this work, we adopt the idea of progressive training to design a single network that can learn these information with a series of training stages. The input images are firstly split into small patches to train a low-level layers of model. Then the number of patches are progressively increased and the corresponding layers high-level lays have been added and trained, correspondingly. Most of the existing work with progressive training are focusing on the task of sample generation. To the best of our knowledge, it has not been attempted earlier for the task of FGVC.

<div align=center>
<img src="https://github.com/YangYuqi317/PRIS-CV_FGVCLib/blob/main/docs/en/configs/framework/PMG_2.png?raw=true"/>
</div>

The training procedure of the progressive training which consists of S + 1 steps at each iteration (Here S = 3 for explanation). The Conv Block represents the combination of two convolution layers with and max pooling layer, and Classif ier represent two fully connected layers with a softmax layer at the end. At each iteration, the training data are augmented by the jigsaw generator and sequentially input into the network by S + 1 steps. In our training process, the hyper-parameter n is 2L−l+1 for the lth stage. At each step, the output from the corresponding classifier will be used for loss computation and parameter updating.
