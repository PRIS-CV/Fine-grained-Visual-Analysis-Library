# MCL

[MCL](https://arxiv.org/abs/2002.04264)

## Abstract

In this paper, we show that it is possible to cultivate subtle details without the need for overly complicated network designs or training mechanisms – a single loss is all it takes. The main trick lies with how we delve into individual feature channels early on, as opposed to the convention of starting from a consolidated feature map. The proposed loss function, termed as mutual-channel loss (MC-Loss), consists of two channel-specific components: a discriminality component and a diversity component. The discriminality component forces all feature channels belonging to the same class to be discriminative, through a novel channel-wise attention mechanism. The diversity component additionally constraints channels so that they become mutually exclusive across the spatial dimension. The end result is therefore a set of feature channels, each of which reflects different locally discriminative regions for a specific class. The MC-Loss can be trained end-to-end, without the need for any bounding-box/part annotations, and yields highly discriminative regions during inference.

<div align=center>
<img src="https://github.com/YangYuqi317/PRIS-CV_FGVCLib/blob/main/docs/en/configs/framework/MCL_1.png?raw=true"/>
</div>

The framework of a typical fine-grained classification network where MC-Loss is used. The MC-Loss function considers the output feature channels of the last convolutional layer as the input and gathers together with the cross-entropy (CE) loss function using a hyper-parameter µ.

## Introduction
we propose the mutual-channel loss (MC-Loss) function to effectively navigate the model focusing on different discriminative regions without any fine-grained bounding-box/part annotations.
<div align=center>
<img src="https://github.com/YangYuqi317/PRIS-CV_FGVCLib/blob/main/docs/en/configs/framework/MCL_2.png?raw=true"/>
</div>
(a) Overview of the MC-Loss. The MC-Loss consists of (i) a discriminality component (left) that makes F to be class-aligned and discriminative, and (ii) a diversity component (right) that supervises the feature channels to focus on different local regions. (b) Comparison of feature maps before (left) and after (right) applying MC-Loss, where feature channels become class aligned, and each attending to different discriminate parts. 

**CWA**

While in case of traditional CNNs, trained with the classical CE loss objective, a certain subset of feature channels contain discriminative information, we here propose channelwise attention operation to enforce the network to equally capture discriminative information in all ξ channels corresponding to a particular class. Unlike other channel-wise-attention design that intends to assign higher priority to the discriminative channels using soft-attention values, we assign random binary weights to the channels and stochastically select a few feature channels from every feature group Fi during each iteration, thus explicitly encouraging every feature channel to contain sufficient discriminative information. This process could be visualized as a random channel-dropping operation. Please note that the CWA is used only during training and that the whole MC-Loss branch is not present at the time of inference. Therefore, the classification layer receives the same input feature distributions during both training and inference.

**CCMP**

Cross-channel max pooling is used to compute the maximum response of each element across each feature channel in Fi corresponding to a particular class, and thus it results into a one dimensional vector of size WH concurring to a particular class. Note that the cross-channel average pooling (CCAP) is an alternative of the CCMP, which only substitutes the max pooling operation by the average pooling. However, the CCAP tends to average each element across the Ngroup which may suppress the peaks of feature channels, i.e., attentions of local regions. On the contrary, the CCMP can preserve these attentions, and is found to be beneficial for fine-grained classification.

**GAP**

Global average pooling is used to compute the average response of each feature channel, resulting in a c-dimensional vector where each element corresponds to one individual class.