# Noisy Differentiable Architecture Search
Xiangxiang Chu, Bo Zhang, Xudong Li  
## Abstract
DARTS 的可复现性不高，遭到了不少研究者和从业人员的质疑，主要集中在，[1] 训练过程中存在 skip-connection 富集现象，导致最终模型出现大幅度的性能损失问题 [2] softmax 离散化存在很大 gap，结构参数最佳的操作和其他算子之间的区分度并不明显，这样选择的操作很难达到最优。  
在之前的工作 FairDARTS ，通过使用 sigmoid 函数而不是 softmax 函数来解决富集和性能损失问题。我们认为，softmax 使不同操作之间的关系变为竞争关系，由于 skip connection 和其他算子的加和操作形成残差结构，这就导致了 skip connection 比其他算子有很大的优势，这种优势在竞争环境下表现为不公平优势并持续放大，而其他有潜力的操作受到排挤，因此任意两个节点之间通常最终会以 skip connection 占据主导，导致最终搜索出的网络性能严重不足。
而 FairDARTS 通过 sigmoid 使每种操作有自己的权重，这样鼓励不同的操作之间相互合作，最终选择算子的时候选择大于某个阈值的一个或多个算子，在这种情形下，所有算子的结构权重都能够如实体现其对超网性能的贡献，而且残差结构也得以保留，因此最终生成的网络不会出现性能崩塌，从而避免了原生 DARTS 的 skip-connection 富集而导致的性能损失问题。  
新作 NoisyDARTS 是在 FairDARTS 基础上的推论，既然 skip connection 存在不公平优势，那么对其注入噪声即可干扰其优势，抑制其过度发挥，从而解决 skip connection 富集现象。
这是一个简单优雅但又极为有效的方法。NoiseDARTS 从数学推导上回答了，实际操作的时候应该注入怎样的噪声，以及注入噪声对网络有何影响，实验部分也符合推理。
## Introduction  
富集的skip conenection导致的性能下滑是DARTS主要的问题，我们认为这是因为不同的op之间存在不公平的排他竞争。根据经验性的观察，我们采取不同且直接的方法，
向skip connection的输出注入噪声，噪声会通过skip connection对梯度流带来扰动，从而减少竞争的不正当性。  
## Method  
### 1. 动机  
如前所述
### 2. 注入噪声的需求
令<img src="https://latex.codecogs.com/gif.latex?\tilde&space;x" title="\tilde x" />为注入skip connection的噪声，α<sup>skip</sup>是对应结构的权重
skip操作的loss可以写作  
<img src="https://latex.codecogs.com/gif.latex?L=g(y)&space;\quad&space;y=f(\alpha^{skip})\cdot&space;(x&plus;\tilde&space;x)" title="L=g(y) \quad y=f(\alpha^{skip})\cdot (x+\tilde x)" />  
g(y)是验证集损失，f(α<sup>skip</sup>)是对α的softmax，近似地，当噪声比输出特征小得多的时候，我们有  
<img src="https://latex.codecogs.com/gif.latex?y^*\approx&space;f(\alpha)\cdot&space;x&space;\qquad&space;when&space;\quad&space;\tilde&space;x&space;\ll&space;x" title="y^*\approx f(\alpha)\cdot x \qquad when \quad \tilde x \ll x" />  
在噪声场景下，skip op结构参数的梯度为  
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L&space;}{\partial&space;\alpha^{skip}}=\frac{\partial&space;L}{\partial&space;y}\frac{\partial&space;y}{\partial&space;\alpha^{skip}}&space;=\frac{\partial&space;L}{\partial&space;y}\frac{\partial&space;f(\alpha^{skip})}{\partial&space;\alpha^{skip}}(x&plus;\tilde&space;x)" title="\frac{\partial L }{\partial \alpha^{skip}}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial \alpha^{skip}} =\frac{\partial L}{\partial y}\frac{\partial f(\alpha^{skip})}{\partial \alpha^{skip}}(x+\tilde x)" />  
因为随机噪声<img src="https://latex.codecogs.com/gif.latex?\tilde&space;x" title="\tilde x" />为梯度更新带来了不确定性，skip connection需要
客服这个困难来与其他op竞争
