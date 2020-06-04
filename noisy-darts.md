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
因为随机噪声<img src="https://latex.codecogs.com/gif.latex?\tilde&space;x" title="\tilde x" />为梯度更新带来了不确定性，skip connection需要克服这个困难来与其他op竞争  
然而，并不是所有的噪声都有有效，一个基本的准则是，在压制不公平竞争的同时，不应该为梯度的期望带来偏置。梯度的期望可以写作  
<img src="https://latex.codecogs.com/gif.latex?\mathbb{E}[\triangledown&space;_{skip}]=\mathbb{E}[\frac{\partial&space;\mathcal{L}}{\partial&space;y}\frac{\partial&space;f(\alpha^{skip})}{\partial&space;\alpha^{skip}}(x&plus;\tilde&space;x)]\approx&space;\frac{\partial&space;\mathcal{L}}{\partial&space;y^*}\frac{\partial&space;f(\alpha^{skip})}{\partial&space;\alpha^{skip}}(\mathbb{E}[x]&space;&plus;\mathbb{E}[\tilde&space;x])" title="\mathbb{E}[\triangledown _{skip}]=\mathbb{E}[\frac{\partial \mathcal{L}}{\partial y}\frac{\partial f(\alpha^{skip})}{\partial \alpha^{skip}}(x+\tilde x)]\approx \frac{\partial \mathcal{L}}{\partial y^*}\frac{\partial f(\alpha^{skip})}{\partial \alpha^{skip}}(\mathbb{E}[x] +\mathbb{E}[\tilde x])" />  
基于上面<img src="https://latex.codecogs.com/gif.latex?\tilde&space;x&space;\ll&space;x" title="\tilde x \ll x" />的假定，把与x无关的项拿出期望算符。在skip connection中仍然有一项<img src="https://latex.codecogs.com/gif.latex?\mathbb{E}[\tilde&space;x]" title="\mathbb{E}[\tilde x]" />，为了保证梯度无偏，这一项必须等于0，所以我们要使用小且无偏的噪声，简单起见，这里只使用高斯噪声。  
### 3. 使用噪声解决性能下降  
我们向skip connection注入高斯噪声<img src="https://latex.codecogs.com/gif.latex?\tilde&space;x\sim&space;\mathcal{N}(\mu&space;,\sigma)" title="\tilde x\sim \mathcal{N}(\mu ,\sigma)" />  
令边e<sub>i,j</sub> 从节点i到j，对应输入为x<sub>i</sub>。其输出记为<img src="https://latex.codecogs.com/gif.latex?o_{i,j}(x_i)" title="o_{i,j}(x_i)" />，中间节点j收集来自所有入边的输入  
<img src="https://latex.codecogs.com/gif.latex?x_j=\sum_{i<j}o_{i,j}(x_i)" title="x_j=\sum_{i<j}o_{i,j}(x_i)" />  
令<img src="https://latex.codecogs.com/gif.latex?\mathcal{O}=\left&space;\{&space;o_{i,j}^0,o_{i,j}^1,\cdot&space;\cdot&space;\cdot,&space;o_{i,j}^{M-1}&space;\right&space;\}" title="\mathcal{O}=\left \{ o_{i,j}^0,o_{i,j}^1,\cdot \cdot \cdot, o_{i,j}^{M-1} \right \}" />为M个候选op，特别地，令第一个元素为skip connection。注入加性噪声来得到混合输出  
<img src="https://latex.codecogs.com/gif.latex?\bar&space;o_{i,j}(x)=\sum_{k=1}^{M-1}f(\alpha_{o^k})o^k(x)&plus;f(\alpha_{o^{skip}})o^{skip}(x&plus;\tilde&space;x)" title="\bar o_{i,j}(x)=\sum_{k=1}^{M-1}f(\alpha_{o^k})o^k(x)+f(\alpha_{o^{skip}})o^{skip}(x+\tilde x)" />  
为了保证skip op梯度无偏且噪声足够小，我们设置<img src="https://latex.codecogs.com/gif.latex?\mu=0&space;\quad&space;\sigma=\lambda\cdot&space;std(x)" title="\mu=0 \quad \sigma=\lambda\cdot std(x)" /> 其中λ是正系数。也就是说，噪声的标准差根据样本batch而变化，设置低的λ来达到噪声足够小。  
## Experiment
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/noisy-darts.png "noisy-darts visual")  
如图所示，随着训练，skip op的权重（深绿色）显著下降。达到作者的预期


