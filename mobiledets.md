# MobileDets: Searching for Object Detection Architectures for Mobile Accelerators
Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel
Bender, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, and Bo Chen
## I Introduction  
**以往工作的缺陷**  
1. 性能好的模型计算量大，不容易部署  
2. 手工设计轻量模型时间代价太大，且可能是次优的；尤其是新的计算平台层出不穷  
3. IBN（Inverted Residual Block）作为building block进行NAS很流行。尽管dw卷积在移动端cpu表现良好，但在一些移动端加速处理器上没有很好的优化。比如
Edge TPU和Qualcomm DSP，这些处理器被设计为加速常规卷积。在特定张量形状和卷积核大小下，常规卷积甚至比dw卷积快出很多，尽管计算量更大。  

**我们的工作** 
1. 我们提出一个扩大的搜索空间，灵感来源于张量分解的结构，该空间命名为Tensor-Decomposition-Based（TDB）空间，可以在很多移动端加速处理器上工作。该搜索空间
以既能扩张又能压缩的building block为基础，一旦在网络中放置在合适的位置，可以运行的很高效。  
2. 为了高效地排列这些block，我们在目标检测任务上进行时延感知的架构搜索，面向CPU/Edge TPU/DSP等一系列平台。我们首先证明专注于检测的搜索是有效的，进一步
证实TDB空间的有效性，通过在选定位置设置完整卷积，我们的方法比IBN-only的网络性能更好。  
## II 探究移动端搜索空间中的卷积  
基于IBN的上述缺陷，我们考虑使用常规卷积来丰富搜索空间。具体地，我们提出两种灵活的结构分别来实现通道的扩张和压缩。  

**Fused Inverted Bottleneck Layers (Expansion)**  
IBN使用空间上的dw卷积核维度上的1x1卷积来替代expensive的常规卷积。然而，expensive的概念来自于FLOPs或者参数量，在如今的设备上并不与推理效率直接相关。
为了纳入常规卷积，我们把IBN的第一个1x1卷积（通常伴随扩张而出现）和后续的KxK dw卷积组合成一个KxK的常规卷积。这个常规卷积仍然可以扩张通道。  

**Generalized Bottleneck Layers (Compression)**  
bottleneck由resnet引入，用来减少高维特征图的计算量。特征图先投影到低通道，再投影回来，这两步操作都是1x1卷积。在每一层的通道维度上进行仔细控制是很有用的。
为了实现灵活的压缩层，我们让传统的bottleneck有一个输入测压缩率和输出的压缩率，这两个参数由NAS来搜索。称这种building block为tucker convolution layer  

**Connections with Tucker/CP decomposition**  
这块看不懂，和后文也没关系，提供了另一种看待上述block的角度罢了。故略。  

## III Architecture Search Method  

**搜索算法**  
我们提出的搜索空间使用与任何架构搜索算法。我们主要使用TuNAS  
TuNAS建立一个包含所有选择的one-shot model，和一个挑选最能够优化平台感知目标函数的架构的控制器。这二者在搜索中联合训练。在每一步，控制器从多项分布中
采样出一个随机结构，然后one-shot模型中这部分的参数得到更新，最终为采样结构计算一个反馈，这个反馈用于更新控制器，更新方法使用标准强化学习算法。  

**Cost Models**  
理想情况下，应该在目标平台测试每个候选结构，但服务器训练-目标平台测试的闭环很难实现。因此我们训练了一个代价模型——一个线性回归模型，接收网络的结构信息，
预测其代价。这个线性代价模型的效果优于加性模型。  
搜索过程中，我们使用代价模型作为目标平台的替代。为了收集该模型的训练数据，我们从搜索空间随机的采样几千个结构，并在目标平台测试其时延。


