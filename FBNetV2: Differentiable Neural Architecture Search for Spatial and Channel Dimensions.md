# FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions
Alvin Wan, Xiaoliang Dai, Peizhao Zhang, Zijian He, Yuandong Tian, Saining Xie, Bichen Wu,
Matthew Yu, Tao Xu, Kan Chen, Peter Vajda, Joseph E. Gonzalez
## Introduction
传统的DARTS-based方法要求候选层可以显式实例化，导致搜索空间很窄，提出了DMaskingNas，扩展了搜索空间，支持对输入分辨率和通道数的搜索。  
## Method
1. 通道搜索
为了所搜可变的通道，以往的方法在super-graph中为每一个可选的通道进行实例化，对于k个滤波器的卷积，这加起来包括k(k+1)/2次卷积。在不考虑庞大的可能性的情况下，
也有两个很明显的问题：
* 维度不匹配。DNAS被分为几个cell，每个cell内部，block都必须对齐通道
* 训练太慢，资源消耗大。不同通道的选项需要分开训练，在内存中要分别保存。
方法如图
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/fbnetv2_0.png "channel search")  
考虑block b, b<sub>i</sub>表征b有i个滤波器。允许的最大滤波器数量为k。Step B中，所有block的输出zero-padding到k维。给定输入x，Gumbel Softmax 的输出
按如下描述，g<sub>i</sub>表示Gumbel权重  
<img src="https://latex.codecogs.com/gif.latex?y=\sum_{i=1}^kg_iPAD(b_i(x),k)" title="y=\sum_{i=1}^kg_iPAD(b_i(x),k)" />  
注意到这相当于把所有卷积的滤波器提高到k，然后把输出mask处理（Step C）。L<sub>i</sub>∈R<sup>k</sup>是一个列向量，前端是1，后端是k-i个0，注意到搜索算法
无所谓1和0的顺序，因为所有block的滤波器数目相同。我们可以令其共享参数（Step D）  
<img src="https://latex.codecogs.com/gif.latex?y=\sum_{i=1}^kg_i(b(x)\circ&space;L_i)" title="y=\sum_{i=1}^kg_i(b(x)\circ L_i)" />  
最终，在这种近似下，变成以gumbel softmax聚合mask，block b只需推理一次（Step E）  
<img src="https://latex.codecogs.com/gif.latex?y=b(x)\circ&space;\sum_{i=1}^kg_iL_i" title="y=b(x)\circ \sum_{i=1}^kg_iL_i" />  

2. 输入分辨率搜索  
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/fbnetv2_1.png "input resolution search")  
与通道搜索相同，我们直接为结果填充0，但有两个坑  
* 像素不对齐。在边缘填充导致像素不对齐，如B所示。使用空间散布填充C解决这个问题  
* 感受野不对齐。如图D，在zero-pad后的fp上做3X3卷积，感受野实际上缩小了。因此采取E的方式。  

3. 有效的形状传播  
定义gumbel softmax weights  
<img src="https://latex.codecogs.com/gif.latex?g_i^l=\frac{exp[(\alpha_i^l&plus;\epsilon&space;_i^l)/\tau]}{\sum_iexp[(\alpha_i^l&plus;\epsilon&space;_i^l)/\tau]}" title="g_i^l=\frac{exp[(\alpha_i^l+\epsilon _i^l)/\tau]}{\sum_iexp[(\alpha_i^l+\epsilon _i^l)/\tau]}" />  
对于第l层的卷积，这样定义其有效输出形状（有效宽、高、通道）  
<img src="https://latex.codecogs.com/gif.latex?\bar{C}_{out}^l=\sum_ig_i^l\cdot&space;C_{i,out}^l\\&space;\bar{h}_{out}^l=\sum_ig_i^l\cdot&space;\bar{h}_{i,in}^l&space;\quad&space;\bar{w}_{out}^l=\sum_ig_i^l\cdot&space;\bar{w}_{i,in}^l\\&space;\bar{S}_{out}^l=(n,\bar{C}_{out}^l,\bar{h}_{out}^l,\bar{w}_{out}^l)" title="\bar{C}_{out}^l=\sum_ig_i^l\cdot C_{i,out}^l\\ \bar{h}_{out}^l=\sum_ig_i^l\cdot \bar{h}_{i,in}^l \quad \bar{w}_{out}^l=\sum_ig_i^l\cdot \bar{w}_{i,in}^l\\ \bar{S}_{out}^l=(n,\bar{C}_{out}^l,\bar{h}_{out}^l,\bar{w}_{out}^l)" />  
给定实际输出C<sub>out</sub>, 有效输入C<sub>in</sub><sup>'</sup>，第l层的cost为  
<img src="https://latex.codecogs.com/gif.latex?cost^l=\left\{\begin{matrix}&space;k^2\cdot\bar{h}_{out}^l\cdot\bar{w}_{out}^l\cdot\bar{C}_{in}^l\cdot\bar{C}_{out}^l/\gamma&space;\qquad&space;if&space;\quad&space;FLOP\\&space;k^2\cdot\bar{C}_{in}^l\cdot\bar{C}_{out}^l/\gamma&space;\qquad&space;if&space;\quad&space;param&space;\end{matrix}\right." title="cost^l=\left\{\begin{matrix} k^2\cdot\bar{h}_{out}^l\cdot\bar{w}_{out}^l\cdot\bar{C}_{in}^l\cdot\bar{C}_{out}^l/\gamma \qquad if \quad FLOP\\ k^2\cdot\bar{C}_{in}^l\cdot\bar{C}_{out}^l/\gamma \qquad if \quad param \end{matrix}\right." />  
γ表示卷积分组数，训练loss包括交叉熵+总cost。



