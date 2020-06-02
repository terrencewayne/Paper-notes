# DARTS: Differentiable Architecture Search
Hanxiao Liu, Karen Simonyan, Yiming Yang
## Introduction 
提出高效的NAS方法，darts。不再从离散的候选结构中搜索，我们松弛搜索空间以使其连续，因此可以根据其在验证集上的表现使用梯度进行优化。以往工作着重于搜索
结构的特定方面，例如滤波器形状或者分支模式，darts能够在大的搜索空间中学习有复杂拓扑结构的building block。另外，darts也不局限于特定的结构族，可应用于
CNN和RNN  
## Method
一个结构的计算过程可以看做是有向无环图，我们会介绍一种简单的对搜索空间的连续松弛机制，使其可以通过可导的目标函数联合优化器结构和权重。最后，我们提出一个
近似技术来使整个算法可行且高效。  
### 1. 搜索空间  
我们为最终的结构搜索一个compution cell来作为building block。  
一个cell是一个包含n个节点的有向无环图。每个节点x<sup>i</sup>是一个潜在表示（例如CNN中的一个feature map就是一个节点），每个有向的边(i, j) 代表对x<sup>i</sup>
的操作o<sup>i,j</sup>, 我们假设cell有两个输入节点和一个输出节点，对卷积cell来说，输入节点定义为前两层的输出。cell的输出通过使用约减操作来聚合所有中间
节点（例如concatenate）  
每个中间节点根据其祖先计算  
<img src="https://latex.codecogs.com/gif.latex?x^j=\sum_{i<j}o^{i,j}(x^i)" title="x^j=\sum_{i<j}o^{i,j}(x^i)" />  
特殊的0操作也被考虑来表示两个节点之间无连接。这样学习cell的过程被简化为学习操作的过程。  
### 2. 连续松弛与优化
令O为候选的操作集（例如卷积、max pooling、0）等等。为了使搜索空间连续，我们把**对特定操作的类别选择**放松为**对所有可能操作的softmax**  
<img src="https://latex.codecogs.com/gif.latex?\bar&space;o^{i,j}(x)=\sum_{o\in&space;O}\frac{exp(\alpha_o^{i,j})}{\sum_{o'\in&space;O}exp(\alpha_{o'}^{i,j})}o(x)" title="\bar o^{i,j}(x)=\sum_{o\in O}\frac{exp(\alpha_o^{i,j})}{\sum_{o'\in O}exp(\alpha_{o'}^{i,j})}o(x)" />  
带权重的操作通过α（维度为|O|）来量化。那么结构搜索的任务约减为学习一组连续变量α={α<sup>i,j</sup>}。结束搜索后，一个固定的结构通过将每条边替换为最大可能的
结构来实现。后文中用α来指代这个结构。  
记L<sub>t</sub>和L<sub>v</sub>为训练和验证loss，loss由结构α和权重w同时决定，结构搜索的目标是找到最小化L<sub>v</sub>的阿尔法，其权重由优化L<sub>t</sub>
得到。  
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/darts0.png "darts")  
### 3. 近似结构梯度  
因为高昂的优化代价，评估结构梯度可能会很难，因此我们提出了简单的近似算法：  
<img src="https://latex.codecogs.com/gif.latex?\triangledown&space;_\alpha&space;L_{val}(\omega^*(\alpha),\alpha)\approx&space;\triangledown&space;_\alpha&space;L_{val}(\omega&space;-&space;\xi&space;\triangledown&space;_\omega&space;L_{train}(\omega,\alpha),\alpha)" title="\triangledown _\alpha L_{val}(\omega^*(\alpha),\alpha)\approx \triangledown _\alpha L_{val}(\omega - \xi \triangledown _\omega L_{train}(\omega,\alpha),\alpha)" />  
ω表示算法保留的当前权重，ξ表示内优化的学习率。思路是通过对ω 调整训练一步来近似ω<sup>*</sup>(α)，而不是将ω训练至收敛。  
我们仍不清楚这样训练到收敛是否能保证算法的最优化，实践中，以恰当的学习率去训练能够达到收敛的阶段。  
对上式使用链式法则，得到  
<img src="https://latex.codecogs.com/gif.latex?\triangledown&space;_\alpha&space;L_{val}(\omega',\alpha)&space;-&space;\xi&space;\triangledown^2_{\alpha,\omega}L_{train}(\omega,&space;\alpha)\triangledown_{\omega'}L_{val}(\omega',\alpha)" title="\triangledown _\alpha L_{val}(\omega',\alpha) - \xi \triangledown^2_{\alpha,\omega}L_{train}(\omega, \alpha)\triangledown_{\omega'}L_{val}(\omega',\alpha)" />  
ω'表示进行一步前向训练的模型的权重，上式第二项包含复杂的矩阵向量乘积。通过几种近似方法，复杂度可以下降。  
令ε 为小的标量，ω<sup>±</sup> = ω ± ε▽<sub>ω'</sub>L<sub>val</sub>(ω', α) 那么  
<img src="https://latex.codecogs.com/gif.latex?\triangledown^2_{\alpha,\omega}L_{train}(\omega,&space;\alpha)\triangledown_{\omega'}L_{val}(\omega',\alpha)\approx&space;\frac{\triangledown_\alpha&space;L_{train}(\omega^&plus;&space;,\alpha)-\triangledown_\alpha&space;L{train}(\omega^-,&space;\alpha)&space;}{2\epsilon}" title="\triangledown^2_{\alpha,\omega}L_{train}(\omega, \alpha)\triangledown_{\omega'}L_{val}(\omega',\alpha)\approx \frac{\triangledown_\alpha L_{train}(\omega^+ ,\alpha)-\triangledown_\alpha L{train}(\omega^-, \alpha) }{2\epsilon}" />  
### 4. 推导离散结构  
为了得到离散结构中的每个节点，我们保留之前节点的非0操作中top-K操作，每个操作的强度定义为  
<img src="https://latex.codecogs.com/gif.latex?\frac{exp(\alpha_o^{(i,j)})}{\sum_{o'\in&space;O}exp(\alpha_{o'}^{(i,j)})}" title="\frac{exp(\alpha_o^{(i,j)})}{\sum_{o'\in O}exp(\alpha_{o'}^{(i,j)})}" />  
卷积cell中k=2  

### 注意
DARTS只搜索cell，也就是building block，但整体网络结构、通道数是固定好的。


