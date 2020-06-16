# Circle Loss: A Unified Perspective of Pair Similarity Optimization
Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, Yichen Wei
## Introduction
本文从相似度优化的角度来审视两类基本的深度特征学习模式，即学习类别级的标签，和学习成对的标签。前者使用分类损失函数（softmax交叉熵）样本间的相似度和权重。后者
使用度量损失函数（triplet loss）来优化样本间的相似度。这两者没有本质的区别。他们都希望最小化类间相似度s<sub>n</sub>，最大化类内相似度s<sub>p</sub>。
从这个角度来说，许多流行的loss有着相同的优化模式，把s<sub>n</sub>和s<sub>p</sub>嵌入成相似度对，降低(s<sub>n</sub> - s<sub>p</sub>)，我们认为这样的
对称优化有两个问题：  

**缺少优化的稳定性。** s<sub>n</sub>和s<sub>p</sub>的的惩罚强度被约束成一样的，给定loss，关于s<sub>n</sub>和s<sub>p</sub>的梯度的幅度是一样的。
在一些边缘case上，比如s<sub>p</sub>很小，s<sub>n</sub>已经接近于0，仍然使用大的梯度来惩罚s<sub>n</sub>，低效且不合理  

**不明确的收敛状态。** 优化(s<sub>n</sub> - s<sub>p</sub>)经常导致决策边界为s<sub>n</sub> - s<sub>p</sub> = m，这个决策边界将允许收敛的不明确性
如图中的T所示。例如，T: {s<sub>n</sub>, s<sub>p</sub>} = {0.2, 0.5}; T': {s<sub>n</sub>', s<sub>p</sub>'} = {0.4, 0.7}，他们的边际都是0.3
但是，s<sub>n</sub>'和s<sub>p</sub>之间的gap只有0.1。不明确的收敛状态模糊了特征空间的可区分性。  
![img](https://github.com/terrencewayne/Paper-notes/blob/master/images/circleloss0.png "0")  

因此我们认为，不同的相似度得分应该有不同的惩罚强度。如果相似度得分脱离最优点很远，那就应该施以强的惩罚。我们首先把(s<sub>n</sub> - s<sub>p</sub>)泛化
成(α<sub>n</sub>s<sub>n</sub> - α<sub>p</sub>s<sub>p</sub>)，允许s<sub>n</sub>和s<sub>p</sub>以不同的步长学习。然后我们又把α<sub>n</sub>和
α<sub>p</sub>实现为s<sub>n</sub>和s<sub>p</sub>的线性函数，让步长根据状态自适应调整。这样就产生了α<sub>n</sub>s<sub>n</sub> - α<sub>p</sub>s<sub>p</sub> = m
的决策边界，在s<sub>n</sub>和s<sub>p</sub>空间是一个圆，因此称为circle loss  

## 统一的视角  
给定特征空间的一个样本x，假设有K个类内相似度得分和L个类间相似度得分。记这些得分为<img src="https://latex.codecogs.com/gif.latex?{s_p^i}(i=1,2,...,K)&space;\quad&space;{s_n^j}(j=1,2,...,L)" title="{s_p^j}(i=1,2,...,K) \quad {s_n^j}(j=1,2,...,L)" />  
最小化<img src="https://latex.codecogs.com/gif.latex?s_n^j" title="s_n^j" />和最大化<img src="https://latex.codecogs.com/gif.latex?s_p^i" title="s_p^i" />是一致的。我们提出一个统一的损失函数  
<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{uni}=log[1&plus;\sum_{i=1}^K\sum_{j=1}^Lexp(\gamma(s_n^j-s_p^i&plus;m))]\\&space;=log[1&plus;\sum_{j=1}^Lexp(\gamma(s_n^j&plus;m))\sum_{i=1}^Kexp(\gamma(-s_p^i))]" title="\mathcal{L}_{uni}=log[1+\sum_{i=1}^K\sum_{j=1}^Lexp(\gamma(s_n^j-s_p^i+m))]\\ =log[1+\sum_{j=1}^Lexp(\gamma(s_n^j+m))\sum_{i=1}^Kexp(\gamma(-s_p^i))]" />  
γ是scale参数，m是边际  

我们注意到，上式在轻微修改的情况下会退化为triplet或交叉熵  

给定class-level标签，计算N-1个类间相似度，

