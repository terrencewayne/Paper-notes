# 机器学习-神经网路基础知识

## 感知机  

### 感知机的学习规则  
对训练样例（x, y）感知机输出为y-hat，则权重这样调整  

<img src="https://latex.codecogs.com/gif.latex?w_i\leftarrow&space;w_i&plus;\bigtriangleup&space;w_i" title="w_i\leftarrow w_i+\bigtriangleup w_i" />  
<img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;w_i=\eta&space;(y-\hat{y})x_i" title="\bigtriangleup w_i=\eta (y-\hat{y})x_i" />  

### BP算法  
机器学习P101

### 缓解过拟合
机器学习P105  
早停：将数据分成训练集和验证集，训练集用于计算梯度、更新连接权和阈值，验证集用来估计训练误差，若训练集误差降低但验证集误差升高，则停止训练，同时返回具有
最小验证集误差的连接权和阈值。  
正则化：在误差目标函数增加一个用于描述网络复杂度的部分，例如连接权与阈值的平方和。  
<img src="https://latex.codecogs.com/gif.latex?E=\lambda\frac{1}{m}\sum_{k=1}^mE_k&plus;(1-\lambda)\sum_iw_i^2" title="E=\lambda\frac{1}{m}\sum_{k=1}^mE_k+(1-\lambda)\sum_iw_i^2" />  

### 跳出局部最小的方案
机器学习P107
1.以多组不同参数初始化多个神经网络，取误差最小的解作为最终参数。  
2.模拟退火，每一步都以一定概率接受比当前解更差的结果，迭代过程中，接受次优解的概率随着时间推移而降低。  
3.随机梯度下降，计算梯度时加入随机因素。
