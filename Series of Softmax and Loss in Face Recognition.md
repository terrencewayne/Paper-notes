# Series of Softmax and Loss in Face Recognition
## 1. Softmax Loss
softmax归一化到[0, 1]区间，再做交叉熵。  

### softmax  
<img src="https://latex.codecogs.com/gif.latex?f\left&space;(&space;v_k&space;\right&space;)=\frac{e^{v_k}}{\sum&space;_je^{v_j}}" title="f\left ( v_k \right )=\frac{e^{v_k}}{\sum _je^{v_j}}" />  

### Cross Entropy Loss  
由KL散度（相对熵）出发  
<img src="https://latex.codecogs.com/gif.latex?L_{KL}(p||q)=\sum_{i=1}^{N}p\left&space;(&space;x_i&space;\right&space;)log\left&space;(&space;\frac{p\left&space;(&space;x_i&space;\right&space;)}{q\left&space;(&space;x_i&space;\right&space;)}&space;\right&space;)=\sum_{i=1}^{N}p\left&space;(&space;x_i&space;\right&space;)\left&space;(&space;log(p(x_i))-log(q(x_i))&space;\right&space;)" title="L_{KL}(p||q)=\sum_{i=1}^{N}p\left ( x_i \right )log\left ( \frac{p\left ( x_i \right )}{q\left ( x_i \right )} \right )=\sum_{i=1}^{N}p\left ( x_i \right )\left ( log(p(x_i))-log(q(x_i)) \right )" />  
*q*是网络输出的分布，*p*是ground truth分布（one-hot），因此进一步写作  

