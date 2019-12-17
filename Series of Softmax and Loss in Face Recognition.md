# Series of Softmax and Loss in Face Recognition
## 1. Softmax Loss
softmax归一化到[0, 1]区间，再做交叉熵。  
**softmax**  
<img src="https://latex.codecogs.com/gif.latex?f\left&space;(&space;v_k&space;\right&space;)=\frac{e^{v_k}}{\sum&space;_je^{v_j}}" title="f\left ( v_k \right )=\frac{e^{v_k}}{\sum _je^{v_j}}" />  
**Cross Entropy Loss**  
