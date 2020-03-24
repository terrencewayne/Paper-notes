# Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
Xinyao Wang, Liefeng Bo, Li Fuxin
## Intro
本文提出了Adaptive Wingloss，一种适用于heatmap regression的损失函数。 
## 方法
heatmap regression的关键在于以每一个ground truth坐标为中心，输出高斯分布。因此，正确预测高斯模式下的像素值至关重要。一般方法存在两个问题：1）MSELoss对小错误不够敏感，2）训练过程中，所有像素点共享一样的公式和权重，然而，heatmap中背景像素要比前景像素多得多；因此，MSE使模型输出一个模糊膨胀的heatmap，具体地，与groound truth相比，模型输出的预测前景像素值要更低。  
WingLoss针对该问题而提出。Wingloss定义如下  
<img src="https://latex.codecogs.com/gif.latex?Wing(y,\hat{y})=\left\{\begin{matrix}&space;\omega&space;ln(1&plus;\left&space;|&space;\frac{y-\hat{y}}{\varepsilon&space;}&space;\right&space;|)&&space;if\left&space;|&space;y-\hat{y}&space;\right&space;|<&space;\omega\\&space;\left&space;|&space;y-\hat{y}&space;\right&space;|-C&otherwise&space;\end{matrix}\right." title="Wing(y,\hat{y})=\left\{\begin{matrix} \omega ln(1+\left | \frac{y-\hat{y}}{\varepsilon } \right |)& if\left | y-\hat{y} \right |< \omega\\ \left | y-\hat{y} \right |-C&otherwise \end{matrix}\right." />  
C的值是为了使w处函数连续。  
当y-y_hat = 0时，Wingloss的导数非常大，使得训练很困难。在hearmap regression中，使用Wingloss计算背景像素loss，这些像素上的loss将站非常大的比重，训练一个小误差的网络将会非常困难。  
我们希望设计这样一个损失函数：在误差比较大时梯度恒定，在误差比较小时，1）对前景像素，梯度值应该增加，使得这些点被模型关注；在误差接近0时，梯度应该迅速减小，使得足够好的点失去模型的关注。2）对背景图像，梯度应该表现的像MSE。  
Adaptive WingLoss设计如下  
<img src="https://latex.codecogs.com/gif.latex?Awing(y,\hat{y})=\left\{\begin{matrix}&space;\omega&space;ln\left&space;(&space;1&plus;\left&space;|&space;\frac{y-\hat{y}}{\varepsilon&space;}&space;\right&space;|^{\alpha-y}&space;\right&space;)&space;&&space;if\left&space;|&space;y-\hat{y}<&space;\theta&space;\right&space;|\\&space;A\left&space;|&space;y-\hat{y}-C&space;\right&space;|&&space;otherwise&space;\end{matrix}\right." title="Awing(y,\hat{y})=\left\{\begin{matrix} \omega ln\left ( 1+\left | \frac{y-\hat{y}}{\varepsilon } \right |^{\alpha-y} \right ) & if\left | y-\hat{y}< \theta \right |\\ A\left | y-\hat{y}-C \right |& otherwise \end{matrix}\right." />  
w，θ，ε，α都是正数，根据|y-y^hat|=θ处连续可以求得A和C的值。  
使用变量θ来控制线性和非线性的切换。|y-y^hat|<θ时，α略大于2，当y的值接近于1，α-y将约等于1，此时损失函数接近Wingloss，前景的小错误有大的影响；当y接近0且误差接近0，影响将快速下降。见图  
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/adaptive_wing_loss.png, "loss")
