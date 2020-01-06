# BatchNormalization、LayerNormalization、InstanceNorm、GroupNorm、SwitchableNorm
![images](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/bn1.png "comparsion")  
## BN
对一个batch求均值和方差

## LN
对一个样例的所有features的值求均值和方差

## IN
对一个样例的每个通道的值求均值和方差

## GN
对通道分组，组内归一化

## SN
自适应的归一化
