# 解析Torch.einsum()

torch.einsum用来实现张量的乘法，使用爱因斯坦简记法。  
```
torch.einsum(equation, *operands)
```
**equations**: 示例'bij,bjk->bik'，使用小写字母，代表操作数和结果中的维度。箭头左边代表操作数的各个维度，箭头右边代表结果的各个维度。没有出现在结果中的维度进行乘法并加和。示例中对b维度的每一个ixj元素和jxk元素相乘，得到ixk矩阵
