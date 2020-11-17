# DPatch: An Adversarial Patch Attack on Object Detectors
Xin Liu, Huanrui Yang, Ziwei Liu, Linghao Song, Hai Li, Yiran Chen
## Introduction
本文提出DPatch，黑盒攻击Faster-RCNN和YOLO，同时进行bbox回归和目标分类两方面的攻击。具有三个特点：  
1）可以实现无目标攻击和有目标攻击。前者指检测器无法预测目标的正确位置，后者指检测器只能检测出DPatch。  
2）DPatch很小，且位置独立。  
3）迁移性好，在不同数据集针对不同检测器得到的DPatch可以互相作用。

