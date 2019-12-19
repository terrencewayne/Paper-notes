# Preprocessing in Facial Landmark Localisation
以300W为例，解析人脸对齐中的数据预处理  
## 1.数据格式  
300w提供的是如下图片，往往超过一个人脸，且landmark是基于这样的图像给出的。为了训练网络我们必须进行裁剪、旋转操作，坐标值也要进行相应的转换。  

<img src="https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/3576294411_1.jpg" width="250" alt="example"/>  

每张图像的数据这样组织  
```
image_name: afw/3576294411_1.jpg   #文件名
scale: 1.17   #等效于人脸框的大小，后面分析
center_w: 906   #人脸框的中心点x
center_h: 399   #人脸框的中心点y
original_0_x: 795.043395   #第一个点x
original_0_y: 357.238158   #第一个点y
...
```

人脸框按下图定义  

<img src="https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/300wbox.png" width="250" alt="box"/>  

<img src="https://latex.codecogs.com/gif.latex?center_w&space;=&space;0.5(x_{min}&plus;x_{max})" title="center_w = 0.5(x_{min}+x_{max})" />  

<img src="https://latex.codecogs.com/gif.latex?center_h&space;=&space;0.5(y_{min}&plus;y_{max})" title="center_h = 0.5(y_{min}+y_{max})" />  

<img src="https://latex.codecogs.com/gif.latex?scale&space;=&space;max(w,h)/200" title="scale = max(w,h)/200" />  

## 2.图像裁剪、旋转
假定我们需要从原图切出人脸，大小为256X256  
```
def crop(img, center, scale, output_size, rot=0):
    center_new = center.clone()

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]   #确定原图的尺寸
    sf = scale * 200.0 / output_size[0]   #scale factor 放缩比例因子 = 人脸框大小/输出尺寸
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))   #要切出output尺寸的人脸，所需要的原图大小（长边）
        new_ht = int(np.math.floor(ht / sf))   #对应的高和宽
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)   #原图resize
            center_new[0] = center_new[0] * 1.0 / sf   #确定新的人脸框中心点坐标
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))   #找到切割的左上角点坐标
    # Bottom right point
    br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))   #找到切割的右下角点坐标

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)    #pad = 0.5（对角线长度-水平距离）
    if not rot == 0:    #如果旋转的话，将左上和右下角点向外平移pad，这是为了旋转之后所有的人脸框内的内容仍然在切割区域内
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)   #新建一个左上角到右下角的矩形区域

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:   #旋转
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = scipy.misc.imresize(new_img, output_size)   #resize到256X256
    return new_img
```
