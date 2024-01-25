# INT8

## 量化技术概述

* 在一些支持向量化计算设备上面，单精度的浮点计算可以同时进行八个
* 32bit 单精度浮点，用于传统的深度神经网络训练和推断工作，在大部分设备上面都可以很好的支持
* 8bit 通常是整形计算，这需要对深度学习模型进行量化，即将深度学习模型中的浮点数值变为整形计算，通常需要硬件支持 并且仅仅适用于推断过程

## 模型量化


* 如何使用低精度计算然后结果输出使用高精度
* 8bit计算相比于32bit可以减少4倍的内存消耗，并且提升2~4倍计算速度
* 量化就是使用整型数字进行计算，这意味着在计算过程中需要将浮点数子进行转换，然后转换之后的精度可以保证,这个得益于深度神经网络对于噪声的鲁棒性


## 实现方式

* 计算时量化，即模型训练和保存都是浮点数，仅仅在计算式进行量化，无法减少计算过程中的访存问题
* 训练后量化，这意味着模型在训练过程中是32位浮点数，然后推断过程中使用底Bit整型，模型保存之后是量化的，此时速度较快，但是精度会由于量化会变低
* 量化感知训练，也就是训练过程中正向计算是伪量化，而反向传播是32bit浮点，保存的模型同样是量化之后的，这种方式由于模型针对性的调整，因此精度比训练之后的量化要高  但是模型文件比较大~


### 导入相应的库

```py

import torch 
import torch.nn as nn 
import torch.nn.functional as F

```


### 定义Conv + BN + relu


```py
# 常规卷积层
class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU

        # 输出的feature map 大小不变
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, kernel_size, 
                      stride, padding, groups=groups, 
                      bias=False),
            norm_layer(n_out),
            nn.ReLU(inplace=True)
        )


```

### 定义BottleNeck

* Inverted residuals  先扩大输入通道  然后 depthwise 最后 1x1 缩小维度
* linear bottleneck到深度卷积之间的的维度比称为Expansion factor(扩展系数)
* 该系数控制了整个block的通道数
* 如果卷积层的过滤器都是使用低维的tensor来提取特征的话，那么就没有办法提取到整体的足够多的信息。
* 所以，如果提取特征数据的话，我们可能更希望有高维的tensor来做这个事情

```py

# Inverted residuals  先扩大输入通道  然后 depthwise 最后 1x1 缩小维度
# linear bottleneck到深度卷积之间的的维度比称为Expansion factor(扩展系数),
# 该系数控制了整个block的通道数
# 如果卷积层的过滤器都是使用低维的tensor来提取特征的话，那么就没有办法提取到整体的足够多的信息。
# 所以，如果提取特征数据的话，我们可能更希望有高维的tensor来做这个事情
# 残差结构
class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失

        # 计算扩张之后的维度
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        
        # 构建多层
        layers = []

        # 先扩张通道数目  低纬度 映射到高纬度
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, 
                            norm_layer=norm_layer))
            
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            norm_layer(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)

    # 如果使用残差结果  将输入x 和卷积块Layers的输出相加  然后 返回结果
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

```


## 量化模型

* 定义一个可分离卷积层的类InvertedResidual 这是MobileNetV2中的一个模块，它包含了扩展部分、深度卷积和逐点卷积 这是一个残差结构，允许信息流通过模块
* 首先扩大通道  然后逐层卷积  最后 恢复通道大小

```py
# Inverted residuals  先扩大输入通道  然后 depthwise 最后 1x1 缩小维度
# linear bottleneck到深度卷积之间的的维度比称为Expansion factor(扩展系数),
# 该系数控制了整个block的通道数
# 如果卷积层的过滤器都是使用低维的tensor来提取特征的话，那么就没有办法提取到整体的足够多的信息。
# 所以，如果提取特征数据的话，我们可能更希望有高维的tensor来做这个事情
# 残差结构
class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失

        # 计算扩张之后的维度
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        
        # 构建多层
        layers = []

        # 先扩张通道数目  低纬度 映射到高纬度
        if expand_ratio != 1:
            # 逐点卷积，增加通道数
            #  扩张维度  n_in -> hidden_dim  减少丢失信息

            #  还是使用1 x 1 卷积进行扩大
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, 
                            norm_layer=norm_layer))
            
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 逐点卷积，本层不加激活函数

            #  将通道的数目恢复到 n_out
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            norm_layer(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)

    # 如果使用残差结果  将输入x 和卷积块Layers的输出相加  然后 返回结果
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


```

## 定义量化版本的可分离卷积层

* 针对加法 使用量化计算方法
* 针对残差结构  重新forward方法
* 模型融合 找到卷积层 将卷积层和批量标准化层进行融合 将本个模块最后的卷积层和BN层进行融合

```py
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

class QInvertedResidual(InvertedResidual):
    """量化模型修改"""
    def __init__(self, *args, **kwargs):
        super(QInvertedResidual, self).__init__(*args, **kwargs)
        # 量化模型应当使用量化计算方法
        self.skip_add = nn.quantized.FloatFunctional()

    #  量化前向传播的残差结构
    # 重写InvertedResidual 的forward方法
    def forward(self, x):
        if self.use_res:
            # 量化加法
            return self.skip_add.add(x, self.conv(x))
            #return x + self.conv(x)
        else:
            return self.conv(x)
        

    # 模型融合
    def fuse_model(self):
        # 模型融合
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                # 找到卷积层 将卷积层和 批量标准化层 进行融合
                # 将本个模块最后的卷积层和BN层融合
                fuse_modules(
                    self.conv, 
                    [str(idx), str(idx + 1)], inplace=True)
                

```

## 定义model

```py
from torch.quantization import QuantStub, DeQuantStub
class Model(nn.Module):
    """
    手写数字识别模型
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            QInvertedResidual(3, 16, 1, 2), 
            QInvertedResidual(16, 32, 1, 2), 
            QInvertedResidual(32, 64, 2, 2), 
            QInvertedResidual(64, 128, 1, 2), 
            QInvertedResidual(128, 128, 2, 2),
            QInvertedResidual(128, 256, 1, 2)
        )
        # 量化函数
        self.quant = QuantStub()
        # 反量化函数
        self.dequant = DeQuantStub()
    def forward(self, x):
        #  对输入数据进行量化
        x = self.quant(x)

        #  通过模型的层次结构进行前向传播
        x = self.layers(x)

        # 对输出数据进行反量化
        x = self.dequant(x)
        return x
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()

```


## 训练
* 创建一个32的模型，切换到测试模式便于进行融合
* 配置模型的量化方式 量化观察者和权重观察者
* 使用fuse_model方法融合模型
* 将模型切换回训练模式 然后准备模型进行量化感知训练
* 执行完训练过程中之后切换为此时模式，将模型转换为int8,减小模型大小和提高推理速度

```py
# 量化感知训练
model_fp32 = Model() # 32bit浮点
model_fp32.eval()  # 调成测试模型才能融合

#  设置模型后端
model_fp32.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
    weight=torch.quantization.default_observer.with_args(dtype=torch.qint8))

# 融合模型
model_fp32.fuse_model()

# 融合后调整为训练模式
model_fp32 = model_fp32.train() 
# 准备模型
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
# 自行编写训练过程
optim = torch.optim.SGD(model_fp32_prepared.parameters(), 0.1) 

for itr in range(100):
    x = torch.randn([12, 3, 28, 28]) #给定训练数据
    y = model_fp32_prepared(x)
    # TODO:给定训练过程

#  推理
model_fp32_prepared.eval() # 调整为测试模式


# 将模型切换为测试模式  将模型转换为八位整数 int8 表示 以减小模型大小和提高推理速度
# 8bit模型
model_int8 = torch.quantization.convert(model_fp32_prepared)


```

## 测试

* 模型大小减小为原来的1/ 4
* Time consumption 未量化3.83s, 量化后2.13s, 加速比1.80

```py
import time 
x = torch.randn([50, 3, 64, 64]) 
model = model_fp32 
model.eval() 
t1 = time.perf_counter()
for i in range(10):
    y1 = model(x) 
t2 = time.perf_counter()


# 使用int8模型进行测试
model_int8.eval() 
t3 = time.perf_counter()
for i in range(10):
    y1 = model_int8(x) 
t4 = time.perf_counter()

# 计算时间  加速比
print("Time consumption", f"未量化{t2-t1:.2f}s, 量化后{t4-t3:.2f}s, 加速比{(t2-t1)/(t4-t3):.2f}")

# 保存模型文件 看看大小
# 一般模型
torch.save(model_fp32.state_dict(), "ckpt/fp32.pt") 

# 经过融合和量化感知训练的模型
torch.save(model_fp32_prepared.state_dict(), "ckpt/fp32_prepared.pt") 

# 量化之后的模型
torch.save(model_int8.state_dict(), "ckpt/int8.pt") 

```
