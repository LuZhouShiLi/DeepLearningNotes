# CPC模型源码阅读

* 将高维数据压缩到更紧凑的隐空间中，在其中条件预测更容易建模。
* 我们使用**强大的自回归模型在这个隐空间中预测未来的许多步骤**。
* 依靠**噪声对比估计（Noise-Contrastive Estimation） 来计算损失函数**（和自然语言模型中的学习单词嵌入方式类似），从而可以对整个模型进行端到端的训练。
* 最终提出的 Contrastive Predictive Codeing（CPC）方法可以用相同的机制在图像、语音、自然语言、强化学习等多个模态的数据上都能学习到高级信息。
 
* I(x,c) = H(x) - H(x|c) 说明了x和c的互信息表示由于c的引入而使得x熵减小的量，也就是x不确定度减小的量
* 那么最大化I(x,c)就是通过充分学习现在的上下文c最大程度减小了未来x的不确定度，从而起到了预测的效果，所以cpc希望网络可以最大化x和c之间的互信息


* 它可以**用于序列数据，也可以用于图片**
* CPC的目标就是要做unsupervised representation learning，并且我们希望这个representation有很好的predictive的能力。


github:```https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch```


## model.py

```python

"""
cpc 原始模型
"""
class CDCK2(nn.Module):
    
    # timestep时间步数 batch 批量大小  seq_len 序列长度
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK2, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
        else: return torch.zeros(1, batch_size, 256)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
        c_t = output[:,t_samples,:].view(batch, 256) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256

        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt

```


```py
self.batch_size = batch_size
self.seq_len = seq_len
self.timestep = timestep

```
* 这一部分是CDCK2类的构造函数，用于初始化该模型的实例。输入参数包括时间步长（timestep），批处理大小（batch_size），以及序列长度（seq_len）

```py
self.encoder = nn.Sequential( # downsampling factor = 160
    nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True)
)

```

* 这部分定义编码器encoder 他是一个nn.Sequential的模块，包含多个卷积层、批量标准化层、Relu激活函数。传入的一维数据按照顺序通过这些层处理
* 每一个卷积层的目的是提取输入数据的特征，而批量标准化层用于加速训练过程并且改善模型的泛化能力
* 最终编码器会对输入进行降采样，整体将采样率是160 由步长决定

```py
self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)

```
* 定义了一个单层、非双向的GRU循环神经网络，输入特征大小为512,隐藏层特征大小是256,并且批次大小指定在数据的第一维(batch_first=True)，预测未来某个采样点的值

```py
self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])

```

* 初始化一个模块列表 ModuleList 包含多个线性层nn.Linear ，每一个时间步一个，这些线性层被用来预测未来时间步的特征。模块列表中每一个线性层wk将GRU的输出(256维)映射回编码器的特征维度512维度

```py

self.softmax = nn.Softmax()
self.lsoftmax = nn.LogSoftmax()
```

* 定义softmax和 log softmax 函数，用于计算预测值的概率分布和对数概率分布，常用于分类问题和多分类的交叉熵损失计算
  
```py
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

```

* 私有函数weights_init用于权重初始化，会被用到类中的所有模块上面，对于线性层和卷积层，它使用Kaiming_normal初始化方法，这个方法基于ReLu激活函数的特征设置权重，对于批量标准化层，则设置权重为1和偏置为0

```py
self.apply(_weights_init)

```

* 这行代码应用_weights_init函数到模块的每一个子模块，即对self模块中所有子模块初始化其权重。

```py
def init_hidden(self, batch_size, use_gpu=True):
    if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
    else: return torch.zeros(1, batch_size, 256)

```
* init_hidden方法用于初始化GRU的隐藏层，这里返回一个全0的张量，其尺寸由GRU的层数，Batch和隐藏层特征大小


```py
def forward(self, x, hidden):
    batch = x.size()[0]
    t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long()

```

* 前向传播逻辑
* 确定批次大小和随机生成一个整数t_samples,该整数用于从序列中随机采样一个起始点，用于从编码器输出中提取时间步样本


```py
z = self.encoder(x)
z = z.transpose(1,2)

```

* 输出数据x通过编码器，并且转置层，以符合GRU输入的期望格式，这里假设x是一个三维张量，其中包含批次、通道数和长度维度

```py
nce = 0 # average over timestep and batch
encode_samples = torch.empty((self.timestep, batch, 512)).float()

```

* 初始化变量nce为0 该变量用于累计归一化交叉熵损失
* 初始化encode_samples张量以保存编码器输出中提取的样本，它们之后用于计算NCE损失
  
```py
for i in np.arange(1, self.timestep+1):
    encode_samples[i-1] = z[:, t_samples+i, :].view(batch, 512)
forward_seq = z[:, :t_samples+1, :]

```

* 在这个循环中，对于每一个时间步，从编码之后的数据z中选择特定的样本，并将它们存储在encode_samples张量中，同时，我们也创建了一个forward_seq变量，它包含从序列开始到随机选择的时间戳中的所有编码数据

```py
output, hidden = self.gru(forward_seq, hidden)
c_t = output[:, t_samples, :].view(batch, 256)

```


* forward_seq和隐藏状态传递给GRU层，并且计算输出和新的隐藏状态
* c_t包含了GRU输出的当前时刻的状态，用于预测未来的状态

```py
pred = torch.empty((self.timestep, batch, 512)).float()
for i in np.arange(0, self.timestep):
    linear = self.Wk[i]
    pred[i] = linear(c_t)

```

* 初始化pred张量，用于存储对未来时间步的预测
* 对于每个时间步，使用对应的线性层wk[i] 从当前状态c_t计算预测值，并且将结果保存在pred中

```py
for i in np.arange(0, self.timestep):
    total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
    correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
    nce += torch.sum(torch.diag(self.lsoftmax(total)))
nce /= -1.*batch*self.timestep
accuracy = 1.*correct.item()/batch

```

* 对于每个时间步长，计算真实的样本encode_samples[i] 与预测值pred[i] 之间的softmax分数
* 然后找出正确的预测correct，并且将这一个信息用于计算批次的精度accuracy
* nce变量被用来计算向量total对角线上每一个元素的log softmax和，它用于计算归一化交叉熵损失

```py
def predict(self, x, hidden):
    batch = x.size()[0]
    z = self.encoder(x)
    z = z.transpose(1,2)
    output, hidden = self.gru(z, hidden)
    return output, hidden

```

* predict方法的流程与forward类似，不过这里返回的是经过GRU处理之后的所有输出和最新的隐藏状态。这方便用于在验证阶段或者模型部署的时候进行序列预测






