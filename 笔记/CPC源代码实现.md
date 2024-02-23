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


