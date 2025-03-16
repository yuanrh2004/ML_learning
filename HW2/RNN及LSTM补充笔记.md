对RNN和LSTM的非常好概述文章👍<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">链接</a><br>
# RNN

<p>RNN思路(图源见水印):</p>
<img src="https://pic2.zhimg.com/v2-2d8b631f3354893f91d5a7dc539f17e3_r.jpg" alt="RNN" width="450" height="250">
<p>biRNN思路(正向+反向)(图源见水印):</p>
<img src="https://i-blog.csdnimg.cn/blog_migrate/a474d4cd3bf27baa594e2742ed06c5dc.png" alt="biRNN" width="400" height="250">
RNN本质上就是在计算**当前输出的时候考虑了之前输入的影响**，在与时间相关的问题比较有用。

```python
# this is a copy of the code in https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# Efficient implementation equivalent to the following with bidirectional=False

# torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh',
#              bias=True, batch_first=False, dropout=0.0, bidirectional=False,
3              device=None, dtype=None)

def forward(x, hx=None):
    if batch_first:
        x = x.transpose(0, 1)          # 默认是(seq_len, batch_size, _)
    seq_len, batch_size, _ = x.size()
    if hx is None:
        hx = torch.zeros(num_layers, batch_size, hidden_size)
        # hx是由rnn层数 batch_size hidden_size(可以理解为经过rnn后y的维数) 组成的 每一个seq经过了rnn后就变了rnn_layers * hidden*size
    h_t_minus_1 = hx
    h_t = hx
    output = []
    for t in range(seq_len):  # 逐个时间步遍历 biRNN我认为就是seq_len - 1 → 0 这样再推一遍
      for layer in range(num_layers):  # 逐层计算
            h_t[layer] = torch.tanh(
            x[t] @ weight_ih[layer].T  # 当前输入 * 输入权重
            + bias_ih[layer]  # 输入偏置
            + h_t_minus_1[layer] @ weight_hh[layer].T  # 上一时间步隐藏状态 * 隐藏层权重
            + bias_hh[layer]  # 隐藏层偏置
            )
        # 这里感觉有点问题 for layer其实没有用哇 搜了些资料实现的基本隐藏层都为1 估计是这个原因
       output.append(h_t[-1]) 
       h_t_minus_1 = h_t
    output = torch.stack(output)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_t
```

# LSTM
LSTM相对于RNN的好处就是引入了**遗忘**这个概念，可以有效地解决梯度爆炸的问题。<br>
非常好的一篇文章，介绍了梯度爆炸的原因👍 <a href="https://blog.csdn.net/mary19831/article/details/129570030">链接</a><br>
同时笔记最开始的链接介绍得非常深入浅出，公式可以参考PyTorch官方文档给出的<a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html">链接</a> <br>

```python
# 参考:https://zhuanlan.zhihu.com/p/451985132

class lstm_origin(nn.Module):
  def __init__(self, input_size, hidden_size):
      super().__init___()
      self.input_size=input_size
      self.hidden_size=hidden_size

      # input
      self.wii=nn.Parameter(torch.Tensor(input_size, hidden_size))
      self.wih=nn.Parameter(torch.Tensor(hidden_size, hidden_size))
      self.bi=nn.Parameter(torch.Tensor(hidden_size)) # 偏置量就把xi的部分和hi的部分合在一起了

      #output
      self.wio=nn.Parameter(torch.Tensor(input_size, hidden_size))
      self.who=nn.Parameter(torch.Tensor(hidden_size, hidden_size))
      self.bo=nn.Parameter(torch.Tensor(hidden_size))

      #cell
      self.wic=nn.Patameter(torch.Tensor(input_size, hidden_size))
      self.whc=nn.Parameter(torch.Tensor(hidden_size, hidden_size))
      self.bc=nn.Parameter(torch.Tensor(hidden_size))

      #forget
      self.wif=nn.Patameter(torch.Tensor(input_size, hidden_size))
      self.whf=nn.Parameter(torch.Tensor(hidden_size, hidden_size))
      self.bf=nn.Parameter(torch.Tensor(hidden_size))


      self.init_weights()

  def forward(self, x):
      bs, seq_sz, _=x.size()  # 默认batch_size是第0维
      hidden_seq=[]
      h_t, c_t=(
              torch.zeros(bs,self.hidden_size).to(x.device),
              torch.zeros(bs,self.hidden_size).to(x.device),
            )
      for t in range(seq_sz):
          x_t=x[:, t, :]   # 只取第t时间步

          i_t=torch.sigmoid(x_t@self.wii + h_t@self.wih + self.bi)
          f_t=torch.sigmoid(x_t@self.wfi + h_t@self.wfh + self.bf)
          cp_t=torch.tanh(x_t@self.wci + h_t@self.wch + self.bc)
          o_t=torch.sigmoid(x_t@self.woi + h_t@self.woh + self.bo)
          c_t=f_t * c_t + i_t * cp_t
          h_t=o_t * torch.tanh(c_t)   #h_t也可以是实际的o_t
          # 现在的size是(batch_size, hidden_sz) 这里在最前面加了一维
          hidden_seq.append(h_t.unsqueeze(0))
      hidden_seq=torch.cat(hidden_seq, dim=0) #将每个seq的输出按dim=0拼接起来
      hidden_seq=hidden_seq.transpose(0, 1).contiguous() #又转变成(batch_sz, seq_sz, hidden_sz)
      return hidden_seq, (h_t, c_t)
          

    
      
```
