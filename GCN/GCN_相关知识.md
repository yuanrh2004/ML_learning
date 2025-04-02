<p>写这篇笔记是因为要修改一段原作者自己写的GCN部分的代码为torch_geometric部分的代码，而之前经常在生物、分子等等领域阅读到GCN相关模块但没有深入了解过。所以来紧急学习一下~<br>
<p>主要结合代码分析，讲解GCN原理的博客还蛮多的！学习时主要参考的下面几篇：<br>
1. <a href="https://mp.weixin.qq.com/s/I3MsVSR0SNIKe-a9WRhGPQ" title="链接">比较简单易懂的GCN原理解释</a><br>
2. <a href="https://tkipf.github.io/graph-convolutional-networks/" title="链接">作者本人的博客</a><br>

# torch_geometric中的实现方式
<p><a href="https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv.forward" title="gcn_conv">gcn_conv的官方代码地址</a><br>
  
![GCN核心公式](https://mmbiz.qpic.cn/mmbiz_png/QLDSy3Cx3YI5Td3Foo3KjnKQYYSh21NwNtkAYWoDgQjLEtdupCNJGw9quCiaos0qiafySOOR5sCPlT1I0SpSkOsw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1 "GCN核心公式") <br>
从上式可以发现，不论传递多少层， 左半部分是不会改变的。而这个部分就是对图的连接的归一化。相当于我的图的特征由X变成了Z，但是我的**图的连接关系**是始终**被保留**住的。
  
  1. **pytorch中的"图"**
     1. **稠密图** - 邻接矩阵
     2. **edge_index 方式**

    
        ```python
        edge_index = torch.tensor([
        [0, 1, 2, 3],  # 源节点 (row)
        [1, 2, 3, 1]   # 目标节点 (col)
        ])
        ```
     3. **SparseTensor 格式**

        等效于2，但是效率更好。
        这里就可以发现 `edge_index[0]` 是源结点序列，`edge_index[1]`是目标结点序列。
        我的理解是: SparseTensor存储的时候可以理解成是edge_index的形式，在使用的时候可以把它理解成邻接矩阵。就比如后面的`torch_sparse.sum(adj_x, dim = 1)` 就是在计算出度。
        ```python
        from torch_sparse import SparseTensor
        # 用稀疏矩阵存储图结构
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 1]
        ])
        num_nodes = 4
        
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))# 这里sparsetensor的
        print(adj_t)
        ```

2. **gcn_norm部分** - 这部分是gcn的归一化部分<br>

   代码理解补充：<br>
   - `.view()`操作<br>
      例如`x.view(1, -1)`就是将x转换成行向量，保证是`1 * N`。
   - `SparseTensor` 与 `sparse`<br>
     注意这里第一种是来自PyG库的torch_geometric，第二种是来自torch内置的sparse
```python
@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass

## 前面这两部分是为了优化编译部分 这里就不深究啦
 
def gcn_norm(  # noqa: F811
    edge_index: Adj,                      # 这里edge_index
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,               # improved表示GCN-improved版 从下面看其实就是fill_value的差别
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.  # 这里指示的A添加自环时的填充值

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)
        # 如果是sparsetensor格式 确保源的个数和目的的个数是相等的

        adj_t = edge_index

        if not adj_t.has_value():    # 检查adj_t是不是有存储边权 如果没有赋1
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:           # 检查是不是要添加自环 如果要就按照fill_value的情况添加
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        # 这部分是在实现 D'AD
        deg = torch_sparse.sum(adj_t, dim=1)  # 在dim = 1求和 就是在求出度
        deg_inv_sqrt = deg.pow_(-0.5)  # 度数矩阵1/sqrt(x)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.) # 把除0变成inf的部分赋0
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))   
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t
    # 下面开始处理torch内置的sparse
    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)  # 这里将adj_t分成两部分: edge_index和value
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        # 表示value根据col进行分组 0表示scatter操作的维度(即对行进行分组) dim_size是结点数
        # reduce指定聚合的方式
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    # 选择用出度还是入度
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight

```

3. **X->Z的网络部分**

  这部分其实就是比较正常的定义网络层、传播的环节。<br>
  涉及到的公式：图卷积公式、节点级公式（结点𝑖的新特征等于其邻居特征（包括本身）的加权和）
  

```python
class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            in case :obj:`normalize` is set to :obj:`True`, and not added
            otherwise. (default: :obj:`None`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,  # V表示结点数
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
```
   

