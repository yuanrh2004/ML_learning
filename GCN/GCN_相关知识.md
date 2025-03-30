<p>写这篇笔记是因为要修改一段原作者自己写的GCN部分的代码为torch_geometric部分的代码，而之前经常在生物、分子等等领域阅读到GCN相关模块但没有深入了解过。所以来紧急学习一下~<br>
<p>主要结合代码分析，讲解GCN原理的博客还蛮多的！学习时主要参考的下面几篇：<br>
1. <a href="https://mp.weixin.qq.com/s/I3MsVSR0SNIKe-a9WRhGPQ" title="链接">比较简单易懂的GCN原理解释</a><br>
2. <a href="https://tkipf.github.io/graph-convolutional-networks/" title="链接">作者本人的博客</a><br>

# torch_geometric中的实现方式
<p><a href="https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv.forward" title="gcn_conv">gcn_conv</a><br>

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

2. **gcn_norm部分** - 这部分是gcn的主要逻辑
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
        assert edge_index.size(0) == edge_index.size(1)  # 如果是sparsetensor格式 检查是不是方阵

        adj_t = edge_index

        if not adj_t.has_value():    # 检查adj_t是不是有存储边权 如果没有赋1
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:           # 检查是不是要添加自环 如果要就按照fill_value的情况添加
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
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
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight

```


