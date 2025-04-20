# -*- coding: utf-8 -*-
import math
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from modules import losses


class SelfAttention(nn.Module):
    def __init__(self, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = F.softmax(scores, dim=2)
        return torch.bmm(self.dropout(attention_weights), values)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(dilation, tuple):
            dilation = dilation[0]
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.w_q = nn.Linear(
            query_size, num_hiddens, bias=bias
        )  # CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.w_k = nn.Linear(
            key_size, num_hiddens, bias=bias
        )  # CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.w_v = nn.Linear(
            value_size, num_hiddens, bias=bias
        )  # CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        queries = self._transpose_qkv(self.w_q(queries), self.num_heads)
        keys = self._transpose_qkv(self.w_k(keys), self.num_heads)
        values = self._transpose_qkv(self.w_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = self._transpose_output(output, self.num_heads)
        return self.w_o(output_concat)

    # =============================================================================
    # 未引入Multi-head机制前：X[batch_size,seq_len,feature_dim]
    # 引入head后：X[batch_size*head_num,seq_len,feature_dim/head_num]
    # 定义 transpose_qkv()，tanspose_output() 函数实现上述转换：
    # =============================================================================
    def _transpose_qkv(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def _transpose_output(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


# =============================================================================
# batch_size, num_queries, num_hiddens, num_heads  = 2, 4, 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans.shape)
#
# attention = SelfAttention(dropout=0.5)
# batch_size, num_queries, num_hiddens  = 2, 4, 10
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans)
# =============================================================================


class GraphLearn(nn.Module):
    """
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    """

    def __init__(self, alpha: torch.Tensor, num_of_features: int, device: str) -> None:
        super().__init__()
        self.alpha = alpha
        self.a = nn.init.ones_(
            nn.Parameter(torch.FloatTensor(num_of_features, 1).to(device))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, v, f = x.shape
        diff = (
            x.expand(v, n, v, f).permute(2, 1, 0, 3) - x.expand(v, n, v, f)
        ).permute(
            1, 0, 2, 3
        )  # 62*61+62
        tmp_s = torch.exp(
            -F.relu(torch.reshape(torch.matmul(torch.abs(diff), self.a), [n, v, v]))
        )
        s = tmp_s / torch.sum(tmp_s, dim=1, keepdim=True)
        s_loss = losses.f_norm_loss(s, 1)
        dloss = losses.diff_loss(diff, s, self.alpha)
        ajloss = s_loss + dloss
        return s, ajloss


class ChebyshevConv(nn.Module):
    """
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_vertices, num_of_filters)
    """

    def __init__(
        self, num_of_filters: int, k: int, num_of_features: int, device: str
    ) -> None:
        super().__init__()
        self.theta = nn.ParameterList(
            [
                nn.init.uniform_(
                    nn.Parameter(
                        torch.FloatTensor(num_of_features, num_of_filters).to(device)
                    )
                )
                for _ in range(k)
            ]
        )
        self.out_channels = num_of_filters
        self.k = k
        self.device = device

    def forward(self, lst: list[torch.Tensor]) -> torch.Tensor:
        """
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = k.max(tf.self_adjoint_eigvals(L), axis=1)
            l_t = (2 * l) / torch.reshape(lambda_max, [-1, 1, 1]) - [torch.eye(int(num_of_vertices))]
        """
        assert isinstance(lst, list)
        x, w = lst
        n, v, _ = x.shape
        # Calculating Chebyshev polynomials
        d = torch.diag_embed(torch.sum(w, dim=1))
        l = d - w
        lambda_max = 2.0
        l_t = (2 * l) / lambda_max - torch.eye(int(v)).to(self.device)
        cheb_polynomials = [torch.eye(int(v)).to(self.device), l_t]
        for i in range(2, self.k):
            cheb_polynomials.append(
                2 * l_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]
            )
        # Graph Convolution
        outputs = []
        graph_signal = x  # (b, V, F_in)
        output = torch.zeros(n, v, self.out_channels).to(self.device)  # (b, V, F_out)
        for k in range(self.k):
            t_k = cheb_polynomials[k]  # (V,V)
            theta_k = self.theta[k]  # (in_channel, out_channel)
            rhs = t_k.matmul(graph_signal)
            output = output + rhs.matmul(
                theta_k
            )  # (b, V, F_in)(F_in, F_out) = (b, V, F_out)
        outputs.append(output)  # (b, V, F_out)
        return F.relu(torch.cat(outputs, dim=1))  # (b, V, F_out)


class GCNBlock(nn.Module):
    def __init__(self, net_params: dict[str, Any]) -> None:
        super().__init__()
        self.num_of_features = net_params["num_of_features"]
        device = net_params["DEVICE"]
        node_feature_hidden1 = net_params["node_feature_hidden1"]
        self.graph_learn = GraphLearn(
            net_params["GLalpha"], self.num_of_features, device
        )
        self.cheb_conv = ChebyshevConv(
            node_feature_hidden1, net_params["K"], self.num_of_features, device
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, ajloss = self.graph_learn(x)
        gcn = self.cheb_conv([x, s])
        return gcn, s, ajloss


class FeatureExtractor(nn.Module):
    def __init__(self, input_size: int, hidden_1: int, hidden_2: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        # x = F.leaky_relu(x)
        x2 = self.fc2(x1)
        x2 = F.relu(x2)
        # x = F.leaky_relu(x)
        return x1, x2


def aug_drop_node_list(
    graph_list: list[torch.Tensor], drop_percent: float
) -> torch.Tensor:
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        # aug_graph = aug_drop_node((graph_list[i]), drop_percent)
        aug_graph = aug_selet_node((graph_list[i]), drop_percent)
        aug_list.append(aug_graph)
    aug = torch.stack(aug_list, 0)
    aug = torch.flatten(aug, start_dim=1, end_dim=-1)
    return aug


def aug_selet_node(graph: torch.Tensor, drop_percent: float = 0.8) -> torch.Tensor:
    num = len(graph)  # number of nodes of one graph
    selet_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = graph.clone()
    all_node_list = [i for i in range(num)]
    selet_node_list = random.sample(all_node_list, selet_num)
    aug_graph = torch.index_select(
        aug_graph, 0, torch.IntTensor(selet_node_list).cuda()
    )
    return aug_graph


def aug_drop_node(graph: torch.Tensor, drop_percent: float = 0.2) -> torch.Tensor:
    num = len(graph)  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = graph.clone()
    all_node_list = [i for i in range(num)]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph[drop_node_list] = 0
    return aug_graph


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:  # L=nb_hidden_layers
        super().__init__()
        self.fc_layer1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc_layer2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        return x


class SemiGCL(nn.Module):
    def __init__(self, net_params: dict[str, Any]) -> None:
        super().__init__()
        self.device = net_params["DEVICE"]
        out_feature = net_params["node_feature_hidden1"]
        channel = net_params["num_of_vertices"]
        linearsize = net_params["linearsize"]
        self.drop_rate = net_params["drop_rate"]
        self.gcn = GCNBlock(net_params)
        self.domain_classifier2 = losses.DomainAdversarialLoss(hidden_1=64)
        self.domain_classifier3 = losses.TripleDomainAdversarialLoss(hidden_1=64)
        self.domain_classifier2_3 = losses.DomainAdversarialLossThreeada(hidden_1=64)
        self.fea_extrator_f = FeatureExtractor(310, 64, 64)
        self.fea_extrator_g = FeatureExtractor(
            int(channel * self.drop_rate) * out_feature, linearsize, 64
        )
        self.fea_extrator_c = FeatureExtractor(64 * 2, 64, 32)
        self.projection_head = ProjectionHead(64, 16)
        self.classifier = nn.Linear(64, net_params["category_number"])
        self.self_attention = MultiHeadAttention(128, 128, 128, 128, 64, 0.5)
        self.batch_size = net_params["batch_size"]
        self.category_number = net_params["category_number"]
        self.multi_att = net_params["Multi_att"]

    def forward(
        self, x: torch.Tensor, tripleada: bool = False, threshold: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature, _, ajloss = self.gcn(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extrator_f(feature1)
        if threshold:
            if tripleada:
                domain_output = self.domain_classifier3(feature1)
            else:
                domain_output = self.domain_classifier2_3(feature1)
        else:
            domain_output = self.domain_classifier2(feature1)
        aug_graph1, aug_graph2 = aug_drop_node_list(
            feature, self.drop_rate
        ), aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extrator_g(aug_graph1)
        _, aug_graph2_feature1 = self.fea_extrator_g(aug_graph2)

        aug_graph1_feature = self.projection_head(aug_graph1_feature1)
        aug_graph2_feature = self.projection_head(aug_graph2_feature1)

        l2 = torch.mean((aug_graph1_feature1 - aug_graph2_feature1) ** 2)

        sim_matrix_tmp2 = self.sim_matrix2(
            aug_graph1_feature, aug_graph2_feature, temp=1
        )
        row_softmax = nn.LogSoftmax(dim=1)
        row_softmax_matrix = -row_softmax(sim_matrix_tmp2)

        colomn_softmax = nn.LogSoftmax(dim=0)
        colomn_softmax_matrix = -colomn_softmax(sim_matrix_tmp2)

        row_diag_sum = self.compute_diag_sum(row_softmax_matrix)
        colomn_diag_sum = self.compute_diag_sum(colomn_softmax_matrix)
        contrastive_loss = (row_diag_sum + colomn_diag_sum) / (
            2 * len(row_softmax_matrix)
        )

        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1)

        class_feature = class_feature.unsqueeze(1)
        if self.multi_att:
            class_feature = self.self_attention(
                class_feature, class_feature, class_feature
            )
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extrator_c(class_feature)
        pred = self.classifier(class_feature)

        s_feature = class_feature[: self.batch_size]
        t_feature = class_feature[-self.batch_size :]

        sim_sample = self.sim_matrix2(s_feature, t_feature)
        sim_weight = torch.mean(sim_sample, dim=1).unsqueeze(1)
        sim_weight = torch.nn.functional.softmax(sim_weight, dim=0)

        return pred, domain_output, ajloss, contrastive_loss, sim_weight, l2

    def sharpen(self, predict: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e = torch.sum((predict) ** (1 / t), dim=1).unsqueeze(dim=1)
        predict = (predict ** (1 / t)) / e.expand(len(predict), self.category_number)
        return predict

    def compute_diag_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        num = len(tensor)
        diag_sum = torch.tensor(0).to(tensor)
        for i in range(num):
            diag_sum += tensor[i][i]
        return diag_sum

    def sim_matrix2(
        self,
        ori_vector: torch.Tensor,
        arg_vector: torch.Tensor,
        temp: float = 1.0,
    ) -> torch.Tensor:
        for i, ori_vec in enumerate(ori_vector):
            sim = torch.cosine_similarity(ori_vec.unsqueeze(0), arg_vector, dim=1) * (
                1 / temp
            )
            if i == 0:
                sim_tensor = sim.unsqueeze(0)
            else:
                sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
        return sim_tensor

    def sim_matrix(
        self, s_vector: torch.Tensor, t_vector: torch.Tensor
    ) -> torch.Tensor:
        pdist = nn.PairwiseDistance(p=1)  # p=2 就是计算欧氏距离，p=1 就是曼哈顿距离
        for i, s_vec in enumerate(s_vector):
            sim = pdist(s_vec.unsqueeze(0).repeat(len(s_vector), 1), t_vector)
            if i == 0:
                sim_tensor = sim.unsqueeze(0)
            else:
                sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
        sim_tensor = torch.exp(torch.neg(sim_tensor))
        return sim_tensor.to(self.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        feature, _, _ = self.gcn(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extrator_f(feature1)
        aug_graph1 = aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extrator_g(aug_graph1)
        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1)
        class_feature = class_feature.unsqueeze(1)
        if self.multi_att:
            class_feature = self.self_attention(
                class_feature, class_feature, class_feature
            )
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extrator_c(class_feature)
        pred = self.classifier(class_feature)
        label_feature = torch.nn.functional.softmax(pred, dim=1)
        return label_feature
