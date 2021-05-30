import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import numpy as np
import math
import sys


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    dimensions: dims = [d0, d1, d2...]
        """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2, .5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
                                  name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, self.glimpse, v_num, q_num), logits

        return logits


class Counter(nn.Module):
    """ Counting module as proposed in [1].
    Count the number of objects from a set of bounding boxes and a set of scores for each bounding box.
    This produces (self.objects + 1) number of count features.

    """

    def __init__(self, objects, already_sigmoided=False):
        super().__init__()
        self.objects = objects
        self.already_sigmoided = already_sigmoided
        self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(16)])

    def forward(self, boxes, attention):
        """ Forward propagation of attention weights and bounding boxes to produce count features.
        `boxes` has to be a tensor of shape (n, 4, m) with the 4 channels containing the x and y coordinates of the top left corner and the x and y coordinates of the bottom right corner in this order.
        `attention` has to be a tensor of shape (n, m). Each value should be in [0, 1] if already_sigmoided is set to True, but there are no restrictions if already_sigmoided is set to False. This value should be close to 1 if the corresponding boundign box is relevant and close to 0 if it is not.
        n is the batch size, m is the number of bounding boxes per image.
        """
        # only care about the highest scoring object proposals
        # the ones with low score will have a low impact on the count anyway
        boxes, attention = self.filter_most_important(self.objects, boxes, attention)
        # normalise the attention weights to be in [0, 1]
        if not self.already_sigmoided:
            attention = torch.sigmoid(attention)

        relevancy = self.outer_product(attention)
        distance = 1 - self.iou(boxes, boxes)

        # intra-object dedup
        score = self.f[0](relevancy) * self.f[1](distance)

        # inter-object dedup
        dedup_score = self.f[3](relevancy) * self.f[4](distance)
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attention)
        score = score / dedup_per_entry

        # aggregate the score
        # can skip putting this on the diagonal since we're just summing over it anyway
        correction = self.f[0](attention * attention) / dedup_per_row
        score = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)
        score = (score + 1e-20).sqrt()
        one_hot = self.to_one_hot(score)

        att_conf = (self.f[5](attention) - 0.5).abs()
        dist_conf = (self.f[6](distance) - 0.5).abs()
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + dist_conf.mean(dim=2).mean(dim=1, keepdim=True))

        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        # using outer-diffs
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        # similarity for each row
        row_sims = sim.sum(dim=2)
        # similarity for each entry
        all_sims = self.outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        """ Turn a bunch of non-negative scalar values into a one-hot encoding.
        E.g. with self.objects = 3, 0 -> [1 0 0 0], 2.75 -> [0 0 0.25 0.75].
        """
        # sanity check, I don't think this ever does anything (it certainly shouldn't)
        scores = scores.clamp(min=0, max=self.objects)
        # compute only on the support
        i = scores.long().data
        f = scores.frac()
        # target_l is the one-hot if the score is rounded down
        # target_r is the one-hot if the score is rounded up
        target_l = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects + 1).fill_(0)

        target_l.scatter_(dim=1, index=i.clamp(max=self.objects), value=1)
        target_r.scatter_(dim=1, index=(i + 1).clamp(max=self.objects), value=1)
        # interpolate between these with the fractional part of the score
        return (1 - f) * target_l + f * target_r

    def filter_most_important(self, n, boxes, attention):
        """ Only keep top-n object proposals, scored by attention weight """
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx)
        return boxes, attention

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        # Y_ij = x_i * x_j
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        # like outer products, except taking the absolute difference instead
        # Y_ij = | x_i - x_j |
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area


class PiecewiseLin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = y.long().data
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False  # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        print('weight_init.shape:', weight_init.shape)
        print('self ntoken: ', self.ntoken)
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init)  # (N x N') x (N', F)
            if 'c' in self.op:
                self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),
                    weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

BertLayerNorm = torch.nn.LayerNorm


class BertConfig():
    def __init__(self, hidden_size=1280, num_attention_heads=8,
                 intermediate_size=2560, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, hidden_act='gelu'):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act


config = BertConfig()


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    # print('scores:', scores.shape)
    return scores


### BAN Models




class QuestionAnswerAdditional(nn.Module):

    def __init__(self, num_hid, num_ans_candidates):
        super().__init__()
        # first implement 1-head unitary attention

        config.hidden_size = num_hid
        self.num_l_layers = 3
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )

        self.classifier = SimpleClassifier(config.hidden_size,
                                           config.hidden_size * 2, num_ans_candidates, dropout=.5)

    def forward(self, q_emb):
        # input: q_emb_base.shape= torch.Size([128, 14, 1280])
        # [batch_size, q_len, num_hid]
        # output: output of classifer [batch_size, num_ans_candidates
        attention_masks = None
        for layer_module in self.bert_layers:
            q_emb = layer_module(q_emb, attention_masks)

        return self.classifier(q_emb.mean(1))


class VisualAnswerAdditional(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, v, b):
        raise Exception('visual answer additional Not Implemented')
        pass

class BanModel(nn.Module):
    def __init__(self, ntoken, num_ans_candidates, num_hid, v_dim, op='', gamma=4, qa_bl=False):
        super(BanModel, self).__init__()

        self.op = op
        self.glimpse = gamma
        self.qa_bl = qa_bl
        self.w_emb = WordEmbedding(ntoken, 300, .0, op)
        self.q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
        self.v_att = BiAttention(v_dim, num_hid, num_hid, gamma)
        b_net = []
        q_prj = []
        c_prj = []
        objects = 10  # minimum number of boxes
        for i in range(gamma):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)
        self.counter = Counter(objects)

        if self.qa_bl:
            self.qa_add_layers = QuestionAnswerAdditional(num_hid=num_hid,
                                                          num_ans_candidates=num_ans_candidates)

        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]

        if self.qa_bl:
            logits_qa = self.qa_add_layers(q_emb)
            return logits_qa, None
        else:
            boxes = b[:, :, :4].transpose(1, 2)



            b_emb = [0] * self.glimpse
            att, logits = self.v_att.forward_all(v, q_emb)  # b x g x v x q

            for g in range(self.glimpse):
                b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

                atten, _ = logits[:, g, :, :].max(2)
                embed = self.counter(boxes, atten)

                q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

            logits = self.classifier(q_emb.sum(1))

            return logits, att


class BanPreModel(nn.Module):
    def __init__(self, ntoken, num_ans_candidates, num_hid, v_dim, op='', gamma=4, pretrain_tasks=[]):
        super(BanPreModel, self).__init__()
        """ pretrain tasks in 'vqa', 'qa', 'vq', 'va', 'va2' """
        print('pretrain tasks = ', ', '.join(pretrain_tasks))
        self.pretrain_tasks = pretrain_tasks

        self.op = op
        self.glimpse = gamma
        self.w_emb = WordEmbedding(ntoken, 300, .0, op)
        self.q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
        self.v_att = BiAttention(v_dim, num_hid, num_hid, gamma)
        b_net = []
        q_prj = []
        c_prj = []
        objects = 10  # minimum number of boxes
        for i in range(gamma):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)
        self.counter = Counter(objects)

        if 'qa' in self.pretrain_tasks:
            self.qa_add_layers = QuestionAnswerAdditional(num_hid=num_hid,
                                                          num_ans_candidates=num_ans_candidates)

        if 'vq' in self.pretrain_tasks:
            self.gamma_vq_shared = self.glimpse - 1
            self.gamma_vq_add = 1
            b_net_vq = b_net[:self.gamma_vq_shared]
            q_prj_vq = q_prj[:self.gamma_vq_shared]
            c_prj_vq = c_prj[:self.gamma_vq_shared]
            objects_vq = 10  # minimum number of boxes
            for i in range(self.gamma_vq_add):
                b_net_vq.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
                q_prj_vq.append(FCNet([num_hid, num_hid], '', .2))
                c_prj_vq.append(FCNet([objects + 1, num_hid], 'ReLU', .0))

            self.b_net_vq = nn.ModuleList(b_net_vq)
            self.q_prj_vq = nn.ModuleList(q_prj_vq)
            self.c_prj_vq = nn.ModuleList(c_prj_vq)
            self.classifier_vq = SimpleClassifier(
                num_hid, num_hid * 2, 2, .5)

            self.counter_vq = Counter(objects_vq)

        if 'va' in self.pretrain_tasks:
            self.gamma_va_shared = self.glimpse - 1
            self.gamma_va_add = 1
            b_net_va = b_net[:self.gamma_va_shared]
            q_prj_va = q_prj[:self.gamma_va_shared]
            c_prj_va = c_prj[:self.gamma_va_shared]
            objects_va = 10  # minimum number of boxes
            for i in range(self.gamma_va_add):
                b_net_va.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
                q_prj_va.append(FCNet([num_hid, num_hid], '', .2))
                c_prj_va.append(FCNet([objects + 1, num_hid], 'ReLU', .0))

            self.b_net_va = nn.ModuleList(b_net_va)
            self.q_prj_va = nn.ModuleList(q_prj_va)
            self.c_prj_va = nn.ModuleList(c_prj_va)
            self.classifier_va = SimpleClassifier(
                num_hid, num_hid * 2, 2, .5)

            self.counter_va = Counter(objects_va)

        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, question, feats, spatials,
                vq_matched, match_question, va_matched, answer_rps, label,
                ans_valid, ans_rps_valid):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        v = feats
        b = spatials
        q = question

        w_emb = self.w_emb(q)
        q_emb_base = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        att_base, logits_base = self.v_att.forward_all(v, q_emb_base)  # b x g x v x q
        # print('att_base.shape=', att_base.shape, 'logits_base.shape', logits_base.shape,
        #       'q_emb_base.shape=', q_emb_base.shape)

        q_emb, att, logits = q_emb_base, att_base, logits_base

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = q_emb + self.q_prj[g](b_emb[g].unsqueeze(1))
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        if 'qa' in self.pretrain_tasks:
            logits_qa = self.qa_add_layers(q_emb_base)

        if 'vq' in self.pretrain_tasks:
            w_emb_vq = self.w_emb(match_question)
            q_emb_vq = self.q_emb.forward_all(w_emb_vq)  # [batch, q_len, q_dim]
            # boxes = b[:, :, :4].transpose(1, 2)

            b_emb_vq = [0] * self.glimpse
            # att_vq, logits_vq = self.v_att.forward_all(v, q_emb_vq)  # b x g x v x q
            att_vq, logits_vq = att_base, logits_base

            for g in range(self.gamma_vq_shared + self.gamma_vq_add):
                b_emb_vq[g] = self.b_net_vq[g].forward_with_weights(v, q_emb_vq, att_vq[:, g, :, :])  # b x l x h

                atten_vq, _ = logits_vq[:, g, :, :].max(2)
                embed_vq = self.counter_vq(boxes, atten_vq)

                q_emb_vq = self.q_prj_vq[g](b_emb_vq[g].unsqueeze(1)) + q_emb_vq
                q_emb_vq = q_emb_vq + self.c_prj_vq[g](embed_vq).unsqueeze(1)

            logits_vq = self.classifier_vq(q_emb_vq.sum(1))

        if 'va' in self.pretrain_tasks:
            w_emb_va = self.w_emb(answer_rps)
            q_emb_va = self.q_emb.forward_all(w_emb_va)  # [batch, q_len, q_dim]
            # boxes = b[:, :, :4].transpose(1, 2)

            b_emb_va = [0] * self.glimpse
            att_va, logits_va = att_base, logits_base

            for g in range(self.gamma_va_shared + self.gamma_va_add):
                b_emb_va[g] = self.b_net_va[g].forward_with_weights(v, q_emb_va, att_va[:, g, :, :])  # b x l x h

                atten_va, _ = logits_va[:, g, :, :].max(2)
                embed_va = self.counter_va(boxes, atten_va)

                q_emb_va = self.q_prj_va[g](b_emb_va[g].unsqueeze(1)) + q_emb_va
                q_emb_va = q_emb_va + self.c_prj_va[g](embed_va).unsqueeze(1)

            logits_va = self.classifier_va(q_emb_va.sum(1))

        if 'va2' in self.pretrain_tasks:
            logits_va2 = self.va2_add_layers(v, b)

        total_loss = 0.0
        if 'vaa' in self.pretrain_tasks:
            loss_vqa = instance_bce_with_logits(logits, label)
            total_loss += loss_vqa
            pass
        if 'qa' in self.pretrain_tasks:
            loss_qa = instance_bce_with_logits(logits_qa, label)
            total_loss += loss_qa
            pass
        if 'vq' in self.pretrain_tasks:
            loss_vq = instance_bce_with_logits(logits_vq, vq_matched)
            total_loss += loss_vq
            pass
        if 'va' in self.pretrain_tasks:
            loss_va = instance_bce_with_logits(logits_va[ans_rps_valid],
                                               va_matched[ans_rps_valid])
            total_loss += loss_va
            pass
        if 'va2' in self.pretrain_tasks:
            loss_va2 = instance_bce_with_logits(logits_va2[ans_valid],
                                                label[ans_valid])
            total_loss += loss_va2

        return total_loss
