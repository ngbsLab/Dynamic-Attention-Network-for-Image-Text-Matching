import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class SingleBlock(nn.Module):

    def __init__(self, num_block, embed_size, drop=0.2):
        super(SingleBlock, self).__init__()
        self.num_block = num_block
        self.embed_size = embed_size

    def DyIntraModalityUpdate(self, K, Q, opt, smooth):
        """
        Q: (n_context, sourceL, d)
        K: (n_context, sourceL, d)
        return (n_context, sourceL, sourceL)
        """
        batch_size, sourceL = K.size(0), K.size(1)
        K = torch.transpose(K, 1, 2).contiguous()
        attn = torch.bmm(Q, K)

        attn = attn.view(batch_size * sourceL, sourceL)
        attn = nn.Softmax(dim=1)(attn * smooth)
        attn = attn.view(batch_size, sourceL, -1)
        attnT = torch.transpose(attn, 1, 2)

        weightedK = torch.bmm(K, attnT)
        weightedK = torch.transpose(weightedK, 1, 2)
        return weightedK

    def InterModalityUpdate(self, query, context, opt, smooth):
        """
        Q: (batch, queryL, d)
        K: (batch, sourceL, d)
        return (batch, queryL, sourceL)
        """
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, sourceL)
        attn = nn.Softmax()(attn*smooth)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext

    def forward(self, images, captions, cap_lens, opt):

        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
            
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
            """
            new_img = self.DyIntraModalityUpdate(images, images, opt, smooth=opt.lambda_softmax)
            new_cap_i_expand = self.DyIntraModalityUpdate(cap_i_expand, cap_i_expand, opt, smooth=opt.lambda_softmax)
            weighted_img = self.InterModalityUpdate(new_cap_i_expand, new_img, opt, smooth=opt.lambda_softmax)
            weighted_cap = self.InterModalityUpdate(new_img, new_cap_i_expand, opt, smooth=opt.lambda_softmax)
            weighted_img = weighted_img.contiguous()
            weighted_cap = weighted_cap.contiguous()
            sim1 = cosine_similarity(images, weighted_cap, dim=2)
            sim1 = sim1.mean(dim=1, keepdim=True)
            sim2 = cosine_similarity(cap_i_expand, weighted_img, dim=2)
            sim2 = sim2.mean(dim=1, keepdim=True)
            row_sim = sim1 + sim2
            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)

        return similarities


