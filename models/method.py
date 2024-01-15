import torch.nn.functional as F
from models.cats import TransformerAggregator
from functools import reduce, partial
from common.utils import *

class Method(nn.Module):

    def __init__(self,
                 args = None,
                 mode=None,
                 feature_size=10,
                 feature_proj_dim=10,
                 depth=1,
                 num_heads=2,
                 mlp_ratio=4):
        super().__init__()
        self.mode = mode
        self.args = args

        vit_dim = feature_size**2
        self.vit_dim = vit_dim
        hyperpixel_ids = [3]
        self.classification_head = nn.Linear(384, self.vit_dim)
        self.encoder_dim = vit_dim
        self.hyperpixel_ids = hyperpixel_ids

        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        self.proj = nn.Linear(vit_dim, feature_proj_dim)
        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))

    def corr(self, src, trg):
        outer_product_matrix = torch.bmm(src.unsqueeze(2), trg.unsqueeze(1))
        # 去除多余的维度
        return outer_product_matrix.squeeze()

    def forward(self, input):
        if self.mode == 'encoder':
            return self.encode(input)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def cca(self, spt, qry):

        # shifting channel activations by the channel mean
        # shape of spt : [25, 9]
        # shape of qry : [75, 9]
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # ----------------------------------cat--------------------------------------#

        # spt_feats = spt.unsqueeze(0).repeat(num_qry, 1, 1).view(-1,*spt.size()[1:]) #shape of spt_feats [75x25, 9]
        # qry_feats = qry.unsqueeze(1).repeat(1, way, 1).view(-1,*qry.size()[1:]) #[75x25, 9]
        #
        # corr = self.corr(spt_feats, qry_feats).unsqueeze(1).repeat(1,1,1,1) #the shape of corr : [75x25,2, 9, 9]
        # spt_feats_proj = self.proj(spt_feats).unsqueeze(1).unsqueeze(2).repeat(1, 1, self.vit_dim, 1) #[75x25,2,9,3]
        # qry_feats_proj = self.proj(qry_feats).unsqueeze(1).unsqueeze(2).repeat(1, 1, self.vit_dim, 1) #[75x25,2,9,3]
        #
        # refined_corr = self.decoder(corr, spt_feats_proj, qry_feats_proj).view(num_qry,way,*[self.feature_size]*4)
        # corr_s = refined_corr.view(num_qry, way, self.feature_size*self.feature_size, self.feature_size*self.feature_size)
        # corr_q = refined_corr.view(num_qry, way, self.feature_size*self.feature_size, self.feature_size*self.feature_size)
        #
        # # applying softmax for each side
        # corr_s = F.softmax(corr_s / 5.0, dim=2)
        # corr_q = F.softmax(corr_q / 5.0, dim=3)
        #
        # # suming up matching scores
        # attn_s = corr_s.sum(dim=[3])
        # attn_q = corr_q.sum(dim=[2])
        #
        # # applying attention
        # spt_attended = attn_s * spt_feats.view(num_qry, way, *spt_feats.shape[1:]) #[75, 25, 9]
        # qry_attended = attn_q * qry_feats.view(num_qry, way, *qry_feats.shape[1:]) #[75, 25, 9]

        # ----------------------------------cat--------------------------------------#

        spt_attended = spt.unsqueeze(0).repeat(num_qry, 1, 1)
        qry_attended = qry.unsqueeze(1).repeat(1, way, 1)

        # ----------------------------------replace--------------------------------------#

        # averaging embeddings for k > 1 shots
        if self.args.k_shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.k_shot, self.args.n_way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.k_shot, self.args.n_way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)


        # similarity_matrix = F.cosine_similarity(spt_attended, qry_attended, dim=-1)
        similarity_matrix = -F.pairwise_distance(spt_attended, qry_attended, p=2)
        return similarity_matrix / 0.2

    def encode(self, x):
        x = self.classification_head(x)
        return x

