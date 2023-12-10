#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from concept import ConceptNet
from encoders import EncoderImage, EncoderText, PatchEncoderImage
from program_engine import ProgramEngine
from utils import SimpleModule, l2norm, params_require_grad


class L2Module(nn.Module):

    def __init__(self):
        super(L2Module, self).__init__()

    def forward(self, x):
        x = l2norm(x)
        return x

class AttentionMechanism(nn.Module):
    """
    Module defining the architecture of the attention mechanisms in ARTEMIS.
    """

    def __init__(self, opt):
        super(AttentionMechanism, self).__init__()

        self.embed_dim = opt.embed_dim
        input_dim = self.embed_dim

        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)

class MyModel(nn.Module):
    """
    ARTEMIS: Attention-based Retrieval with Text-Explicit Matching and Implicit Similarity,
    ICLR 2022
    """

    def __init__(self, word2idx, all_nouns_token, noun_token_id_to_idx, opt):
        super(MyModel, self).__init__()

        self.embed_dim = opt.embed_dim
        self.all_nouns_token = all_nouns_token
        self.noun_token_id_to_idx = noun_token_id_to_idx

        # Text encoder & finetuning
        self.txt_enc = EncoderText(word2idx, opt)
        params_require_grad(self.txt_enc.embed, opt.txt_finetune)

        # Image encoder & finetuning
        if opt.load_image_feature:
            self.img_enc = SimpleModule(opt.load_image_feature, self.embed_dim)
            # needs to be learned --> not conditioned on opt.img_finetune
        else :
            self.img_enc = PatchEncoderImage(opt)
            params_require_grad(self.img_enc.model, opt.img_finetune)

        # potentially learn the loss temperature/normalization scale at training time
        # (stored here in the code for simplicity)
        self.temperature = nn.Parameter(torch.FloatTensor((opt.temperature,)))

        # --- modules
        self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())
        self.Attention_EM = AttentionMechanism(opt)
        self.Attention_IS = AttentionMechanism(opt)

        # --- scoring strategy
        self.model_version = opt.model_version
        self.compute_score = self.compute_score_cpt
        self.compute_score_broadcast = self.compute_score_broadcast_cpt

        # --- for heatmap processing
        self.gradcam = opt.gradcam
        self.hold_results = dict() # holding intermediate results

        #####################
        #self.concept_net = ConceptNet(3221, 49, 512, 2, 8, 4)
        self.concept_net = ConceptNet(self.embed_dim, opt.concept_depth, opt.concept_heads)
        self.noun_encoder = nn.Linear(opt.word_dim, self.embed_dim)
        self.program_engine = ProgramEngine(
            self.embed_dim, 
            opt.engine_depth, 
            opt.num_type_instruction, 
            opt.num_instruction_generator_fc_layer, 
            opt.num_parameter_generator_fc_layer,
            opt.module_depth, 
            opt.module_heads, 
            opt.module_dim_head,
        )


    ############################################################################
    # *** SCORING METHODS
    ############################################################################

    # All scoring methods exist in 2 versions:
    # - a "regular" version, which returns a tensor of shape (batch_size), where
    #   coefficient (i) is the score between query (i) and target (i). 
    # - a broadcast version, which returns a tensor of shape (batch_size,
    #   batch_size), corresponding to the score matrix where coefficient (i,j)
    #   is the score between query (i) and target (j).

    # Shape notations in comments:
    #   Bq: "query" batch size (number of queries) [in practice, Bq = B]
    #   Bt: "target" batch size (number of targets) [in practice, Bt = B]
    #   d: embedding dimension

    def inference_fg_bg(self, x):
        img_feat = self.get_image_embedding(x)
        tokens, fg_feat, bg_feat = self.concept_net(img_feat)
        return tokens, fg_feat, bg_feat


    def compute_score_cpt(self, r_token, r_fg, r_bg, m, t_token, t_fg, t_bg, sentence_len, noun_token_id, noun_len, word_feats):
        cls_weights = self.noun_encoder(self.txt_enc.embed(self.all_nouns_token.cuda()))
        concepts = self.gen_cpt(cls_weights, noun_token_id, noun_len)

        if word_feats.size(1) != sentence_len:
            sentence_len += word_feats.size(1) - sentence_len
        word_mask = torch.arange(sentence_len.max())[None, :].cuda() < sentence_len[:, None]
        concept_mask = torch.arange(noun_len.max())[None, :].cuda() < noun_len[:, None]

        inputs_token = torch.cat((r_fg.unsqueeze(1), r_token), dim=1)
        hat_t = self.program_engine(inputs_token, m, word_feats, word_mask, concepts, concept_mask)
        artemis_score = self.artemis(r_fg, r_bg, m, t_fg, t_bg)
        program_score = (l2norm(hat_t) * l2norm(t_fg)).sum(-1)
        score = program_score + artemis_score

        return score

    def compute_score_broadcast_cpt(self, r, m, t, sentence_len, noun_token_id, noun_len, word_feats):
        src_trg_fg, src_token, src_fg, src_bg, trg_token, trg_fg, trg_bg = self.concept_net(r, t)
        cls_weights = self.noun_encoder(self.txt_enc.embed(self.all_nouns_token.cuda()))
        cls_logits = src_trg_fg @ cls_weights.t()

        concepts = self.gen_cpt(cls_weights, noun_token_id, noun_len)
        word_mask = torch.arange(sentence_len.max())[None, :].cuda() < sentence_len[:, None]
        concept_mask = torch.arange(noun_len.max())[None, :].cuda() < noun_len[:, None]
        artemis_score = self.artemis_broadcast(src_fg, src_bg, m, trg_fg, trg_bg)

        inputs_token = torch.cat((src_fg.unsqueeze(1), src_token), dim=1)
        hat_t = self.program_engine(inputs_token, m, word_feats, word_mask, concepts, concept_mask)
        program_score = (l2norm(hat_t.view(r.size(0), 1, self.embed_dim)) * l2norm(trg_fg)).sum(-1)
        score = program_score + artemis_score

        return score, cls_logits
    
    @torch.no_grad()
    def gen_cpt(self, cls_weights, noun_token_id, noun_length):
        bs = len(noun_token_id)
        cpts = torch.zeros((bs, noun_length.max(), self.embed_dim), dtype=cls_weights.dtype).cuda()
        for i in range(bs):
            for j in range(noun_length[i]):
                key = noun_token_id[i][j]
                weight_idx = self.noun_token_id_to_idx.get(key.item())
                if weight_idx:
                    cpts[i][j] = cls_weights[weight_idx]
                else:
                    cpts[i][j] = self.noun_encoder(self.txt_enc.embed(key.cuda()))
        return cpts

    def compute_score_FG(self, r, m, t):
        return self.compute_score_EM(r, m, t)

    def compute_score_BG(self, r, m, t):
        return self.compute_score_IS(r, m, t)


    def compute_score_broadcast_FG(self, r, m, t):
        return self.compute_score_broadcast_EM(r, m, t)

    def compute_score_broadcast_BG(self, r, m, t):
        return self.compute_score_broadcast_IS(r, m, t)


    #############################################################################

    def apply_attention(self, a, x):
        return l2norm(a * x)
    
    def compute_score_artemis(self, r, m, t, store_intermediary=False):
        EM = self.compute_score_EM(r, m, t, store_intermediary)
        IS = self.compute_score_IS(r, m, t, store_intermediary)
        if store_intermediary:
            self.hold_results["EM"] = EM
            self.hold_results["IS"] = IS
        return EM + IS
    def compute_score_broadcast_artemis(self, r, m, t):
        return self.compute_score_broadcast_EM(r, m, t) + self.compute_score_broadcast_IS(r, m, t)

    def compute_score_EM(self, r, m, t, store_intermediary=False):
        Tr_m = self.Transform_m(m)
        A_EM_t = self.apply_attention(self.Attention_EM(m), t)
        if store_intermediary:
            self.hold_results["Tr_m"] = Tr_m
            self.hold_results["A_EM_t"] = A_EM_t
        return (Tr_m * A_EM_t).sum(-1)

    def compute_score_broadcast_EM(self, r, m, t):
        batch_size = r.size(0)
        A_EM = self.Attention_EM(m) # shape (Bq, d)
        Tr_m = self.Transform_m(m) # shape (Bq, d)
        # apply each query attention mechanism to all targets
        A_EM_all_t = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim)) # shape (Bq, Bt, d)
        EM_score = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j
        return EM_score

    def compute_score_IS(self, r, m, t, store_intermediary=False):
        return (F.normalize(r, p=2, dim=-1) * F.normalize(t, p=2, dim=-1)).sum(-1)

    def compute_score_broadcast_IS(self, r, m, t):
        batch_size = r.size(0)
        r = F.normalize(r, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        return (r.view(batch_size, 1, self.embed_dim) * t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j

    def artemis(self, r_fg, r_bg, m, t_fg, t_bg):
        fg_score = self.compute_score_FG(r_fg, m, t_fg)
        bg_score = self.compute_score_BG(r_bg, m, t_bg)
        return fg_score + bg_score

    def artemis_broadcast(self, r_fg, r_bg, m, t_fg, t_bg):
        fg_score = self.compute_score_broadcast_FG(r_fg, m, t_fg)
        bg_score = self.compute_score_broadcast_BG(r_bg, m, t_bg)
        return fg_score + bg_score

    ############################################################################
    # *** FOR HEATMAP PROCESSING
    ############################################################################

    def forward_save_intermediary(self, images_src, images_trg, sentences, lengths):

        # clean previously stored results, if any
        self.hold_results.clear()

        # compute embeddings & store activation map if in gradcam mode for heatmap visualizations
        r = self.get_image_embedding(images_src)
        if self.gradcam:
            self.hold_results["r_activation"] = self.img_enc.get_activation()
        t = self.get_image_embedding(images_trg)
        if self.gradcam:
            self.hold_results["t_activation"] = self.img_enc.get_activation()
        m = self.get_txt_embedding(sentences, lengths)

        return self.compute_score(r, m, t, store_intermediary=True)


    def get_image_embedding(self, images):
        return self.img_enc(Variable(images))


    def get_txt_embedding(self, sentences, lengths):
        return self.txt_enc(Variable(sentences), lengths)


    ############################################################################
    # *** SCORING METHODS
    ############################################################################

    # 2 versions of scoring methods:
    # - a "regular" version, which returns a tensor of shape (batch_size), where
    #   coefficient (i) is the score between query (i) and target (i). 
    # - a broadcast version, which returns a tensor of shape (batch_size,
    #   batch_size), corresponding to the score matrix where coefficient (i,j)
    #   is the score between query (i) and target (j).

    # Input:
    # - r: tensor of shape (batch_size, self.embed_dim), reference image embeddings
    # - m: tensor of shape (batch_size, self.embed_dim), modifier texts embeddings
    # - t: tensor of shape (batch_size, self.embed_dim), target image embeddings

    def compute_score(self, r_fg, r_bg, m, t_fg, t_bg):
        raise NotImplementedError

    def compute_score_broadcast(self, r, m, t):
        raise NotImplementedError

    ############################################################################
    # *** TRAINING & INFERENCE METHODS
    ############################################################################

    # Input:
    # - images_src, images_trg: tensors of shape (batch_size, 3, 256, 256)
    # - sentences: tensor of shape (batch_size, max_token, word_embedding)
    # - lengths: tensor (long) of shape (batch_size) containing the real size of
    #   the sentences (before padding)

    def forward(self, images_src, images_trg, sentences, lengths):
        """
        Returning a tensor of shape (batch_size), where coefficient (i) is the
        score between query (i) and target (i). 
        """
        r = self.get_image_embedding(images_src)
        r_fg, r_bg = self.concept_net(r)
        t = self.get_image_embedding(images_trg)
        t_fg, t_bg = self.concept_net(t)
        m = self.get_txt_embedding(sentences, lengths)
        return self.compute_score(r_fg, r_bg, m, t_fg, t_bg)

    def forward_broadcast(self, images_src, images_trg, sentences, lengths, noun_token_id, noun_lengths):
        """
        Returning a tensor of shape (batch_size, batch_size), corresponding to
        the score matrix where coefficient (i,j) is the score between query (i)
        and target (j).
        """
        r = self.get_image_embedding(images_src)
        m, word_feats = self.get_txt_embedding(sentences, lengths)
        t = self.get_image_embedding(images_trg)
        return self.compute_score_broadcast(r, m, t, lengths, noun_token_id, noun_lengths, word_feats)

    def get_compatibility_from_embeddings_one_query_multiple_targets(self, r_token, r_fg, r_bg, m, t_token, t_fg, t_bg, sentence_len, noun_token_id, noun_len, word_feats):
        """
        Input:
            - r: tensor of size (self.embed_dim), embedding of the query image.
            - m: tensor of size (self.embed_dim), embedding of the query text.
            - t: tensor of size (nb_imgs, self.embed_dim), embedding of the
              candidate target images.

        Returns a tensor of size (1, nb_imgs) with the compatibility scores of
        each candidate target from t with regard to the provided query (r,m).
        """
        return self.compute_score(r_token.unsqueeze(0), r_fg.view(1, -1), r_bg.view(1, -1), m.view(1, -1), t_token, t_fg, t_bg, sentence_len.unsqueeze(0), noun_token_id.unsqueeze(0), noun_len.unsqueeze(0), word_feats.unsqueeze(0))
