import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, tokenize
from .triplet_loss import TripletLoss


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class MLPMapping(nn.Module):
    """
    3-layer FC mapping like SCGI fMg / fMl.
    """
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=3, dropout=0.1):
        super().__init__()
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            layers.append(nn.Sequential(
                nn.Linear(dim, middle_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            ))
            dim = middle_dim
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(middle_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return self.fc_out(x)


class TextEncoder(nn.Module):
    """
    Use the CLIP text stack already inside base_model.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x])
        x = outputs[0].permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_feature = x[
            torch.arange(x.shape[0], device=x.device),
            tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection
        return text_feature


class PromptLearner(nn.Module):
    """
    Prompt: 'A photo of a X person'
    """
    def __init__(self, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X person"
        n_ctx = 4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_prompts = tokenize(ctx_init).to(device)
        token_embedding = token_embedding.to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 2:, :])
        self.dtype = dtype

    def forward(self, bias: torch.Tensor):
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)
        prompts = torch.cat([prefix, bias, suffix], dim=1)
        return prompts


class LPNC(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.current_task = ["supid"]

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]

        # freeze CLIP text side so caption/prompt spaces stay stable
        for p in self.base_model.transformer.parameters():
            p.requires_grad = False
        self.base_model.positional_embedding.requires_grad = False
        self.base_model.text_projection.requires_grad = False
        for p in self.base_model.token_embedding.parameters():
            p.requires_grad = False
        for p in self.base_model.ln_final.parameters():
            p.requires_grad = False

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        # CFF head
        self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim, self.embed_dim // 64, batch_first=True
        )
        self.cross_modal_transformer = Transformer(
            width=self.embed_dim,
            layers=args.cmt_depth,
            heads=self.embed_dim // 64
        )

        scale = self.cross_modal_transformer.width ** -0.5
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # shared/frozen text encoder view on base_model text stack
        self.text_encoder = TextEncoder(self.base_model)

        # CGI mappings
        self.global_mapping = MLPMapping(
            embed_dim=self.embed_dim, middle_dim=self.embed_dim,
            output_dim=self.embed_dim, n_layer=3
        )
        self.local_mapping = MLPMapping(
            embed_dim=self.embed_dim, middle_dim=self.embed_dim,
            output_dim=self.embed_dim, n_layer=3
        )

        # local CGI branch
        self.local_num_queries = 2
        self.local_queries = nn.Parameter(
            torch.randn(self.local_num_queries, self.embed_dim) * 0.02
        )
        self.ln_q_local = LayerNorm(self.embed_dim)
        self.ln_kv_local = LayerNorm(self.embed_dim)
        self.cross_attn_local = nn.MultiheadAttention(
            self.embed_dim, self.embed_dim // 64, batch_first=True
        )
        nn.init.normal_(self.cross_attn_local.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn_local.out_proj.weight, std=proj_std)

        self.ln_ffn_local = LayerNorm(self.embed_dim)
        self.ffn_local = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(0.1),
        )

        self.prompt_learner = PromptLearner(self.base_model.dtype, self.base_model.token_embedding)

        # simplified prompt for inference
        fixed_prompt = "A photo of a person"
        self.register_buffer("fixed_prompt_tokens", tokenize(fixed_prompt))

    def _eot_text_feature(self, text_tokens: torch.Tensor):
        x, _ = self.base_model.encode_text(text_tokens.long())
        return x[
            torch.arange(x.shape[0], device=x.device),
            text_tokens.argmax(dim=-1)
        ].float()

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0).permute(1, 0, 2)  # LND -> NLD
        return self.ln_post(x)

    def cross_former_local(self, patch_feats: torch.Tensor) -> torch.Tensor:
        B = patch_feats.shape[0]
        q = self.local_queries.unsqueeze(0).expand(B, -1, -1)
        qn = self.ln_q_local(q)
        kv = self.ln_kv_local(patch_feats)

        qc = q + self.cross_attn_local(qn, kv, kv, need_weights=False)[0]
        p = qc + self.ffn_local(self.ln_ffn_local(qc))
        avg_p = p.mean(dim=1)
        return avg_p

    def _cgi_prompt_feature(self, image_feats: torch.Tensor, caption_tokens: torch.Tensor):
        # raw caption embedding
        text_feats, _ = self.base_model.encode_text(caption_tokens.long())
        t_feats = text_feats[
            torch.arange(text_feats.shape[0], device=text_feats.device),
            caption_tokens.argmax(dim=-1)
        ].float()

        # SCGI Eq.(3): v_tilde = v ⊙ t
        refined_image_feats = image_feats * t_feats.unsqueeze(1)

        global_token = self.global_mapping(refined_image_feats[:, 0, :].float())
        local_token = self.local_mapping(self.cross_former_local(refined_image_feats[:, 1:, :].float()))
        pseudo_token = global_token + local_token

        prompts = self.prompt_learner(pseudo_token)
        prompt_feat = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts).float()

        return prompt_feat, t_feats, refined_image_feats

    def _pair_contrastive(self, image_feat: torch.Tensor, prompt_feat: torch.Tensor):
        image_feat = F.normalize(image_feat, p=2, dim=1)
        prompt_feat = F.normalize(prompt_feat, p=2, dim=1)
        logits = self.logit_scale * (prompt_feat @ image_feat.t())
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_t2i + loss_i2t)

    def encode_text(self, text):
        return self._eot_text_feature(text)

    def encode_image(self, image):
        # raw BGE, kept only for diagnosis
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_image1(self, image):
        x, _ = self.base_model.encode_image(image)
        return x.float()

    def encode_image_cff(self, image):
        """
        SCGI-style inference: remove caption-guided branch and use a fixed prompt.
        Return the last layer of CFF as gallery embedding.
        """
        image_feats, _ = self.base_model.encode_image(image)
        B = image_feats.shape[0]
        fixed_tokens = self.fixed_prompt_tokens.expand(B, -1).to(image.device)

        with torch.no_grad():
            fixed_prompt_feat = self._eot_text_feature(fixed_tokens)

        with autocast():
            cross_x = self.cross_former(
                fixed_prompt_feat.unsqueeze(1).type(image_feats.dtype),
                image_feats,
                image_feats
            )
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))

        return cross_x_bn.float()

    def forward(self, batch):
        ret = {}

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        pids = batch["pids"]

        triplet = TripletLoss(
            margin=getattr(self.args, "triplet_margin", 0.3),
            hard_factor=0.0
        )

        image_feats, _ = self.base_model.encode_image(images)
        i_feats = image_feats[:, 0, :].float()

        with autocast():
            prompt_feat, _, _ = self._cgi_prompt_feature(image_feats, caption_ids)

            # CFF: prompt as query, image tokens as key/value
            cross_x = self.cross_former(prompt_feat.unsqueeze(1).type(image_feats.dtype), image_feats, image_feats)
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))
            cls_score = self.classifier_proj(cross_x_bn.half()).float()

        con_loss = self._pair_contrastive(i_feats, prompt_feat)
        id_loss = objectives.compute_id(cls_score, pids)
        tri_loss, _, _ = triplet(cross_x_bn.float(), pids)

        ret["supid_loss"] = tri_loss + id_loss + self.args.lambda1_weight * con_loss
        ret["temperature"] = 1 / self.logit_scale
        return ret


def build_model(args, num_classes=11003):
    model = LPNC(args, num_classes)
    convert_weights(model)
    return model