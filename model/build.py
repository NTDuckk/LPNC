import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, tokenize
import torch
import torch.nn as nn
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss
from torch.cuda.amp import autocast


def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class TextEncoder(nn.Module):
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
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        text_feature = x[
            torch.arange(x.shape[0], device=x.device),
            tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection
        return text_feature


class LPNC(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]
        self.tse_embed_dim = 1024

        # keep these modules to minimize structural changes, but they are no longer
        # used in training/evaluation after removing cid/cotrl paths
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

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
            heads=self.embed_dim // 64,
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

        self.text_encoder = TextEncoder(copy.deepcopy(self.base_model))
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.img2text = IM2TEXT(
            embed_dim=512,
            middle_dim=512,
            output_dim=512,
            n_layer=2,
        )

        dataset_name = args.dataset_name
        self.W = nn.Parameter(torch.eye(512))
        self.prompt_learner = PromptLearner(
            num_classes,
            dataset_name,
            self.base_model.dtype,
            self.base_model.token_embedding,
        )

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def _set_task(self):
        # force only PromptSG-like supid branch
        self.current_task = ["supid"]
        print(f"Training Model with {self.current_task} tasks")

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_image1(self, image):
        x, _ = self.base_model.encode_image(image)
        return x.float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[
            torch.arange(x.shape[0], device=x.device),
            text.argmax(dim=-1)
        ].float()

    def encode_image_tse(self, image):
        x, atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)
        return i_tse_f.float()

    def encode_text_tse(self, text):
        x, atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def forward(self, batch):
        ret = dict()

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        pids = batch["pids"]

        triplet = TripletLoss(
            margin=getattr(self.args, "triplet_margin", 0.3),
            hard_factor=0.0,
        )
        supcon = SupConLoss(images.device)

        ret.update({"temperature": 1 / self.logit_scale})

        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[
            torch.arange(text_feats.shape[0], device=text_feats.device),
            caption_ids.argmax(dim=-1)
        ].float()

        # keep caption feature in the prompt composition exactly as you requested
        token_features = self.img2text(i_feats.half())

        with autocast():
            prompts = self.prompt_learner(token_features + t_feats @ self.W)
            text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

            cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))
            cls_score = self.classifier_proj(cross_x_bn.half()).float()

        supcon_loss = (
            supcon(i_feats, text_feature.float(), pids, pids)
            + supcon(text_feature.float(), i_feats, pids, pids)
        )
        id_loss = objectives.compute_id(cls_score, pids)

        # PromptSG-like placement: triplet on the interaction output feature
        triplet_loss, _, _ = triplet(cross_x_bn.float(), pids)

        ret.update({
            'supid_loss': triplet_loss + id_loss + self.args.lambda1_weight * supcon_loss
        })

        return ret


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X person"
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_prompts = tokenize(ctx_init).to(device)
        token_embedding = token_embedding.to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("token_prefix", embedding[:, : n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 2 :, :])
        self.num_class = num_class
        self.token_ = token_embedding
        self.dtype = dtype

    def forward(self, bias):
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)
        prompts = torch.cat([prefix, bias, suffix], dim=1)
        return prompts


def build_model(args, num_classes=11003):
    model = LPNC(args, num_classes)
    convert_weights(model)
    return model