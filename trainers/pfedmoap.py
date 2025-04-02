import os.path as osp
import os
import copy
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from trainers.promptfl import TextEncoder, load_clip_to_cpu

import random

_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PFEDMOAP.N_CTX
        ctx_init = cfg.TRAINER.PFEDMOAP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PFEDMOAP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_class = len(classnames)
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # pFedMoAP specifics
        self.nonlocal_ctx = None # nonlocal prompt learner context
        self.nonlocal_text_features = None
        # self.num_experts = cfg.TRAINER.PFEDMOAP.NUM_EXPERTS
        self.lmbda = cfg.TRAINER.PFEDMOAP.LMBDA

        # initialize mixture of experts gating network
        num_heads = cfg.TRAINER.PFEDMOAP.GATING_HEADS
        gating_embed_dim = cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM
        reduce_times = self.image_encoder.output_dim // gating_embed_dim
        self.reduce_times = reduce_times
        
        self.gating = MultiheadAttention(gating_embed_dim, num_heads, dropout=0.1, scaling=cfg.TRAINER.PFEDMOAP.SCALING, dtype=self.dtype)
        
    def pool(self, t):
        if len(t.shape) == 4:
            return t[:, :, :, ::self.reduce_times]
        if len(t.shape) == 3:
            return t[:, :, ::self.reduce_times]
        if len(t.shape) == 2:
            return t[:, ::self.reduce_times]
        return None
    
    def _compute_nonlocal_text_features(self):
        if not self.nonlocal_ctx:
            return
        
        # store local state dict
        temp_local_state_dict = copy.deepcopy(self.prompt_learner.state_dict())
        self.nonlocal_text_features = []

        # if only one nonlocal context is provided, convert it to a list
        if not isinstance(self.nonlocal_ctx, list):
            self.nonlocal_ctx = [self.nonlocal_ctx]

        # iterate through different nonlocal contexts (global or other clients)
        for ctx in self.nonlocal_ctx:
            # load nonlocal ctx
            self.load_ctx(ctx)

            # compute nonlocal text features
            with torch.no_grad():
                text_features = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = self.pool(text_features)
                self.nonlocal_text_features.append(text_features.detach())
        
        # restore local state dict
        self.prompt_learner.load_state_dict(temp_local_state_dict)

    def load_ctx(self, ctx):
        temp_dict = self.prompt_learner.state_dict()
        temp_dict['ctx']= ctx
        self.prompt_learner.load_state_dict(temp_dict)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * image_features @ text_features.t()

        if self.nonlocal_text_features:
            q = self.pool(image_features).repeat(self.n_class, 1, 1) # (n_class, Batch, feature_dim)
            k = v = torch.stack([self.pool(text_features)] + self.nonlocal_text_features).permute(1, 0, 2) # (n_class, n_experts, feature_dim)
            new_features = self.gating(q, k, v)[0].permute(1, 2, 0) # (Batch, feature_dim, n_class)
            return self.lmbda * local_logits + logit_scale * torch.bmm(self.pool(image_features).unsqueeze(1), new_features).squeeze(1)
        
        else:
            return local_logits


class PFEDMOAP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PFEDMOAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print("==> PFEDMOAP model")
        cfg = self.cfg
        self.num_experts = cfg.TRAINER.PFEDMOAP.NUM_EXPERTS
        classnames = self.dm.dataset.classnames
        # print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PFEDMOAP.PREC == "fp32" or cfg.TRAINER.PFEDMOAP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # print(name,":",param.size())
            if ("prompt_learner" not in name) and ("gating" not in name):
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.lmbda = cfg.TRAINER.PFEDMOAP.LMBDA
        self.optim_p = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_p = build_lr_scheduler(self.optim_p, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_p, self.sched_p)

        self.optim_g = build_optimizer(self.model.gating, cfg.OPTIMGATING)
        self.sched_g = build_lr_scheduler(self.optim_g, cfg.OPTIMGATING)
        self.register_model("gating", self.model.gating, self.optim_g, self.sched_g)

        self.scaler = GradScaler() if cfg.TRAINER.PFEDMOAP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model, device_ids=[1])

        # for sparse selection
        self.shuffled_all_indices = list(range(cfg.DATASET.USERS))
        random.shuffle(self.shuffled_all_indices)
        self.random_selection_condition = lambda idx, current_idx, ctxs: idx != current_idx and ctxs[idx] != []
        self.reset_distance_cache()
    
    def reset_distance_cache(self, update_indices=None):
        if update_indices is None:
            self.distance_cache = {i: {j: None for j in range(self.cfg.DATASET.USERS)} for i in range(self.cfg.DATASET.USERS)}
        else:
            for idx in update_indices:
                self.distance_cache[idx] = {j: None for j in range(self.cfg.DATASET.USERS)}
                for i in range(self.cfg.DATASET.USERS):
                    self.distance_cache[i][idx] = None

    def download_nonlocal_ctx(self, nonlocal_ctx):
        self.model.nonlocal_ctx = nonlocal_ctx
        self.model._compute_nonlocal_text_features()

    def _get_dist_from_cache(self, idx, x):
        if x in self.distance_cache[idx]:
            return self.distance_cache[idx][x]
        elif idx in self.distance_cache[x]:
            return self.distance_cache[x][idx]
        return None

    def sparse_selection(self, idx, ctxs, method="random"):
        def random_selection(idx, ctxs):
            selected_indices = []
            for x in self.shuffled_all_indices:
                if self.random_selection_condition(x, idx, ctxs):
                    selected_indices.append(x)
                if len(selected_indices) == self.num_experts - 1: # exclude the current client
                    break
            return selected_indices
        
        if method == "random":
            return random_selection(idx, ctxs)
        

        if method == "nearest":
            if ctxs[idx] == []:
                return random_selection(idx, ctxs)
            trained_indices = [i for i in range(len(ctxs)) if ctxs[i] != []]
            if len(trained_indices) <= self.num_experts:
                return [i for i in trained_indices if i != idx]
                
            distances = []
            for a_trained_idx in trained_indices:
                if a_trained_idx == idx:
                    continue
                dist = self._get_dist_from_cache(idx, a_trained_idx)
                if dist is None:
                    dist = torch.norm(ctxs[idx] - ctxs[a_trained_idx])
                    self.distance_cache[idx][a_trained_idx] = dist
                    self.distance_cache[a_trained_idx][idx] = dist
                distances.append(dist)
            indices_for_smallest_dist = torch.topk(distances, self.num_experts-1, largest=False)[1]
            return [int(i.item()) for i in indices_for_smallest_dist]

            # raise NotImplementedError(f"Method: {method} has not been implemented yet")
        raise ValueError(f"Unknown sparse selection method for experts: {method}")

    
    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PFEDMOAP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim_p.zero_grad()
            self.optim_g.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim_p)
            self.scaler.step(self.optim_g)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(["prompt_learner"])

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, scaling=1.0, dtype=torch.float16):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.scaling = self.embed_dim ** -0.5
        self.scaling = scaling
        self.dtype = dtype
        
        self.W_q = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_k = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_v = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_o = nn.Linear(d_model, d_model, dtype=self.dtype)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, torch.mean(attn_probs, dim=1)