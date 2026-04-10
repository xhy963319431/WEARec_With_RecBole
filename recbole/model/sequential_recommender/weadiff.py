import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# 假设复用 DIFF 提供的基础 Layer 和 Loss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, VanillaAttention, FeedForward
from recbole.model.loss import BPRLoss

class WEA_DIFF_Layer(nn.Module):
    def __init__(self, config, num_heads=4,alpha=0.3, dropout=0.1, adaptive=True, combine_mode='gate'):
        super(WEA_DIFF_Layer, self).__init__()
        self.hidden_size = config['hidden_size']
        if self.hidden_size % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = config['num_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.seq_len = 50
        self.adaptive = adaptive
        self.combine_mode = combine_mode
        self.alpha = config['alpha']
        
        # Frequency bins for rFFT: (seq_len//2 + 1)
        self.freq_bins = self.seq_len // 2 + 1
        
        # Wavelet params
        self.complex_weight = nn.Parameter(torch.randn(1, self.num_heads, self.seq_len // 2, self.head_dim, dtype=torch.float32) * 0.02)
        
        # FFT base params
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(torch.full((self.num_heads, self.freq_bins, 1), -0.1))

        if adaptive:
            # 创新点：使用 Side Information 的维度来生成频率调制参数
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_heads * self.freq_bins * 2)
            )
        
        if self.combine_mode == 'concat':
            self.proj_concat = nn.Linear(2 * self.hidden_size, self.hidden_size)
        
            
        self.out_dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
    
    def wavelet_transform(self, x_heads):
        # Haar wavelet decomposition (保持原版 WEARec 逻辑)
        B, H, N, D = x_heads.shape
        N_even = N if (N % 2) == 0 else (N - 1)
        x_trunc = x_heads[:, :, :N_even, :] 

        x_even = x_trunc[:, :, 0::2, :] 
        x_odd  = x_trunc[:, :, 1::2, :] 

        approx = 0.5 * (x_even + x_odd) 
        detail = 0.5 * (x_even - x_odd)
        
        detail = detail * self.complex_weight

        x_even_recon = approx + detail
        x_odd_recon  = approx - detail

        out = torch.zeros_like(x_trunc)
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon

        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)
        return out

    def forward(self, item_tensor, attr_tensor):
        """
        item_tensor: [batch, seq_len, hidden] - 物品序列特征
        attr_tensor: [batch, seq_len, hidden] - 融合后的辅助属性特征 (Side Information)
        """
        batch, seq_len, hidden = item_tensor.shape
        x_heads = item_tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- (1) FFT-based global features ----
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')

        if self.adaptive:
            # 融合创新：使用 attr_tensor (辅助信息) 作为全局上下文，调制物品序列的频域滤波器！
            attr_context = attr_tensor.mean(dim=1)  # [B, hidden_size]
            adapt_params = self.adaptive_mlp(attr_context)
            adapt_params = adapt_params.view(batch, self.num_heads, self.freq_bins, 2)
            
            adaptive_scale = adapt_params[..., 0:1] # [B, H, freq_bins, 1]
            adaptive_bias  = adapt_params[..., 1:2]
        else:
            adaptive_scale = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=item_tensor.device)
            adaptive_bias  = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=item_tensor.device)

        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias   = self.base_bias + adaptive_bias

        F_fft_mod = F_fft * effective_filter + effective_bias
        x_fft = torch.fft.irfft(F_fft_mod, dim=2, n=seq_len, norm='ortho')

        # ---- (2) Wavelet-based local features ----
        # 小波变换保留捕捉物品序列的局部（短程）时序依赖
        x_wavelet = self.wavelet_transform(x_heads)

        # ---- (3) Combine local/global ----
        if self.combine_mode == 'gate':
            x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft
        else:
            x_wavelet_reshaped = x_wavelet.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
            x_fft_reshaped     = x_fft.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
            x_cat = torch.cat([x_wavelet_reshaped, x_fft_reshaped], dim=-1)
            x_combined = self.proj_concat(x_cat)
            x_combined = x_combined.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
        
        hidden_states = self.out_dropout(x_out)
        hidden_states = self.LayerNorm(hidden_states + item_tensor)
        
        return hidden_states


class WEA_DIFF_Encoder(nn.Module):
    def __init__(self, config):
        super(WEA_DIFF_Encoder, self).__init__()
        self.num_layers = config['n_layers']
        
        self.layers = nn.ModuleList([
            WEA_DIFF_Layer(config) for _ in range(self.num_layers)
        ])
        self.feed_forwards = nn.ModuleList([
            FeedForward(config['hidden_size'], config['inner_size'], config['hidden_dropout_prob'], config['hidden_act'], config['layer_norm_eps']) 
            for _ in range(self.num_layers)
        ])

    def forward(self, item_hidden_states, attr_hidden_states):
        for i in range(self.num_layers):
            # 将 item 和 attribute 共同传入 Layer
            layer_output = self.layers[i](item_hidden_states, attr_hidden_states)
            item_hidden_states = self.feed_forwards[i](layer_output)
        return item_hidden_states


class WEADIFF(SequentialRecommender):
    def __init__(self, config, dataset):
        super(WEADIFF, self).__init__(config, dataset)
        
        self.hidden_size = config['hidden_size']
        self.selected_features = config['selected_features']
        self.fusion_type = config['fusion_type']
        self.align =  config['align']
        self.lambda_ = config['lambda']
        self.initializer_range = 0.02
        # Embeddings
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # Side Information Embeddings (从 DIFF 继承)
        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset, config['attribute_hidden_size'][_], [self.selected_features[_]], config['pooling_mode'], config['device'])) 
             for _ in range(len(self.selected_features))]
        )
        
        if self.fusion_type == 'concat':
            self.attr_fusion_layer = nn.Linear(self.hidden_size * (len(self.selected_features) + 1), self.hidden_size)
        elif self.fusion_type == 'gate':
            self.attr_fusion_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.encoder = WEA_DIFF_Encoder(config)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm 的 bias 设为 0，weight 设为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # Linear 层的 bias 设为 0
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        
        # 获取 Position 和 Side Info Embeddings
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, _ = feature_embed_layer(None, item_seq)
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding['item'].squeeze(2)) # [B, L, D]
        
        # 构建统一的 Attribute Tensor
        if self.fusion_type == 'sum':
            attr_tensor = sum(feature_table) + position_embedding
        elif self.fusion_type == 'concat':
            attr_tensor = torch.cat(feature_table + [position_embedding], dim=-1)
            attr_tensor = self.attr_fusion_layer(attr_tensor)
        elif self.fusion_type == 'gate':
            # 将 feature 和 position 扩展到一致的维度并堆叠: [B, L, num_features+1, D]
            pos_emb_expanded = position_embedding.expand(item_seq.size(0), -1, -1)
            attr_stack = torch.stack(feature_table + [pos_emb_expanded], dim=2) 
            # 使用 Attention 计算各属性的权重并融合: [B, L, D]
            attr_tensor, _ = self.attr_fusion_layer(attr_stack) 
        else:
            # 兜底报错，防止拼写错误导致变量未定义
            raise ValueError(f"不支持的 fusion_type: {self.fusion_type}")
        
        input_emb = self.dropout(self.LayerNorm(item_emb))
        
        # 送入融合了时频域处理与属性调制的 Encoder
        seq_output = self.encoder(input_emb, attr_tensor)
        
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        return seq_output, feature_table

    def align_loss(self, X, A_list, item_seq, temperature=0.1):
        # 复用 DIFF 中的对比对齐损失，约束隐空间
        A = sum(A_list)
        X_norm = F.normalize(X, p=2, dim=-1)
        A_norm = F.normalize(A, p=2, dim=-1)

        similarity_matrix_X = torch.matmul(X_norm, A_norm.transpose(-1, -2)) / temperature
        similarity_matrix_A = torch.matmul(A_norm, X_norm.transpose(-1, -2)) / temperature

        Y_X = F.softmax(similarity_matrix_X, dim=-1)
        Y_A = F.softmax(similarity_matrix_A, dim=-1)

        epsilon = 1e-6  
        ground_truth = (torch.abs(torch.matmul(A_norm, A_norm.transpose(-1, -2)) - 1) < epsilon).float()
        
        seq_mask = (item_seq > 0).float()
        mask = seq_mask.unsqueeze(-1) @ seq_mask.unsqueeze(-1).transpose(-1, -2)
        
        loss_X = - (ground_truth * torch.log(Y_X + epsilon)) * mask
        loss_A = - (ground_truth * torch.log(Y_A + epsilon)) * mask

        loss_X = loss_X.sum(dim=-1) / (seq_mask.sum(-1).unsqueeze(-1) + epsilon)
        loss_A = loss_A.sum(dim=-1) / (seq_mask.sum(-1).unsqueeze(-1) + epsilon)

        return (loss_X + loss_A).mean()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output, feature_table = self.forward(item_seq, item_seq_len)
        
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            
        if self.align:
            # 加入辅助信息的表征对齐约束
            item_seq_emb = self.item_embedding(item_seq)
            loss += self.lambda_ * self.align_loss(item_seq_emb, feature_table, item_seq)
       
        return loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,feature_emb = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, feature_emb = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores