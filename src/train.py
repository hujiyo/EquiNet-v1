'''
训练脚本

评分制度（收益率制度，以代码实现为准）：
采用排序能力评估，更贴近真实选股场景。
按预测概率从高到低排序，统计Top-K%样本的收益：
每个区间统计：样本数、平均收益、累计收益、上涨准确率、非负率
'''

import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
import random
import math
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import (ModelConfig, TrainingConfig, DataConfig,
                   DeviceConfig, ModelSaveConfig,
                   print_config_summary)

# 学习率预热调度器
class WarmupScheduler:
    """
    学习率预热调度器
    在前几轮训练中，学习率从很小的值逐步增加到目标学习率
    这有助于模型在训练初期更稳定地收敛
    """
    def __init__(self, optimizer, warmup_epochs, target_lr, start_lr=None):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热轮数
            target_lr: 目标学习率（预热结束后的学习率）
            start_lr: 预热起始学习率，如果为None则使用target_lr的1/100
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.start_lr = start_lr if start_lr is not None else target_lr / 100
        self.current_epoch = 0
        
        # 设置初始学习率为预热起始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr
    
    def step(self, epoch=None):
        """
        更新学习率
        Args:
            epoch: 当前轮数，如果为None则使用内部计数器
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.start_lr + (self.target_lr - self.start_lr) * ((self.current_epoch + 1) / self.warmup_epochs)
        else:
            # 预热结束后保持目标学习率
            lr = self.target_lr
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率（兼容PyTorch调度器接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def is_warmup_phase(self):
        """判断是否还在预热阶段"""
        return self.current_epoch < self.warmup_epochs

# 动态加权BCE损失函数实现
class DynamicWeightedBCE(nn.Module):
    """
    动态加权BCE损失函数：按标签桶分配权重
    - 标签1.0固定权重4.0
    - 标签0.6/0.3/0.0按样本数量动态分配权重（样本少=权重高）
    """
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(DynamicWeightedBCE, self).__init__()
        self.reduction = reduction
        
        # 固定正样本权重
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        
        # 动态负样本权重（按标签桶分配）
        self.register_buffer('weight_0_6', torch.tensor(1.0))
        self.register_buffer('weight_0_3', torch.tensor(1.0))
        self.register_buffer('weight_0_0', torch.tensor(1.0))
        
    def update_weights(self, targets):
        """
        二分类动态权重：根据正负样本比例动态调整
        targets: [batch_size] 标签 (1.0/0.0)
        """
        if isinstance(targets, torch.Tensor):
            # BF16需要先转为FP32再转numpy
            targets = targets.float().cpu().numpy()
        
        # 统计正负样本数量
        count_positive = np.sum(targets >= 0.5)  # 上涨样本（≥5%）
        count_negative = np.sum(targets < 0.5)   # 不上涨样本（<5%）
        
        if count_positive > 0 and count_negative > 0:
            # 动态调整负样本权重，保持正负样本对总损失的贡献平衡
            # neg_weight = pos_weight * (正样本数 / 负样本数)
            neg_weight = float(self.pos_weight) * (count_positive / count_negative)
            
            # 更新负样本权重（复用weight_0_0变量）
            self.weight_0_0 = torch.tensor(neg_weight)
        elif count_positive == 0:
            # 没有正样本，负样本权重设为正样本权重
            self.weight_0_0 = torch.tensor(float(self.pos_weight))
        else:
            # 没有负样本，权重设为较小值
            self.weight_0_0 = torch.tensor(0.1)
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, 1] 模型输出的logits (BF16)
        targets: [batch_size] 真实标签 (1.0/0.0) (BF16)
        """
        # 确保输入形状正确
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        
        inputs = inputs.squeeze()
        
        # 计算BCE loss（带logits）
        # sigmoid(x) 的数值稳定计算
        max_val = torch.clamp(inputs, min=0)
        loss = inputs - inputs * targets + max_val + torch.log(torch.exp(-max_val) + torch.exp(-inputs - max_val))
        
        # 二分类动态权重：正样本和负样本分别使用动态权重
        pos_weight = self.pos_weight.to(dtype=inputs.dtype, device=inputs.device)
        neg_weight = self.weight_0_0.to(dtype=inputs.dtype, device=inputs.device)
        
        # 根据标签分配权重：正样本用pos_weight，负样本用动态neg_weight
        weights = torch.where(targets >= 0.5, pos_weight, neg_weight)
        loss = loss * weights
        
        # 🔥 新增：对预测偏差较大的样本进行指数级额外惩罚
        # 计算预测概率值
        predictions = torch.sigmoid(inputs)  # 将logits转为概率 [0, 1]
        
        # 计算预测值与真实标签之间的绝对差值
        prediction_error = torch.abs(predictions - targets)
        
        # 当差值 >= 0.15时，应用指数级惩罚（阈值从0.2降低到0.15）
        # 使用 3^(1.5×差值) 作为额外惩罚因子（底数从2提升到3，指数放大1.5倍）
        # 例如：差值0.2 -> 3^0.3 ≈ 1.39 (温和惩罚)
        #       差值0.5 -> 3^0.75 ≈ 2.28 (中等惩罚)
        #       差值0.8 -> 3^1.2 ≈ 3.74 (强惩罚)
        #       差值1.0 -> 3^1.5 ≈ 5.20 (损失放大5倍！)
        penalty_multiplier = torch.where(
            prediction_error >= 0.15,
            torch.pow(3.0, prediction_error * 1.5),   # 指数级惩罚：3^(1.5×差值)
            torch.ones_like(prediction_error)         # 差值<0.15时，惩罚因子为1（不额外惩罚）
        )
        
        # 应用额外惩罚
        loss = loss * penalty_multiplier
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class RMSNorm(nn.Module):
    """
    RMSNorm: 只做缩放，不减均值
    相比LayerNorm，保留了特征间的相对大小关系
    这对于OHLC价格特征很重要，因为 High > Close > Open > Low 的关系需要保持
    """
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # 计算RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 只做缩放，不减均值
        return x / rms * self.scale

class PositionalEncoding(nn.Module):
    """
    标准的正弦位置编码
    让 Transformer 自己学习时间依赖关系，不加人为规则
    """
    def __init__(self, d_model, max_seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()
        
        # 创建标准的正弦/余弦位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 直接添加位置编码，不使用LayerNorm（会在后续层中使用Pre-Norm）
        seq_len = x.size(1)
        pe_slice = self.pe[:seq_len, :].unsqueeze(0)
        return x + pe_slice

class MultiHeadAttention(nn.Module):
    """
    标准的多头注意力机制（Pre-Norm架构）
    让模型自动学习每个头应该关注什么特征，不人为干预
    """
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        assert d_model % nhead == 0
        
        # 使用标准的MultiheadAttention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Pre-Norm: 在注意力之前进行归一化（使用RMSNorm保留特征相对关系）
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)
        
    def forward(self, x, attn_mask=None):
        # Pre-Norm架构：先归一化，再计算注意力，最后残差连接
        # 输出 = 输入 + Dropout(Attention(LayerNorm(输入)))
        
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)

        # Pre-Norm: 先对输入进行归一化
        normalized_x = self.norm(x)
        
        # 计算注意力
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x, attn_mask=mask)
        
        # 残差连接（注意这里是加到原始输入x上，而不是normalized_x）
        output = x + self.dropout(attn_output)
        return output

class TransformerLayer(nn.Module):
    """
    标准的 Transformer 层（Pre-Norm架构）
    设计理念：让模型自动学习应该关注什么特征，不加人为干预
    Pre-Norm相比Post-Norm有更好的训练稳定性
    """
    def __init__(self, d_model, nhead, use_ffn=True):
        super(TransformerLayer, self).__init__()
        
        self.use_ffn = use_ffn
        
        # 使用Pre-Norm多头注意力
        self.attention = MultiHeadAttention(d_model, nhead)
        
        if self.use_ffn:
            # 前馈网络，用于进一步处理注意力的输出
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # 先扩展维度
                nn.ReLU(),                        # 激活函数
                nn.Dropout(ModelConfig.DROPOUT_RATE),  # 防过拟合
                nn.Linear(d_model * 4, d_model),  # 再压缩回原维度
            )
            
            # Pre-Norm: 在前馈网络之前进行归一化（使用RMSNorm保留特征相对关系）
            self.norm = RMSNorm(d_model)
            self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        
    def forward(self, x):
        # x的shape: [batch_size, seq_len, d_model]
        
        # Pre-Norm架构的注意力子层（MultiHeadAttention内部已经实现了Pre-Norm）
        # 输出 = 输入 + Dropout(Attention(LayerNorm(输入)))
        x = self.attention(x, attn_mask=None)
        
        if self.use_ffn:
            # Pre-Norm架构的前馈网络子层
            # 输出 = 输入 + Dropout(FFN(LayerNorm(输入)))
            normalized_x = self.norm(x)
            ff_out = self.feed_forward(normalized_x)
            x = x + self.dropout(ff_out)
        
        return x

class EnhancedStockTransformer(nn.Module):
    """
    改进的 Transformer 模型（Pre-Norm架构 + 分离Embedding + 渐进式FFN）
    
    核心改进1：分离Embedding - 避免LayerNorm时互相干扰
    - 价格特征(OHLC 4维) -> Embedding -> 48维 (占75%，主导地位)
    - 成交量特征(Volume 1维) -> Embedding -> 16维 (占25%，辅助信息)
    - 拼接后得到64维向量，送入Transformer
    
    核心改进2：渐进式FFN - 分层学习策略
    - Layer 1: 只用Attention（专注学习时序依赖和特征关系）
    - Layer 2-5: Attention + FFN（增加非线性变换能力）
    - 好处：第1层纯粹学习模式，后续层增强表达能力
    
    总体优势：
    1. 避免成交量的大值主导LayerNorm，扭曲价格信号
    2. 保持价格特征之间的相对关系
    3. 价格特征有更大的表达空间（48维 vs 16维，3:1比例）
    4. 渐进式学习：第1层纯学模式，后续层增强表达
    5. 参数量减少约13%（第1层省掉FFN），略微降低过拟合风险
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, max_seq_len):
        super(EnhancedStockTransformer, self).__init__()
        
        # 分离Embedding：价格和成交量独立处理
        self.price_embedding = nn.Linear(ModelConfig.PRICE_DIM, ModelConfig.PRICE_EMBED_DIM)
        self.volume_embedding = nn.Linear(ModelConfig.VOLUME_DIM, ModelConfig.VOLUME_EMBED_DIM)
        
        # 使用标准位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 第1层只用Attention（专注学习序列模式）
        # 第2-5层用Attention+FFN（增加非线性变换能力）
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, use_ffn=False) if i == 0 
            else TransformerLayer(d_model, nhead, use_ffn=True)
            for i in range(num_layers)
        ])
        
        # Pre-Norm架构：在最后添加一个RMSNorm
        # 因为Pre-Norm的最后一层没有归一化输出
        self.final_norm = RMSNorm(d_model)
        
        # 简化输出层，减少过拟合
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 降维
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(d_model // 2, output_dim)  # 最终输出
        )
        
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        
    def forward(self, x):
        # x: [batch_size, seq_len, 5] (OHLCV)
        
        # 1. 分离Embedding：价格和成交量独立处理
        prices = x[:, :, :4]   # [batch_size, seq_len, 4] OHLC
        volumes = x[:, :, 4:5] # [batch_size, seq_len, 1] Volume
        
        price_emb = self.price_embedding(prices)      # [batch_size, seq_len, 48]
        volume_emb = self.volume_embedding(volumes)   # [batch_size, seq_len, 16]
        
        # 2. 拼接成64维（而不是相加！）
        # 这样价格和成交量各占据独立的子空间，LayerNorm时干扰最小
        # 价格占48维(75%)，成交量占16维(25%)，价格主导
        x = torch.cat([price_emb, volume_emb], dim=-1)  # [batch_size, seq_len, 64]
        
        # 3. 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 4. Transformer层（Pre-Norm架构）
        for layer in self.layers:
            x = layer(x)
        
        # 5. Pre-Norm架构需要在最后进行归一化
        #    因为每层的输出没有经过归一化
        x = self.final_norm(x)
        
        # 6. 取最后时间步 + 输出投影
        last_hidden = x[:, -1, :]
        output = self.output_projection(last_hidden)
        
        return output

# 单个文件处理函数（用于多进程）
def process_single_file(args):
    """
    处理单个文件，返回原始数据（不做全局标准化，避免数据泄露）
    按时间划分训练集和测试集：最近80天作为测试集，其余作为训练集
    """
    file_path, file_name, test_days = args
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        # 使用OHLCV（5维特征）
        data = df[['start', 'max', 'min', 'end', 'volume']].values
        
        data_length = len(data)
        
        # 按时间划分：最近test_days天作为测试集
        if data_length > test_days:
            train_split_point = data_length - test_days
            train_data = data[:train_split_point]  # 历史数据作为训练集
            test_data = data  # 保留全部数据用于测试集（需要前面历史数据作为上下文）
        else:
            # 数据不足，只能用作训练
            train_data = data
            test_data = None
        
        stock_info = {
            'file_name': file_name,
            'data_length': data_length,
            'train_data': train_data,
            'test_data': test_data,
            'train_length': len(train_data) if train_data is not None else 0,
            'test_split_point': data_length - test_days if data_length > test_days else data_length
        }
        
        return stock_info
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return None

# 数据预处理函数（按时间划分训练集和测试集）
def load_and_preprocess_data(data_dir=DataConfig.DATA_DIR, test_days=DataConfig.TEST_DAYS):
    """
    数据加载和预处理，使用多进程并行加载
    按时间划分：每只股票的最近test_days天作为测试集，其余作为训练集
    """
    from multiprocessing import Pool, cpu_count
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    all_files.sort()
    
    print(f"总共 {len(all_files)} 只股票")
    print(f"划分策略: 每只股票的最近 {test_days} 天作为测试集，其余作为训练集")
    
    # 处理所有文件
    file_args = [(os.path.join(data_dir, f), f, test_days) for f in all_files]
    num_workers = min(cpu_count(), 8)
    
    with Pool(num_workers) as pool:
        all_stock_info = [r for r in pool.map(process_single_file, file_args) if r is not None]
    
    # 分离训练和测试数据
    train_stock_info = []
    test_stock_info = []
    
    for stock_info in all_stock_info:
        # 所有股票的历史数据都用于训练
        if stock_info['train_data'] is not None and len(stock_info['train_data']) >= DataConfig.REQUIRED_LENGTH:
            train_stock_info.append({
                'file_name': stock_info['file_name'],
                'data': stock_info['train_data'],
                'data_length': stock_info['train_length']
            })
        
        # 有足够数据的股票用于测试
        if stock_info['test_data'] is not None and len(stock_info['test_data']) >= DataConfig.REQUIRED_LENGTH:
            test_stock_info.append({
                'file_name': stock_info['file_name'],
                'data': stock_info['test_data'],
                'data_length': len(stock_info['test_data']),
                'test_split_point': stock_info['test_split_point']  # 测试集起始位置
            })
    
    print(f"训练集: {len(train_stock_info)} 只股票的历史数据")
    print(f"测试集: {len(test_stock_info)} 只股票的最近 {test_days} 天数据")
    
    return train_stock_info, test_stock_info

# 计算股票选择权重
def calculate_stock_weights(stock_info_list):
    """
    计算每只股票的采样权重
    数据量越大的股票权重越大，但最大不超过平均值的1.5倍
    """
    data_lengths = [info['data_length'] for info in stock_info_list]
    avg_length = np.mean(data_lengths)
    
    # 计算权重：数据长度 / 平均长度，但限制在1.0到1.5之间
    weights = []
    for length in data_lengths:
        weight = length / avg_length
        weight = max(1.0, min(1.5, weight))  # 限制在1.0到1.5之间
        weights.append(weight)
    
    # 归一化权重，使其总和为1.0（np.random.choice要求）
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    return normalized_weights

# 改进的样本生成函数（使用滚动窗口标准化，避免数据泄露）
def generate_single_sample_improved(stock_info_list, stock_weights):
    """
    改进的样本生成函数（修复数据泄露问题）
    1. 根据数据量大小选择股票（数据量大的概率更高）
    2. 随机选择训练集时间范围内的时间窗口
    3. 使用滚动窗口标准化：只使用当前样本的历史数据进行标准化
    """
    for _ in range(100):  # 最多尝试100次生成有效样本
        # 第一步：根据权重选择股票
        stock_index = np.random.choice(len(stock_info_list), p=stock_weights)
        stock_info = stock_info_list[stock_index]
        stock_data = stock_info['data']  # 训练集数据（已经按时间划分）
        
        context_length = DataConfig.CONTEXT_LENGTH
        required_length = DataConfig.REQUIRED_LENGTH
        
        if len(stock_data) < required_length:
            continue
            
        # 第二步：在训练集范围内随机选择起始位置
        # 注意：start_index 必须 > 0，因为需要前一天的数据来计算涨跌幅
        max_start_index = len(stock_data) - required_length
        if max_start_index < 1:
            continue  # 数据不足，至少需要 required_length + 1 天
        start_index = np.random.randint(1, max_start_index + 1)
        
        # 提取原始数据窗口
        input_seq_raw = stock_data[start_index:start_index + context_length]
        
        # 🔑 特征标准化：计算每天相对前一天的涨跌幅
        # 这是最符合人类交易思维的方式：今天相比昨天涨了多少
        
        # 初始化标准化后的数据
        input_seq = np.zeros_like(input_seq_raw, dtype=np.float64)
        
        # 获取窗口前一天的数据作为第1天的基准
        prev_day_data = stock_data[start_index - 1]
        prev_prices = prev_day_data[:4]  # OHLC
        prev_volume = prev_day_data[4]   # Volume
        
        # 避免除零错误
        if np.any(prev_prices == 0) or prev_volume == 0:
            continue
        
        # 第1天：相对于窗口前一天的收盘价（价格特征）
        prev_close = prev_prices[3]  # 前一天的收盘价
        if prev_close == 0:
            continue
        input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
        # 成交量特征：直接使用相对变化比例
        input_seq[0, 4] = (input_seq_raw[0, 4] - prev_volume) / prev_volume
        
        # 第2-40天：相对于前一天的收盘价
        for i in range(1, context_length):
            # 价格特征：所有价格(OHLC)都相对于前一天的收盘价
            # 这符合真实交易逻辑：今天的开盘/最高/最低/收盘都和昨天收盘价比
            yesterday_close = input_seq_raw[i-1, 3]  # 前一天的收盘价
            yesterday_volume = input_seq_raw[i-1, 4]  # 前一天的成交量
            if yesterday_close == 0 or yesterday_volume == 0:
                # 如果昨天收盘价或成交量为0，跳过这个样本
                break
            input_seq[i, :4] = (input_seq_raw[i, :4] - yesterday_close) / yesterday_close
            # 成交量特征：直接使用相对变化比例
            input_seq[i, 4] = (input_seq_raw[i, 4] - yesterday_volume) / yesterday_volume
        else:
            # 只有for循环正常结束（没有break）才会执行这里
            # 这表示所有历史数据都成功标准化了
            
            # 统一使用旧的涨幅型标签：基于未来涨幅大小
            original_start_price = stock_data[start_index + context_length - 1, 3]  # 当前收盘价
            original_end_price = stock_data[start_index + DataConfig.REQUIRED_LENGTH - 1, 3]   # N天后收盘价

            if original_start_price == 0:  # 避免除零错误
                continue

            cumulative_return = (original_end_price - original_start_price) / original_start_price

            # 软标签机制：降低边界区域的惩罚
            # - 收益 ≥ 8% → 1.0（明确上涨）
            # - 收益 0-8% → 0.4（边界区域，降低矛盾惩罚）
            # - 收益 < 0% → 0.0（明确不涨）
            if cumulative_return >= DataConfig.UPRISE_THRESHOLD:  # 涨幅≥阈值
                target = 1.0
            elif cumulative_return >= 0:  # 0-8%之间
                target = 0.4
            else:  # 涨幅<0%
                target = 0.0

            return input_seq, target
    
    raise ValueError("无法生成有效样本：股票数据长度不足或收盘价为0")

def generate_batch_samples_improved(stock_info_list, stock_weights, batch_size):
    """
    改进的批量生成训练样本（使用滚动窗口标准化）
    返回: (batch_inputs, batch_targets)
    batch_inputs: numpy array, shape [batch_size, context_length, 5]  
    batch_targets: numpy array, shape [batch_size]
    """
    batch_inputs = []
    batch_targets = []
    
    attempts = 0
    max_attempts = batch_size * 10  # 防止无限循环
    
    while len(batch_inputs) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            input_seq, target = generate_single_sample_improved(stock_info_list, stock_weights)
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        except ValueError:
            continue
    
    if len(batch_inputs) < batch_size:
        raise ValueError(f"无法生成足够的样本，只生成了 {len(batch_inputs)}/{batch_size} 个")
    
    return np.array(batch_inputs), np.array(batch_targets)

# 创建固定的评估数据集（使用滚动窗口标准化，只使用测试集时间范围）
def create_fixed_evaluation_dataset(test_stock_info, seed=DataConfig.RANDOM_SEED):
    """
    创建固定的评估数据集，使用滚动窗口标准化避免数据泄露
    只使用测试集的时间范围（最近80天），严格时间分离
    使用全部测试样本进行评估，确保评估结果更加准确和稳定
    """
    print("正在创建固定的评估数据集（使用滚动窗口标准化，严格时间分离）...")
    # 设置所有可能的随机种子以确保完全可重复
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []  # 存储实际涨跌幅
    
    # 预先生成所有可能的样本
    all_possible_samples = []
    context_length = DataConfig.CONTEXT_LENGTH
    required_length = DataConfig.REQUIRED_LENGTH
    test_days = DataConfig.TEST_DAYS
    
    for stock_idx, stock_info in enumerate(test_stock_info):
        stock_data = stock_info['data']  # 原始数据（包含全部历史）
        test_split_point = stock_info['test_split_point']  # 测试集起始位置
        
        if len(stock_data) < required_length:
            continue
        
        # 🔑 关键：严格的测试集划分，避免数据泄露
        # 测试集80天：[test_split_point, len(stock_data))
        # 每只股票可生成的测试样本数 = 测试集天数 - 序列长度 - 预测天数 = 80 - 40 - 3 = 37个
        
        # 最早预测时间点：测试集第41天（前40天作为上下文）
        # 最晚预测时间点：测试集倒数第4天（需要预留3天未来数据）
        min_predict_point = test_split_point + context_length
        max_predict_point = len(stock_data) - DataConfig.FUTURE_DAYS - 1
        
        # 检查是否有足够的数据
        if min_predict_point > max_predict_point:
            continue  # 测试集不够80天，无法生成样本
        
        # 将预测时间点转换为start_idx
        # 预测时间点 = start_idx + context_length - 1
        # start_idx = 预测时间点 - context_length + 1
        min_start_idx = min_predict_point - context_length + 1
        max_start_idx = max_predict_point - context_length + 1
        
        # 为每只股票生成测试样本
        # 每个样本的预测时间点在测试集时间范围内
        # 上下文使用的是测试集前部分天数+训练集数据（如有需要）
        for start_idx in range(min_start_idx, max_start_idx + 1):
            
            # 提取原始数据窗口（可能包含部分训练集数据作为上下文）
            input_seq_raw = stock_data[start_idx:start_idx + context_length]
            
            # 🔑 特征标准化：计算每天相对前一天的涨跌幅
            # 初始化标准化后的数据
            input_seq = np.zeros_like(input_seq_raw, dtype=np.float64)
            
            # 第1天：使用窗口前一天的数据作为基准
            # 注意：理论上start_idx不可能为0（因为测试集前面有训练集数据）
            # 但为了代码健壮性，仍然检查
            if start_idx == 0:
                continue  # 跳过，因为没有前一天数据
            
            prev_day_data = stock_data[start_idx - 1]
            prev_prices = prev_day_data[:4]  # OHLC
            prev_volume = prev_day_data[4]   # Volume
            
            # 避免除零错误
            if np.any(prev_prices == 0) or prev_volume == 0:
                continue
            
            # 第1天：相对于窗口前一天的收盘价（价格特征）
            prev_close = prev_prices[3]  # 前一天的收盘价
            if prev_close == 0:
                continue
            input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
            # 成交量特征：直接使用相对变化比例
            input_seq[0, 4] = (input_seq_raw[0, 4] - prev_volume) / prev_volume
            
            # 第2-40天：相对于前一天的收盘价
            valid_sample = True
            for i in range(1, context_length):
                yesterday_close = input_seq_raw[i-1, 3]  # 前一天的收盘价
                yesterday_volume = input_seq_raw[i-1, 4]  # 前一天的成交量
                
                if yesterday_close == 0 or yesterday_volume == 0:
                    valid_sample = False
                    break
                
                input_seq[i, :4] = (input_seq_raw[i, :4] - yesterday_close) / yesterday_close
                # 成交量特征：直接使用相对变化比例
                input_seq[i, 4] = (input_seq_raw[i, 4] - yesterday_volume) / yesterday_volume
            
            if not valid_sample:
                continue
            
            # 统一使用涨幅型标签：基于未来涨幅大小
            original_start_price = stock_data[start_idx + context_length - 1, 3]
            original_end_price = stock_data[start_idx + required_length - 1, 3]

            if original_start_price == 0:
                continue

            cumulative_return = (original_end_price - original_start_price) / original_start_price

            # 测试集二分类：与训练集保持一致
            if cumulative_return >= DataConfig.UPRISE_THRESHOLD:  # 涨幅≥阈值
                target = 1.0
            else:  # 涨幅<阈值
                target = 0.0

            all_possible_samples.append((input_seq, target, stock_idx, start_idx, cumulative_return))
    
    print(f"总共可用样本: {len(all_possible_samples)} 个")
    
    # 使用全部样本进行评估（更科学，且评估速度很快）
    selected_samples = all_possible_samples
    print(f"使用全部 {len(selected_samples)} 个样本进行评估")
    
    # 按股票索引和时间索引排序，确保顺序一致
    selected_samples.sort(key=lambda x: (x[2], x[3]))
    
    # 分离输入和标签
    for input_seq, target, stock_idx, start_idx, cumulative_return in selected_samples:
        eval_inputs.append(input_seq)
        eval_targets.append(target)
        eval_cumulative_returns.append(cumulative_return)
    
    eval_inputs = np.array(eval_inputs)
    eval_targets = np.array(eval_targets)
    eval_cumulative_returns = np.array(eval_cumulative_returns)
    
    # 保存评估样本信息以便调试
    print(f"评估样本详细信息:")
    print(f"  样本总数: {len(eval_inputs)}")
    print(f"  来自股票数: {len(set(s[2] for s in selected_samples))}")
    print(f"  时间窗口范围: {min(s[3] for s in selected_samples)} - {max(s[3] for s in selected_samples)}")
    
    # 打印类别分布
    unique, counts = np.unique(eval_targets, return_counts=True)
    class_names = ['不上涨', '上涨']
    print("评估集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[int(cls)]}: {count} 个样本 ({count/len(eval_targets)*100:.1f}%)")
    
    # 打印收益率统计
    print(f"\n真实收益率统计:")
    print(f"  最小值: {np.min(eval_cumulative_returns)*100:.2f}%")
    print(f"  最大值: {np.max(eval_cumulative_returns)*100:.2f}%")
    print(f"  平均值: {np.mean(eval_cumulative_returns)*100:.2f}%")
    print(f"  中位数: {np.median(eval_cumulative_returns)*100:.2f}%")
    print(f"  ≥0%样本: {np.sum(eval_cumulative_returns >= 0)} ({np.sum(eval_cumulative_returns >= 0)/len(eval_cumulative_returns)*100:.1f}%)")
    print(f"  ≥3%样本: {np.sum(eval_cumulative_returns >= 0.03)} ({np.sum(eval_cumulative_returns >= 0.03)/len(eval_cumulative_returns)*100:.1f}%)")
    print(f"  ≥10%样本: {np.sum(eval_cumulative_returns >= 0.10)} ({np.sum(eval_cumulative_returns >= 0.10)/len(eval_cumulative_returns)*100:.1f}%)")
    
    return eval_inputs, eval_targets, eval_cumulative_returns

# 批量评估函数
def evaluate_model_batch(model, eval_inputs, eval_targets, eval_cumulative_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE):
    """
    使用批处理进行快速评估（二分类）
    返回: (total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_percent_stats)
    
    top_percent_stats: 按预测概率排序后，前1%/5%/10%样本的收益统计
    """
    model.eval()
    total = 0
    class_correct = [0, 0]  # [不上涨正确数, 上涨正确数]
    class_total = [0, 0]    # [不上涨总数, 上涨总数]
    
    # 新增：预测统计
    pred_positive_correct = 0  # 预测上涨且正确的数量
    pred_positive_total = 0    # 预测上涨的总数量
    pred_non_negative = 0       # 预测上涨且实际涨幅≥0%的数量
    
    # 新增：用于AUC计算和Top-K排序的列表
    all_probabilities = []
    all_targets = []
    all_returns = []  # 存储所有样本的实际收益率
    
    # 新增：置信度区间统计 {区间名称: [预测上涨且正确数, 预测上涨总数, 预测上涨且实际涨幅≥0%数]}
    confidence_stats = {
        '0.50-0.55': [0, 0, 0],
        '0.55-0.58': [0, 0, 0],
        '0.58-0.60': [0, 0, 0],
        '0.60-0.70': [0, 0, 0],
        '0.70-1.00': [0, 0, 0]
    }
    
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # 批量处理 (使用BF16精度)
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                      dtype=torch.bfloat16).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            batch_returns = eval_cumulative_returns[start_idx:end_idx]  # 获取实际涨跌幅
            
            # 批量推理
            batch_outputs = model(batch_inputs)  # [batch_size, 1]
            # BF16需要先转为FP32再转numpy
            batch_probabilities = torch.sigmoid(batch_outputs).float().cpu().numpy().flatten()
            batch_predictions = (batch_probabilities > 0.5).astype(int)  # 概率>0.5预测为上涨
            
            # 收集所有概率、标签和收益率用于后续计算
            all_probabilities.extend(batch_probabilities)
            all_targets.extend(batch_targets)
            all_returns.extend(batch_returns)
            
            # 批量计算得分
            for j in range(len(batch_targets)):
                target = int(batch_targets[j])
                prediction = batch_predictions[j]
                actual_return = batch_returns[j]  # 获取实际涨跌幅
                probability = batch_probabilities[j]  # 获取预测概率
                
                class_total[target] += 1
                total += 1
                
                # 统计预测上涨的情况
                if prediction == 1:
                    pred_positive_total += 1
                    if target == 1:  # 预测上涨且实际上涨
                        pred_positive_correct += 1
                    if actual_return >= 0:  # 预测上涨且实际涨幅≥0%
                        pred_non_negative += 1
                    
                    # 统计不同置信度区间的精确度
                    if 0.50 <= probability < 0.55:
                        confidence_stats['0.50-0.55'][1] += 1
                        if target == 1:
                            confidence_stats['0.50-0.55'][0] += 1
                        if actual_return >= 0:
                            confidence_stats['0.50-0.55'][2] += 1
                    elif 0.55 <= probability < 0.58:
                        confidence_stats['0.55-0.58'][1] += 1
                        if target == 1:
                            confidence_stats['0.55-0.58'][0] += 1
                        if actual_return >= 0:
                            confidence_stats['0.55-0.58'][2] += 1
                    elif 0.58 <= probability < 0.60:
                        confidence_stats['0.58-0.60'][1] += 1
                        if target == 1:
                            confidence_stats['0.58-0.60'][0] += 1
                        if actual_return >= 0:
                            confidence_stats['0.58-0.60'][2] += 1
                    elif 0.60 <= probability < 0.70:
                        confidence_stats['0.60-0.70'][1] += 1
                        if target == 1:
                            confidence_stats['0.60-0.70'][0] += 1
                        if actual_return >= 0:
                            confidence_stats['0.60-0.70'][2] += 1
                    elif 0.70 <= probability <= 1.00:
                        confidence_stats['0.70-1.00'][1] += 1
                        if target == 1:
                            confidence_stats['0.70-1.00'][0] += 1
                        if actual_return >= 0:
                            confidence_stats['0.70-1.00'][2] += 1
                
                # 统计预测正确性（用于显示准确率）
                if prediction == target:
                    class_correct[target] += 1
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_targets, all_probabilities)
    except ValueError:
        # 如果所有标签都是同一类，AUC无法计算
        auc_score = 0.5  # 随机分类器的AUC
    
    # 🔑 核心改进：按预测概率排序，计算Top N%样本的收益统计
    # 这能真实反映模型的排序能力（选股能力）
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)
    
    # 按预测概率从高到低排序
    sorted_indices = np.argsort(all_probabilities)[::-1]  # 降序排列
    
    # 计算Top N%的统计（使用配置文件中的百分比）
    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(sorted_indices) * percent / 100))  # 至少1个样本
    top_indices = sorted_indices[:top_k]
    
    top_returns = all_returns[top_indices]
    top_targets = all_targets[top_indices]
    
    # 统计：样本数、累计收益、平均收益
    top_stats = {
        'count': top_k,
        'total_return': np.sum(top_returns),
        'avg_return': np.mean(top_returns),
    }
    
    return total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats

def calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=DataConfig.EVAL_BATCH_SIZE):
    """
    计算测试集损失值
    """
    model.eval()
    total_loss = 0
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # 批量处理 (使用BF16精度)
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                      dtype=torch.bfloat16).to(device)
            batch_targets = torch.tensor(eval_targets[start_idx:end_idx], 
                                       dtype=torch.bfloat16).to(device)
            
            # 计算损失
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            # BF16的loss可以直接取item
            total_loss += batch_loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def print_sample_predictions(model, eval_inputs, eval_targets, device, num_samples=10, epoch=None):
    """
    随机挑选样本并打印模型的输出值，用于观察预测集中的问题
    """
    model.eval()
    
    # 随机选择样本索引
    total_samples = len(eval_inputs)
    if num_samples > total_samples:
        num_samples = total_samples
    
    # 使用当前epoch作为随机种子，确保每轮选择不同的样本
    if epoch is not None:
        np.random.seed(DataConfig.RANDOM_SEED + epoch)
    
    sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
    sample_indices = sorted(sample_indices)  # 排序以便观察
    
    print(f"  随机样本预测详情 (第{epoch}轮):")
    print(f"  {'样本':<4} {'真实标签':<8} {'模型输出':<12} {'预测概率':<10} {'预测标签':<8} {'预测结果':<8}")
    print(f"  {'-'*4} {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # 获取单个样本 (使用BF16精度)
            sample_input = torch.tensor(eval_inputs[idx:idx+1], dtype=torch.bfloat16).to(device)
            true_label = eval_targets[idx]
            
            # 模型预测
            model_output = model(sample_input)
            # BF16需要先转为FP32再转python标量
            raw_output = model_output.float().cpu().item()
            probability = torch.sigmoid(model_output).float().cpu().item()
            predicted_label = 1 if probability > 0.5 else 0
            
            # 判断预测结果
            if predicted_label == int(true_label):
                result = "✓正确"
            else:
                if int(true_label) == 1 and predicted_label == 0:
                    result = "✗漏涨"
                elif int(true_label) == 0 and predicted_label == 1:
                    result = "✗误涨"
                else:
                    result = "✗错误"
            
            # 格式化输出
            true_label_str = "上涨" if int(true_label) == 1 else "不上涨"
            pred_label_str = "上涨" if predicted_label == 1 else "不上涨"
            
            print(f"  {i+1:<4} {true_label_str:<8} {raw_output:<12.4f} {probability:<10.4f} {pred_label_str:<8} {result:<8}")
    
    print()  # 空行

# 预计算训练数据集函数
def precompute_training_dataset(train_stock_info, train_weights, 
                               batch_size, batches_per_epoch, seed=None):
    """
    预计算每轮训练所需的训练数据集（使用滚动窗口标准化）
    自动根据批大小和批数量计算需要的样本数
    返回: (epoch_inputs, epoch_targets)
    """
    samples_per_epoch = batch_size * batches_per_epoch
    
    if seed is not None:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        random.seed(seed)
    
    # 直接生成所有需要的样本（使用滚动窗口标准化）
    epoch_inputs, epoch_targets = generate_batch_samples_improved(
        train_stock_info, train_weights, samples_per_epoch)
    
    return np.array(epoch_inputs), np.array(epoch_targets)

# 改进的训练函数
def train_model(model, train_stock_info, test_stock_info, train_weights, epochs=TrainingConfig.EPOCHS, 
               learning_rate=TrainingConfig.LEARNING_RATE, device=None, 
               batch_size=TrainingConfig.BATCH_SIZE, batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH):
    """
    使用预计算训练数据集和固定评估集的训练函数（使用滚动窗口标准化避免数据泄露）
    提高训练效率，确保评估的一致性
    
    注意：本训练函数使用 BF16 (bfloat16) 精度进行训练
    - 训练速度比FP32快约2倍
    - 内存占用减半
    - 模型精度与FP32相当
    """
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)
    print("训练精度: BF16 (Brain Floating Point 16)")
    print("数据标准化: 滚动窗口标准化（避免数据泄露）")
    print(f"数据划分: 按时间划分，最近{DataConfig.TEST_DAYS}天作为测试集")
    print("="*60 + "\n")
    # 设置训练随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)
    
    # 创建固定的评估数据集（训练开始前创建一次，使用滚动窗口标准化）
    eval_inputs, eval_targets, eval_cumulative_returns = create_fixed_evaluation_dataset(test_stock_info)
    
    # 使用动态加权BCE损失函数，根据每轮训练数据的正负样本比例动态调整权重
    # 正样本权重固定为4.0，负样本权重动态调整（0.5~1.0）
    criterion = DynamicWeightedBCE(pos_weight=4.0)
    
    # 创建测试集专用的损失函数（使用标准BCE，正负样本权重都为1.0，保证可比性）
    eval_criterion = DynamicWeightedBCE(pos_weight=1.0)
    # 不调用update_weights，保持初始值: pos_weight=1.0, neg_weight=1.0
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # 创建预热调度器
    warmup_scheduler = WarmupScheduler(
        optimizer, 
        warmup_epochs=TrainingConfig.WARMUP_EPOCHS,
        target_lr=learning_rate,
        start_lr=TrainingConfig.WARMUP_START_LR
    )
    
    # 创建主调度器
    # 注意：虽然warmup_scheduler已经将optimizer的学习率设置为start_lr，
    # 但主调度器应该基于target_lr来工作。
    # 我们在创建主调度器前先临时设置为target_lr，这样主调度器就会以正确的学习率为基准
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # 根据配置选择主调度器
    if TrainingConfig.USE_COSINE_ANNEALING:
        # 修复：使用总轮数-预热轮数作为T_max，确保余弦退火覆盖整个训练过程
        # 避免在训练后期学习率再次上升
        total_main_epochs = epochs - TrainingConfig.WARMUP_EPOCHS
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_main_epochs,  # 使用实际的主训练轮数
            eta_min=TrainingConfig.COSINE_ETA_MIN
        )
        scheduler_type = f"余弦退火(周期={total_main_epochs}轮)"
    else:
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=TrainingConfig.SCHEDULER_STEP_SIZE, 
            gamma=TrainingConfig.SCHEDULER_GAMMA
        )
        scheduler_type = "阶梯衰减"
    
    # 创建主调度器后，需要将学习率重新设置回start_lr，因为训练从预热开始
    for param_group in optimizer.param_groups:
        param_group['lr'] = TrainingConfig.WARMUP_START_LR
    
    print(f"学习率调度策略: {scheduler_type}")
    
    best_loss = float('inf')  # 使用测试集loss作为保存标准（越低越好）
    best_model_state = None  # 缓存最佳模型状态（内存中）
    best_epoch = 0  # 记录最佳模型所在轮次
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段 - 更新学习率
        if warmup_scheduler.is_warmup_phase():
            # 预热阶段：使用预热调度器
            current_lr = warmup_scheduler.step(epoch)
            lr_status = f"预热阶段 ({epoch + 1}/{TrainingConfig.WARMUP_EPOCHS})"
        else:
            # 预热结束后：使用主调度器获取当前学习率
            current_lr = main_scheduler.get_last_lr()[0]
            lr_status = "正常训练"
        
        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status})')
        
        # 预计算当前轮次的训练数据（使用滚动窗口标准化）
        epoch_seed = DataConfig.RANDOM_SEED + epoch  # 每轮使用不同的种子确保数据多样性
        epoch_inputs, epoch_targets = precompute_training_dataset(
            train_stock_info, train_weights, batch_size, batches_per_epoch, epoch_seed)
        
        # 注意：动态权重更新已移至每个Batch内部，确保每次参数更新时都基于当前Batch的正负样本比例进行平衡
        
        # 打印本轮标签分布信息（二分类：1.0/0.0）
        count_positive = np.sum(epoch_targets >= 0.5)  # 正样本（涨幅≥5%）
        count_negative = np.sum(epoch_targets < 0.5)   # 负样本（涨幅<5%）
        total_count = len(epoch_targets)
        
        print(f'  标签分布: 上涨(≥5%)={count_positive}({count_positive/total_count:.1%}), 不上涨(<5%)={count_negative}({count_negative/total_count:.1%})')
        print(f'  动态权重: 每Batch独立计算（正样本固定={criterion.pos_weight.item():.1f}，负样本按比例动态调整）')
        
        # 显示预热进度和调度器信息
        if warmup_scheduler.is_warmup_phase():
            warmup_progress = (epoch + 1) / TrainingConfig.WARMUP_EPOCHS * 100
            print(f'  预热进度: {warmup_progress:.1f}% (第{epoch + 1}轮/共{TrainingConfig.WARMUP_EPOCHS}轮)')
            print(f'  学习率变化: {TrainingConfig.WARMUP_START_LR:.2e} → {current_lr:.2e} → {learning_rate:.2e}(目标)')
        else:
            # 预热结束后显示当前调度器状态
            if TrainingConfig.USE_COSINE_ANNEALING:
                # 计算余弦退火的理论学习率
                import math
                # 修复：使用实际的主训练轮数计算进度
                total_main_epochs = epochs - TrainingConfig.WARMUP_EPOCHS
                current_main_epoch = epoch - TrainingConfig.WARMUP_EPOCHS
                progress = current_main_epoch / total_main_epochs
                theoretical_lr = TrainingConfig.COSINE_ETA_MIN + (learning_rate - TrainingConfig.COSINE_ETA_MIN) * \
                               (1 + math.cos(math.pi * progress)) / 2
                print(f'  余弦退火进度: {progress*100:.1f}% (第{current_main_epoch+1}轮/共{total_main_epochs}轮), 理论学习率: {theoretical_lr:.2e}')
            else:
                print(f'  阶梯衰减: 每{TrainingConfig.SCHEDULER_STEP_SIZE}轮衰减{TrainingConfig.SCHEDULER_GAMMA}倍')
        
        # 将预计算的数据转换为tensor并移到设备上 (使用BF16精度)
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.bfloat16).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.bfloat16).to(device)
        
        # 训练循环：使用预计算的数据
        for step in range(batches_per_epoch):
            start_idx = step * batch_size
            end_idx = min((step + 1) * batch_size, len(epoch_inputs_tensor))
            
            # 从预计算的数据中取一个batch
            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            
            # 🔑 每个Batch都更新动态权重，确保每次参数更新时正负样本贡献平衡
            criterion.update_weights(batch_targets)
            
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = criterion(output, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 实时更新进度显示
            progress = (step + 1) / batches_per_epoch * 100
            avg_loss = total_loss / (step + 1)
            print(f'\r  训练进度: {progress:.1f}% ({step + 1}/{batches_per_epoch}), 平均损失: {avg_loss:.4f}', end='', flush=True)
        
        print()  # 换行
        print()  # 空行
        
        # 清理预计算的数据以释放内存
        del epoch_inputs_tensor, epoch_targets_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 更新学习率
        # 注意：预热阶段的学习率已经在epoch开始时由warmup_scheduler.step()更新
        # 只有预热结束后才使用主调度器
        if not warmup_scheduler.is_warmup_phase():
            main_scheduler.step()  # 更新主调度器（余弦退火或阶梯衰减）
        
        # 固定评估集评估
        total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats = evaluate_model_batch(
            model, eval_inputs, eval_targets, eval_cumulative_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE
        )
        
        # 计算测试集损失（使用固定权重的eval_criterion，保证可比性）
        test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device, batch_size=DataConfig.EVAL_BATCH_SIZE)
        
        # 随机挑选5组样本打印模型输出值
        print_sample_predictions(model, eval_inputs, eval_targets, device, num_samples=5, epoch=epoch+1)
        
        # 打印详细结果
        class_names = ['不上涨', '上涨']
        for i in range(2):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f'  {class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.3f}')
            else:
                print(f'  {class_names[i]}: 0/0 = 0.000 (无样本)')
        
        # 计算上涨准确率（预测上涨后真上涨的概率）
        if pred_positive_total > 0:
            precision = pred_positive_correct / pred_positive_total
            non_negative_rate = pred_non_negative / pred_positive_total
            print(f'  上涨准确率: {pred_positive_correct}/{pred_positive_total} = {precision:.3f} 准确率: {pred_non_negative}/{pred_positive_total} = {non_negative_rate:.3f}')
        else:
            print(f'  上涨准确率: 0/0 = 0.000 (无预测上涨)')
        
        # 打印置信度区间的精确度统计
        print(f'  置信度区间精确度:')
        for interval in ['0.50-0.55', '0.55-0.58', '0.58-0.60', '0.60-0.70', '0.70-1.00']:
            correct, total_pred, non_negative = confidence_stats[interval]
            if total_pred > 0:
                precision = correct / total_pred
                non_negative_rate = non_negative / total_pred
                print(f'    {interval}: 上涨准确={correct}/{total_pred}={precision:.3f}, 非负准确={non_negative}/{total_pred}={non_negative_rate:.3f}')
            else:
                print(f'    {interval}: 无预测')
        
        overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
        avg_loss = total_loss / batches_per_epoch
        
        print(f'  总体准确率: {overall_acc:.3f}')
        print(f'  Top{DataConfig.TOP_PERCENT}%收益: 样本数={top_stats["count"]}, 平均={top_stats["avg_return"]*100:+.2f}%, 累计={top_stats["total_return"]*100:+.2f}%')
        print(f'  AUC得分: {auc_score:.4f}')
        print(f'  训练集损失: {avg_loss:.4f}, 测试集损失: {test_loss:.4f}')
        
        # 保存最佳模型（使用测试集loss作为主要标准，同时监控AUC）
        MIN_AUC = DataConfig.MIN_AUC
        
        # 判断是否保存模型
        should_save = False
        save_reason = ""
        
        if auc_score < MIN_AUC:
            print(f'  ⚠ AUC过低({auc_score:.4f}<{MIN_AUC})，模型分类能力不足，暂不更新')
        elif test_loss < best_loss:
            should_save = True
            save_reason = f'测试集Loss降低: {best_loss:.4f} → {test_loss:.4f}'
        
        if should_save:
            best_loss = test_loss
            best_epoch = epoch + 1
            # 缓存模型状态到内存（深拷贝），不立即写入磁盘
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'  ✓ 发现更好的模型！{save_reason}（已缓存到内存）')
            print(f'    详情: AUC={auc_score:.4f}, Top{DataConfig.TOP_PERCENT}%收益: 平均={top_stats["avg_return"]*100:+.2f}%, 累计={top_stats["total_return"]*100:+.2f}%')
        
        print("-" * 50)
    
    # 训练结束后，将最佳模型保存到磁盘
    if best_model_state is not None:
        print("\n" + "=" * 50)
        print(f"训练完成！正在保存最佳模型...")
        print(f"最佳模型来自第 {best_epoch} 轮，测试集Loss: {best_loss:.4f}")
        torch.save(best_model_state, ModelSaveConfig.get_best_model_path())
        print(f"✓ 最佳模型已保存到: {ModelSaveConfig.get_best_model_path()}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠ 警告：未找到符合条件的最佳模型（AUC要求未达标）")
        print("=" * 50)

if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 打印配置摘要
    print_config_summary()
    
    # 获取设备信息
    device = DeviceConfig.print_device_info()

    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    
    # 使用改进的数据加载函数（按时间划分，避免数据泄露）
    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()

    # 计算股票选择权重
    train_weights = calculate_stock_weights(train_stock_info)
    test_weights = calculate_stock_weights(test_stock_info)
    
    # 打印数据集统计信息
    print("\n" + "="*60)
    print("数据集划分统计")
    print("="*60)
    
    train_lengths = [info['data_length'] for info in train_stock_info]
    test_lengths = [info['data_length'] for info in test_stock_info]
    
    print(f"训练集:")
    print(f"  股票数量: {len(train_stock_info)}")
    print(f"  数据长度: 最小={min(train_lengths)}, 最大={max(train_lengths)}, 平均={np.mean(train_lengths):.1f}")
    print(f"  采样权重: {min(train_weights):.3f} - {max(train_weights):.3f}")
    
    print(f"\n测试集:")
    print(f"  股票数量: {len(test_stock_info)}")
    print(f"  数据长度: 最小={min(test_lengths)}, 最大={max(test_lengths)}, 平均={np.mean(test_lengths):.1f}")
    print(f"  时间范围: 每只股票的最近 {DataConfig.TEST_DAYS} 天")
    
    print(f"\n前3只股票示例:")
    for i in range(min(3, len(train_stock_info))):
        train_info = train_stock_info[i]
        print(f"  {train_info['file_name']}: 训练集长度={train_info['data_length']}, 权重={train_weights[i]:.3f}")
    
    print("="*60)

    print("正在创建 Transformer 模型 (BF16精度)...")
    model = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM, 
        d_model=ModelConfig.D_MODEL, 
        nhead=ModelConfig.NHEAD, 
        num_layers=ModelConfig.NUM_LAYERS, 
        output_dim=ModelConfig.OUTPUT_DIM,
        max_seq_len=ModelConfig.MAX_SEQ_LEN
    ).to(device)
    
    # 将模型参数转换为BF16精度
    model = model.to(dtype=torch.bfloat16)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    print("开始训练...")
    # 使用带固定评估集的训练函数（使用滚动窗口标准化）
    train_model(model, train_stock_info, test_stock_info, train_weights, device=device)
    
    # 保存最终模型（训练结束时的状态）
    final_model_path = ModelSaveConfig.get_final_model_path(ModelConfig.D_MODEL)
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

# ==================== 统一预测函数 ====================
def normalize_data_for_prediction(data):
    """
    统一的数据归一化函数（滚动窗口标准化）
    用于所有预测场景，确保与训练时完全一致
    
    Args:
        data: numpy array, shape [seq_len, 5] (OHLCV)
        
    Returns:
        normalized_data: numpy array, shape [seq_len-1, 5] 或 None（如果数据无效）
    """
    if len(data) < 2:
        return None
    
    normalized_data = np.zeros_like(data, dtype=np.float64)
    
    # 滚动窗口标准化：每天相对于前一天的涨跌幅
    for i in range(1, len(data)):
        yesterday_close = data[i-1, 3]  # 前一天的收盘价
        yesterday_volume = data[i-1, 4]  # 前一天的成交量
        
        if yesterday_close == 0 or yesterday_volume == 0:
            return None  # 数据异常
        
        # 价格特征：相对于前一天收盘价的涨跌幅
        normalized_data[i, :4] = (data[i, :4] - yesterday_close) / yesterday_close
        # 成交量特征：相对于前一天成交量的变化比例
        normalized_data[i, 4] = (data[i, 4] - yesterday_volume) / yesterday_volume
    
    # 只返回标准化后的数据（去掉第0天基准数据）
    return normalized_data[1:]

def predict_single_stock(model_path, stock_data, device=None):
    """
    统一的单股票预测函数
    
    Args:
        model_path: 模型文件路径
        stock_data: numpy array, shape [seq_len, 5] (OHLCV)，至少需要CONTEXT_LENGTH+1天数据
        device: 计算设备
        
    Returns:
        probability: float, 预测概率 [0, 1]，如果预测失败返回None
    """
    if device is None:
        device = DeviceConfig.get_device()
    
    # 检查数据长度
    if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
        return None
    
    # 取最新数据
    recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
    
    # 归一化
    normalized_data = normalize_data_for_prediction(recent_data)
    if normalized_data is None:
        return None
    
    # 加载模型
    try:
        model = EnhancedStockTransformer(
            input_dim=ModelConfig.INPUT_DIM,
            d_model=ModelConfig.D_MODEL,
            nhead=ModelConfig.NHEAD,
            num_layers=ModelConfig.NUM_LAYERS,
            output_dim=ModelConfig.OUTPUT_DIM,
            max_seq_len=ModelConfig.MAX_SEQ_LEN
        ).to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 预测
    try:
        input_tensor = torch.tensor(normalized_data, dtype=torch.bfloat16).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).float().cpu().item()
        
        return probability
    except Exception as e:
        print(f"预测失败: {e}")
        return None

def predict_multiple_stocks(model_path, stock_files_data, device=None):
    """
    统一的多股票预测函数
    
    Args:
        model_path: 模型文件路径
        stock_files_data: dict, {文件名: numpy_array}
        device: 计算设备
        
    Returns:
        predictions: list of (filename, probability)
    """
    if device is None:
        device = DeviceConfig.get_device()
    
    predictions = []
    
    # 加载模型（只加载一次）
    try:
        model = EnhancedStockTransformer(
            input_dim=ModelConfig.INPUT_DIM,
            d_model=ModelConfig.D_MODEL,
            nhead=ModelConfig.NHEAD,
            num_layers=ModelConfig.NUM_LAYERS,
            output_dim=ModelConfig.OUTPUT_DIM,
            max_seq_len=ModelConfig.MAX_SEQ_LEN
        ).to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return predictions
    
    # 批量预测
    with torch.no_grad():
        for filename, stock_data in stock_files_data.items():
            # 检查数据长度
            if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
                continue
            
            # 取最新数据并归一化
            recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
            normalized_data = normalize_data_for_prediction(recent_data)
            if normalized_data is None:
                continue
            
            try:
                # 预测
                input_tensor = torch.tensor(normalized_data, dtype=torch.bfloat16).unsqueeze(0).to(device)
                output = model(input_tensor)
                probability = torch.sigmoid(output).float().cpu().item()
                
                predictions.append((filename, probability))
            except Exception as e:
                print(f"{filename} 预测失败: {e}")
                continue
    
    return predictions