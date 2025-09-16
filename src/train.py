'''
训练脚本

评分制度（以代码实现为准）：
提供预测机会，预测正确加1分
预测错误则按下面策略处理：
1.假阳性（预测上涨但实际不上涨）：-1分 
2.假阴性（预测不上涨但实际上涨）：-0.5分 
3.其余情况不加分也不扣分。
'''

import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
import random
import math
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import (ModelConfig, TrainingConfig, DataConfig, 
                   EvaluationConfig, DeviceConfig, ModelSaveConfig,
                   print_config_summary)

# 动态加权BCE损失函数实现
class DynamicWeightedBCE(nn.Module):
    """
    动态加权BCE损失函数，根据每轮训练数据的正负样本比例动态调整权重
    使用标准的类别不平衡处理公式：weight = total_samples / (num_classes * class_count)
    """
    def __init__(self, reduction='mean'):
        super(DynamicWeightedBCE, self).__init__()
        self.reduction = reduction
        
        # 动态权重，会在训练过程中更新
        self.register_buffer('positive_weight', torch.tensor(1.0))
        self.register_buffer('negative_weight', torch.tensor(1.0))
        
    def update_weights(self, targets):
        """
        根据当前批次的目标标签更新权重
        使用标准的类别不平衡处理公式：weight = total_samples / (num_classes * class_count)
        targets: [batch_size] 真实标签 (0=不上涨, 1=上涨)
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算正负样本数量
        positive_count = np.sum(targets == 1)
        negative_count = np.sum(targets == 0)
        total_count = len(targets)
        
        if total_count == 0:
            return
            
        # 使用标准的类别不平衡权重公式
        # weight = total_samples / (num_classes * class_count)
        num_classes = 2  # 二分类：不上涨(0) 和 上涨(1)
        
        if positive_count > 0 and negative_count > 0:
            # 标准类别权重计算
            self.positive_weight = torch.tensor(total_count / (num_classes * positive_count))
            self.negative_weight = torch.tensor(total_count / (num_classes * negative_count))
            
            # 限制权重范围，避免过度不平衡
            max_weight = 5.0
            min_weight = 0.1
            self.positive_weight = torch.clamp(self.positive_weight, min_weight, max_weight)
            self.negative_weight = torch.clamp(self.negative_weight, min_weight, max_weight)
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, 1] 模型输出的logits
        targets: [batch_size] 真实标签 (0=不上涨, 1=上涨)
        """
        # 确保输入形状正确
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets, reduction='none')
        
        # 应用动态权重
        weights = torch.where(targets == 1, self.positive_weight, self.negative_weight)
        weighted_loss = weights * bce_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


# 时间感知的位置编码类
class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=ModelConfig.MAX_SEQ_LEN, decay_factor=ModelConfig.POSITIONAL_ENCODING_DECAY):
        super(TimeAwarePositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        time_weights = torch.exp(-decay_factor * torch.arange(max_seq_len - 1, -1, -1, dtype=torch.float))
        pe = pe * time_weights.unsqueeze(1)
        
        self.register_buffer('pe', pe)
        
        # 添加层归一化用于残差连接
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 使用残差连接：输出 = LayerNorm(输入 + 位置编码)
        seq_len = x.size(1)
        pe_slice = self.pe[:seq_len, :].unsqueeze(0)
        return self.norm(x + pe_slice)

class MultiHeadAttention(nn.Module):
    """
    标准的多头注意力机制
    让模型自动学习每个头应该关注什么特征，不人为干预
    """
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        assert d_model % nhead == 0
        
        # 使用标准的MultiheadAttention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # 添加层归一化和残差连接支持
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)
        
    def forward(self, x, attn_mask=None):
        # 保存输入用于残差连接
        residual = x
        
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)

        # 标准注意力计算
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # 残差连接 + 层归一化
        output = self.norm(residual + self.dropout(attn_output))
        return output

# 增强的注意力层
class TransformerLayer(nn.Module):
    """
    增强的注意力层，使用标准的多头注意力机制
    设计理念：让模型自动学习应该关注什么特征，不人为干预
    """
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        
        # 使用标准多头注意力
        self.attention = MultiHeadAttention(d_model, nhead)
        
        # 前馈网络，用于进一步处理注意力的输出
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 先扩展维度
            nn.ReLU(),                        # 激活函数
            nn.Dropout(ModelConfig.DROPOUT_RATE),  # 防过拟合
            nn.Linear(d_model * 4, d_model),  # 再压缩回原维度
        )
        
        # 层归一化，帮助训练稳定
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        
    def create_temporal_mask(self, seq_len):
        """
        创建时间感知的注意力掩码
        设计理念：近期数据更重要，但也要考虑历史信息
        """
        mask = torch.zeros(seq_len, seq_len)
        
        # 使用滑动窗口 + 时间衰减
        window_size = ModelConfig.ATTENTION_WINDOW_SIZE
        
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                
                # 如果距离在窗口内，给予高权重
                if distance <= window_size:
                    mask[i, j] = 1.0
                else:
                    # 超出窗口的部分使用时间衰减
                    decay_factor = ModelConfig.TEMPORAL_DECAY
                    mask[i, j] = math.exp(-decay_factor * (distance - window_size))
                    
        return mask
        
    def forward(self, x):
        # x的shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)

        # 创建时间感知掩码
        temporal_mask = self.create_temporal_mask(seq_len).to(dtype=x.dtype, device=x.device)

        # 注意力计算
        attention_out = self.attention(x, attn_mask=temporal_mask)
        
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(attention_out))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# 增强版的Transformer模型
class EnhancedStockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, max_seq_len, decay_factor):
        super(EnhancedStockTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        # 添加嵌入层的归一化
        self.embedding_norm = nn.LayerNorm(d_model)
        
        self.pos_encoding = TimeAwarePositionalEncoding(d_model, max_seq_len, decay_factor)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        
        # 在输出前添加最终的层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 简化输出层，减少过拟合
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 降维
            nn.ReLU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(d_model // 2, output_dim)  # 最终输出
        )
        
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        
    def forward(self, x):
        # 1. 特征嵌入 + 残差连接风格的归一化
        x = self.embedding_norm(self.embedding(x))
        
        # 2. 位置编码（内部已有残差连接）
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. Transformer层（每层内部都有残差连接）
        for layer in self.layers:
            x = layer(x)
        
        # 4. 最终归一化
        x = self.final_norm(x)
        
        # 5. 取最后时间步 + 输出投影
        last_hidden = x[:, -1, :]
        output = self.output_projection(last_hidden)
        
        return output

# 数据预处理函数
def load_and_preprocess_data(data_dir=DataConfig.DATA_DIR, test_ratio=DataConfig.TEST_RATIO, seed=DataConfig.RANDOM_SEED):
    """
    改进的数据加载和预处理函数
    确保训练集和测试集完全独立，没有数据泄露
    使用固定的31个测试文件以确保评估的一致性
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    all_files.sort()  # 确保文件顺序一致
    
    # 使用固定的31个测试文件（按文件名排序后的前31个）
    test_size = 31
    if len(all_files) < test_size:
        print(f"警告: 可用文件数 ({len(all_files)}) 少于31个，将使用所有文件作为测试集")
        test_size = len(all_files)
    
    test_files = set(all_files[:test_size])  # 固定使用前31个文件作为测试集
    train_files = [f for f in all_files if f not in test_files]
    
    print(f"训练股票文件: {len(train_files)} 个")
    print(f"测试股票文件: {len(test_files)} 个 (固定31个文件)")
    print(f"测试文件列表: {list(test_files)[:5]}...")  # 显示前5个测试文件

    def process_files(file_list):
        data_list = []
        stock_info_list = []  # 新增：存储股票信息
        
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path)
            try:
                # 获取时间列用于判断2021年
                time_column = df['time'].values
                
                # 找到2021年的起始位置
                year_2021_start = None
                for i, time_str in enumerate(time_column):
                    year = int(time_str.split('/')[0])
                    if year >= 2021:
                        year_2021_start = i
                        break
                
                # 如果没找到2021年，使用最后一个位置
                if year_2021_start is None:
                    year_2021_start = len(time_column) - 1
                
                data = df[['start', 'max', 'min', 'end', 'volume', 'marketvolume', 'marketlimit', 'marketrange']].values
                
                # 每只股票单独标准化
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                if np.any(std == 0):
                    raise ValueError(f"文件 {file} 包含标准差为0的列")
                normalized_data = (data - mean) / std
                
                data_list.append(normalized_data)
                
                # 存储股票信息
                stock_info = {
                    'data_length': len(normalized_data),
                    'year_2021_start': year_2021_start,
                    'file_name': file
                }
                stock_info_list.append(stock_info)
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
        
        return data_list, stock_info_list

    train_data, train_stock_info = process_files(train_files)
    test_data, test_stock_info = process_files(test_files)
    
    return train_data, test_data, train_stock_info, test_stock_info

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

# 改进的样本生成函数
def generate_single_sample_improved(all_data, stock_info_list, stock_weights):
    """
    改进的样本生成函数
    1. 根据数据量大小选择股票（数据量大的概率更高）
    2. 选中股票后，选择起始时间在2021年后概率设置为0.6
    """
    for _ in range(100):  # 最多尝试100次生成有效样本
        # 第一步：根据权重选择股票
        stock_index = np.random.choice(len(all_data), p=stock_weights)
        stock_data = all_data[stock_index]
        stock_info = stock_info_list[stock_index]
        
        context_length = DataConfig.CONTEXT_LENGTH  # 使用配置的历史数据长度
        required_length = DataConfig.REQUIRED_LENGTH  # 需要额外3天来计算未来收益
        
        if len(stock_data) < required_length:
            continue
            
        # 第二步：选择起始时间，2021年后概率为0.6
        year_2021_start = stock_info['year_2021_start']
        total_valid_windows = len(stock_data) - required_length + 1
        
        # 计算2021年前后的窗口数量
        windows_before_2021 = max(0, year_2021_start - required_length + 1)
        windows_after_2021 = total_valid_windows - windows_before_2021
        
        if windows_after_2021 > 0 and windows_before_2021 > 0:
            # 有2021年前后的数据，使用0.6概率选择2021年后
            if np.random.random() < 0.6:
                # 选择2021年后的窗口
                start_index = np.random.randint(year_2021_start, len(stock_data) - required_length + 1)
            else:
                # 选择2021年前的窗口
                start_index = np.random.randint(0, year_2021_start)
        else:
            # 只有2021年前或后的数据，随机选择
            start_index = np.random.randint(0, len(stock_data) - required_length + 1)
        
        input_seq = stock_data[start_index:start_index + context_length]  # 60天历史数据
        target_seq = stock_data[start_index + context_length:start_index + required_length]  # 未来3天
        
        # 计算收益率：(未来价格 - 当前价格) / 当前价格
        start_price = input_seq[-1, 3]  # 当前收盘价（第3列是end收盘价）
        end_price = target_seq[-1, 3]   # 3天后的收盘价
        
        if start_price == 0:  # 避免除零错误
            continue
            
        cumulative_return = (end_price - start_price) / start_price
        
        # 二分类标签：上涨为1，不上涨为0
        if cumulative_return >= DataConfig.UPRISE_THRESHOLD:      # 涨幅≥2%：上涨
            target = 1.0
        else:                              # 其他情况：不上涨
            target = 0.0
            
        return input_seq, target
    
    raise ValueError("无法生成有效样本：股票数据长度不足或收盘价为0")

def generate_batch_samples_improved(all_data, stock_info_list, stock_weights, batch_size):
    """
    改进的批量生成训练样本
    返回: (batch_inputs, batch_targets)
    batch_inputs: numpy array, shape [batch_size, context_length, 8]  
    batch_targets: numpy array, shape [batch_size]
    """
    batch_inputs = []
    batch_targets = []
    
    attempts = 0
    max_attempts = batch_size * 10  # 防止无限循环
    
    while len(batch_inputs) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            input_seq, target = generate_single_sample_improved(all_data, stock_info_list, stock_weights)
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        except ValueError:
            continue
    
    if len(batch_inputs) < batch_size:
        raise ValueError(f"无法生成足够的样本，只生成了 {len(batch_inputs)}/{batch_size} 个")
    
    return np.array(batch_inputs), np.array(batch_targets)

# 生成单个样本（保持原有函数用于兼容性）
def generate_single_sample(all_data):
    """
    从股票数据中随机生成一个训练样本
    输入：60天的历史数据
    输出：根据未来3天收益率确定的类别标签
    """
    for _ in range(100):  # 最多尝试100次生成有效样本
        stock_index = np.random.randint(0, len(all_data))
        stock_data = all_data[stock_index]
        context_length = DataConfig.CONTEXT_LENGTH  # 使用配置的历史数据长度
        required_length = DataConfig.REQUIRED_LENGTH  # 需要额外3天来计算未来收益
        
        if len(stock_data) < required_length:
            continue
            
        start_index = np.random.randint(0, len(stock_data) - required_length + 1)
        input_seq = stock_data[start_index:start_index + context_length]  # 60天历史数据
        target_seq = stock_data[start_index + context_length:start_index + required_length]  # 未来3天
        
        # 计算收益率：(未来价格 - 当前价格) / 当前价格
        start_price = input_seq[-1, 3]  # 当前收盘价（第3列是end收盘价）
        end_price = target_seq[-1, 3]   # 3天后的收盘价
        
        if start_price == 0:  # 避免除零错误
            continue
            
        cumulative_return = (end_price - start_price) / start_price
        
        # 二分类标签：上涨为1，不上涨为0
        if cumulative_return >= DataConfig.UPRISE_THRESHOLD:      # 涨幅≥1%：上涨
            target = 1.0
        else:                              # 其他情况：不上涨
            target = 0.0
            
        return input_seq, target
    
    raise ValueError("无法生成有效样本：股票数据长度不足或收盘价为0")

def generate_batch_samples(all_data, batch_size):
    """
    批量生成训练样本
    返回: (batch_inputs, batch_targets)
    batch_inputs: numpy array, shape [batch_size, context_length, 8]  
    batch_targets: numpy array, shape [batch_size]
    """
    batch_inputs = []
    batch_targets = []
    
    attempts = 0
    max_attempts = batch_size * 10  # 防止无限循环
    
    while len(batch_inputs) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            input_seq, target = generate_single_sample(all_data)
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        except ValueError:
            continue
    
    if len(batch_inputs) < batch_size:
        raise ValueError(f"无法生成足够的样本，只生成了 {len(batch_inputs)}/{batch_size} 个")
    
    return np.array(batch_inputs), np.array(batch_targets)

def create_evaluation_dataset(test_data, num_samples=DataConfig.EVAL_SAMPLES):
    eval_inputs = []
    eval_targets = []
    
    print('生成评估样本...')
    for i in range(num_samples):
        input_seq, target = generate_single_sample(test_data)
        eval_inputs.append(input_seq)
        eval_targets.append(target)
        
        # 实时更新进度显示
        progress = (i + 1) / num_samples * 100
        print(f'\r  进度: {progress:.1f}% ({i + 1}/{num_samples})', end='', flush=True)
    
    print()  # 换行
    
    return np.array(eval_inputs), np.array(eval_targets)

# 创建固定的评估数据集
def create_fixed_evaluation_dataset(test_data, num_samples=DataConfig.EVAL_SAMPLES, seed=DataConfig.RANDOM_SEED):
    """
    创建固定的评估数据集，确保每次评估使用相同的样本
    这样可以准确衡量模型的进步情况
    使用严格的随机种子控制以确保完全可重复
    """
    print("正在创建固定的评估数据集...")
    # 设置所有可能的随机种子以确保完全可重复
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    eval_inputs = []
    eval_targets = []
    
    # 预先生成所有可能的样本
    all_possible_samples = []
    context_length = DataConfig.CONTEXT_LENGTH
    required_length = DataConfig.REQUIRED_LENGTH
    
    for stock_idx, stock_data in enumerate(test_data):
        if len(stock_data) < required_length:
            continue
            
        # 为每只股票生成所有可能的时间窗口样本
        for start_idx in range(len(stock_data) - required_length + 1):
            input_seq = stock_data[start_idx:start_idx + context_length]
            target_seq = stock_data[start_idx + context_length:start_idx + required_length]
            
            start_price = input_seq[-1, 3]  # 当前收盘价
            end_price = target_seq[-1, 3]   # 3天后收盘价
            
            if start_price == 0:
                continue
                
            cumulative_return = (end_price - start_price) / start_price
            
            # 二分类标签：上涨为1，不上涨为0
            if cumulative_return >= DataConfig.UPRISE_THRESHOLD:
                target = 1.0  # 上涨
            else:
                target = 0.0  # 不上涨
                
            all_possible_samples.append((input_seq, target, stock_idx, start_idx))
    
    print(f"总共可用样本: {len(all_possible_samples)} 个")
    
    # 随机选择固定的评估样本
    if len(all_possible_samples) < num_samples:
        print(f"警告: 可用样本数 ({len(all_possible_samples)}) 少于请求的样本数 ({num_samples})")
        selected_samples = all_possible_samples
    else:
        # 使用固定的随机种子选择样本，确保每次选择相同的样本
        selected_samples = random.sample(all_possible_samples, num_samples)
    
    # 按股票索引和时间索引排序，确保顺序一致
    selected_samples.sort(key=lambda x: (x[2], x[3]))  # 按股票索引和时间索引排序
    
    # 分离输入和标签
    for input_seq, target, stock_idx, start_idx in selected_samples:
        eval_inputs.append(input_seq)
        eval_targets.append(target)
    
    eval_inputs = np.array(eval_inputs)
    eval_targets = np.array(eval_targets)
    
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
    
    return eval_inputs, eval_targets

# 批量评估函数
def evaluate_model_batch(model, eval_inputs, eval_targets, device, batch_size=EvaluationConfig.EVAL_BATCH_SIZE):
    """
    使用批处理进行快速评估（二分类）
    返回: (score, total, class_correct, class_total, pred_positive_correct, pred_positive_total, auc_score)
    """
    model.eval()
    score = 0
    total = 0
    class_correct = [0, 0]  # [不上涨正确数, 上涨正确数]
    class_total = [0, 0]    # [不上涨总数, 上涨总数]
    
    # 新增：预测统计
    pred_positive_correct = 0  # 预测上涨且正确的数量
    pred_positive_total = 0    # 预测上涨的总数量
    
    # 新增：用于AUC计算的列表
    all_probabilities = []
    all_targets = []
    
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # 批量处理
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                      dtype=torch.float32).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            
            # 批量推理
            batch_outputs = model(batch_inputs)  # [batch_size, 1]
            batch_probabilities = torch.sigmoid(batch_outputs).cpu().numpy().flatten()
            batch_predictions = (batch_probabilities > 0.5).astype(int)  # 概率>0.5预测为上涨
            
            # 收集所有概率和标签用于AUC计算
            all_probabilities.extend(batch_probabilities)
            all_targets.extend(batch_targets)
            
            # 批量计算得分
            for j in range(len(batch_targets)):
                target = int(batch_targets[j])
                prediction = batch_predictions[j]
                
                class_total[target] += 1
                
                # 统计预测上涨的情况
                if prediction == 1:
                    pred_positive_total += 1
                    if target == 1:  # 预测上涨且实际上涨
                        pred_positive_correct += 1
                
                # 应用评分规则
                if prediction == target:
                    score += EvaluationConfig.CORRECT_PREDICTION_SCORE
                    class_correct[target] += 1
                elif target == 0 and prediction == 1:  # 假阳性：预测上涨但实际不上涨
                    score += EvaluationConfig.FALSE_POSITIVE_PENALTY
                elif target == 1 and prediction == 0:  # 假阴性：预测不上涨但实际上涨
                    score += EvaluationConfig.FALSE_NEGATIVE_PENALTY
                
                total += 1
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_targets, all_probabilities)
    except ValueError:
        # 如果所有标签都是同一类，AUC无法计算
        auc_score = 0.5  # 随机分类器的AUC
    
    return score, total, class_correct, class_total, pred_positive_correct, pred_positive_total, auc_score

def calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=EvaluationConfig.EVAL_BATCH_SIZE):
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
            
            # 批量处理
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                      dtype=torch.float32).to(device)
            batch_targets = torch.tensor(eval_targets[start_idx:end_idx], 
                                       dtype=torch.float32).to(device)
            
            # 计算损失
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
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
            # 获取单个样本
            sample_input = torch.tensor(eval_inputs[idx:idx+1], dtype=torch.float32).to(device)
            true_label = eval_targets[idx]
            
            # 模型预测
            model_output = model(sample_input)
            raw_output = model_output.cpu().item()
            probability = torch.sigmoid(model_output).cpu().item()
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
def precompute_training_dataset(train_data, train_stock_info, train_weights, 
                               batch_size, batches_per_epoch, seed=None):
    """
    预计算每轮训练所需的训练数据集
    自动根据批大小和批数量计算需要的样本数
    返回: (epoch_inputs, epoch_targets)
    """
    samples_per_epoch = batch_size * batches_per_epoch
    
    if seed is not None:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        random.seed(seed)
    
    epoch_inputs = []
    epoch_targets = []
    
    # 直接生成所有需要的样本
    epoch_inputs, epoch_targets = generate_batch_samples_improved(
        train_data, train_stock_info, train_weights, samples_per_epoch)
    
    return np.array(epoch_inputs), np.array(epoch_targets)

# 改进的训练函数
def train_model(model, train_data, test_data, train_stock_info, train_weights, epochs=TrainingConfig.EPOCHS, 
               learning_rate=TrainingConfig.LEARNING_RATE, device=None, 
               batch_size=TrainingConfig.BATCH_SIZE, batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH):
    """
    使用预计算训练数据集和固定评估集的训练函数
    提高训练效率，确保评估的一致性
    """
    # 设置训练随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)
    
    # 创建固定的评估数据集（训练开始前创建一次）
    eval_inputs, eval_targets = create_fixed_evaluation_dataset(test_data, num_samples=DataConfig.EVAL_SAMPLES)
    
    # 使用动态加权BCE损失函数，根据每轮训练数据的正负样本比例动态调整权重
    criterion = DynamicWeightedBCE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=TrainingConfig.SCHEDULER_STEP_SIZE, gamma=TrainingConfig.SCHEDULER_GAMMA)
    
    best_score = float('-inf')  # 改用得分而不是准确率
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段
        print(f'Epoch {epoch + 1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 预计算当前轮次的训练数据
        epoch_seed = DataConfig.RANDOM_SEED + epoch  # 每轮使用不同的种子确保数据多样性
        epoch_inputs, epoch_targets = precompute_training_dataset(
            train_data, train_stock_info, train_weights, batch_size, batches_per_epoch, epoch_seed)
        
        # 根据本轮训练数据的正负样本比例动态更新损失函数权重
        criterion.update_weights(epoch_targets)
        
        # 打印本轮权重信息
        positive_count = np.sum(epoch_targets == 1)
        negative_count = np.sum(epoch_targets == 0)
        total_count = len(epoch_targets)
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        negative_ratio = negative_count / total_count if total_count > 0 else 0
        
        print(f'  本轮数据分布: 正样本={positive_count}({positive_ratio:.1%}), 负样本={negative_count}({negative_ratio:.1%})')
        print(f'  动态权重: 正样本权重={criterion.positive_weight.item():.3f}, 负样本权重={criterion.negative_weight.item():.3f}')
        
        # 将预计算的数据转换为tensor并移到设备上
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.float32).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.float32).to(device)
        
        # 训练循环：使用预计算的数据
        for step in range(batches_per_epoch):
            start_idx = step * batch_size
            end_idx = min((step + 1) * batch_size, len(epoch_inputs_tensor))
            
            # 从预计算的数据中取一个batch
            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            
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
        scheduler.step()
        
        # 固定评估集评估
        score, total, class_correct, class_total, pred_positive_correct, pred_positive_total, auc_score = evaluate_model_batch(
            model, eval_inputs, eval_targets, device, batch_size=EvaluationConfig.EVAL_BATCH_SIZE
        )
        
        # 计算测试集损失
        test_loss = calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=EvaluationConfig.EVAL_BATCH_SIZE)
        
        # 随机挑选10组样本打印模型输出值
        print_sample_predictions(model, eval_inputs, eval_targets, device, num_samples=10, epoch=epoch+1)
        
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
            print(f'  上涨准确率: {pred_positive_correct}/{pred_positive_total} = {precision:.3f}')
        else:
            print(f'  上涨准确率: 0/0 = 0.000 (无预测上涨)')
        
        overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
        avg_score = score / total if total > 0 else 0
        
        print(f'  总体准确率: {overall_acc:.3f}')
        print(f'  评估得分: {score} / {total} = {avg_score:.3f}')
        print(f'  AUC得分: {auc_score:.4f}')
        print(f'  测试集损失: {test_loss:.4f}')
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), ModelSaveConfig.get_best_model_path())
            print(f'  ✓ 发现更好的模型！得分提升到: {score}')
        
        print("-" * 50)

if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 打印配置摘要
    print_config_summary()
    
    # 获取设备信息
    device = DeviceConfig.print_device_info()

    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    
    # 使用改进的数据加载函数
    print("正在加载和预处理数据...")
    train_data, test_data, train_stock_info, test_stock_info = load_and_preprocess_data()
    print(f"训练数据: {len(train_data)} 只股票")
    print(f"测试数据: {len(test_data)} 只股票")

    # 计算股票选择权重
    train_weights = calculate_stock_weights(train_stock_info)
    test_weights = calculate_stock_weights(test_stock_info)
    
    # 打印权重信息
    print("\n股票采样权重信息:")
    data_lengths = [info['data_length'] for info in train_stock_info]
    print(f"训练股票数据长度统计:")
    print(f"  最小长度: {min(data_lengths)}")
    print(f"  最大长度: {max(data_lengths)}")
    print(f"  平均长度: {np.mean(data_lengths):.1f}")
    print(f"  权重范围: {min(train_weights):.3f} - {max(train_weights):.3f}")
    
    # 显示一些样本的权重
    print(f"\n前5只股票的权重示例:")
    for i in range(min(5, len(train_stock_info))):
        info = train_stock_info[i]
        weight = train_weights[i]
        print(f"  {info['file_name']}: 数据长度={info['data_length']}, 权重={weight:.3f}, 2021年起始位置={info['year_2021_start']}")

    print("正在创建 Transformer 模型...")
    model = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM, 
        d_model=ModelConfig.D_MODEL, 
        nhead=ModelConfig.NHEAD, 
        num_layers=ModelConfig.NUM_LAYERS, 
        output_dim=ModelConfig.OUTPUT_DIM,
        max_seq_len=ModelConfig.MAX_SEQ_LEN,
        decay_factor=ModelConfig.DECAY_FACTOR
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    print("开始训练...")
    # 使用带固定评估集的训练函数
    train_model(model, train_data, test_data, train_stock_info, train_weights, device=device)
    
    # 保存最终模型
    final_model_path = ModelSaveConfig.get_final_model_path(ModelConfig.D_MODEL)
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")
    print(f"最佳模型已保存到: {ModelSaveConfig.get_best_model_path()}")