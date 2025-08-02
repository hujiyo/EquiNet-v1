"""
EquiNet 模型配置文件
统一管理模型参数、训练参数和评估参数
"""

import torch

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""

    # 基础模型参数（减少参数量防止过拟合）
    INPUT_DIM = 8                    # 输入特征维度数（OHLCV + 市值相关特征）
    D_MODEL = 128                     # 模型维度（减少维度防止过拟合）
    NHEAD = 4                        # 注意力头数（减少头数）
    NUM_LAYERS = 2                   # Transformer层数（减少层数）
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1之间）
    MAX_SEQ_LEN = 60                 # 最大序列长度
    DECAY_FACTOR = 0.1               # 时间衰减因子

    # 位置编码参数
    POSITIONAL_ENCODING_DECAY = 0.1  # 位置编码的时间衰减因子

    # 注意力机制参数
    DROPOUT_RATE = 0.3               # Dropout比率（增加防止过拟合）
    ATTENTION_DROPOUT = 0.2          # 注意力Dropout比率

    # 专业化注意力头分配（简化分配）
    PRICE_HEADS = 2                  # 价格趋势头数量（最重要）
    VOLUME_HEADS = 1                 # 成交量头数量（次重要）
    VOLATILITY_HEADS = 1             # 波动率头数量
    PATTERN_HEADS = 0                # 综合模式头数量（移除，但需要处理）

    # 简化多尺度注意力参数（使用更简单有效的方法）
    ATTENTION_WINDOW_SIZE = 20       # 注意力窗口大小（关注最近20天）
    TEMPORAL_DECAY = 0.05            # 时间衰减因子（更温和的衰减）

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""

    # 基础训练参数（优化训练策略）
    EPOCHS = 20                     # 训练轮数（增加轮数以充分训练小模型）
    LEARNING_RATE = 0.002            # 初始学习率（提高学习率）
    BATCH_SIZE = 128                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 15           # 每轮训练的批次数（减少批次数）

    # 优化器参数
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # 学习率调度器参数
    SCHEDULER_STEP_SIZE = 10         # 学习率调度步长
    SCHEDULER_GAMMA = 0.5            # 学习率衰减因子

    '''
    # 更保守预测
    POSITIVE_WEIGHT = 0.8
    NEGATIVE_WEIGHT = 2.5

    # 更积极预测  
    POSITIVE_WEIGHT = 1.2
    NEGATIVE_WEIGHT = 1.5
    '''
    # 加权Focal Loss参数（根据评分规则调整）
    POSITIVE_WEIGHT = 1.5                  # 正样本（上涨）权重（提高，因为正样本更少）
    NEGATIVE_WEIGHT = 1.0                  # 负样本（不上涨）权重（降低，平衡类别）
    FOCAL_LOSS_GAMMA = 2.5                # Focal Loss聚焦参数（增加聚焦强度）
    FOCAL_LOSS_ALPHA = 1.0                # 类别平衡参数（1.0表示不额外调整）

    # 动态权重调整参数
    DYNAMIC_WEIGHT_WINDOW_SIZE = 1000    # 动态权重调整窗口大小
    DYNAMIC_WEIGHT_MIN = 0.5             # 最小权重
    DYNAMIC_WEIGHT_MAX = 3.0             # 最大权重

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""

    # 数据路径
    DATA_DIR = './data'              # 数据目录
    OUTPUT_DIR = './out'             # 输出目录

    # 数据分割参数
    TEST_RATIO = 0.1                 # 测试集比例
    RANDOM_SEED = 42                 # 随机种子

    # 样本生成参数
    CONTEXT_LENGTH = 60              # 历史数据长度
    FUTURE_DAYS = 3                  # 未来预测天数
    REQUIRED_LENGTH = CONTEXT_LENGTH + FUTURE_DAYS  # 总需求长度

    # 二分类阈值（上涨/不上涨）
    UPRISE_THRESHOLD = 0.02          # 上涨阈值（2%，超过2%算上涨）
    DOWNFALL_THRESHOLD = -0.02       # 下跌阈值（-2%，低于-2%算下跌）

    # 评估参数
    EVAL_SAMPLES = 200               # 评估样本数量
    EVAL_BATCH_SIZE = 50             # 评估批处理大小

# ==================== 评估参数 ====================
class EvaluationConfig:
    """评估相关参数"""

    # 评分规则（二分类）
    CORRECT_PREDICTION_SCORE = 1     # 预测正确得分
    FALSE_POSITIVE_PENALTY = -1      # 假阳性惩罚（预测上涨但实际下跌）
    FALSE_NEGATIVE_PENALTY = -0.5    # 假阴性惩罚（预测下跌但实际上涨）

    # 评估设置
    EVAL_SAMPLES = 200               # 评估样本数量
    EVAL_BATCH_SIZE = 50             # 评估批处理大小

# ==================== 设备配置 ====================
class DeviceConfig:
    """设备相关配置"""

    @staticmethod
    def get_device():
        """获取训练设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def print_device_info():
        """打印设备信息"""
        device = DeviceConfig.get_device()
        if device.type == "cuda":
            print(f"使用 GPU 进行训练: {torch.cuda.get_device_name()}")
        else:
            print("CUDA 不可用，将使用 CPU 进行训练，训练速度可能较慢。")
        return device

# ==================== 模型保存配置 ====================
class ModelSaveConfig:
    """模型保存相关配置"""

    # 模型文件名
    BEST_MODEL_NAME = 'EnhancedEquiNet_focal_best.pth'
    FINAL_MODEL_NAME = 'EnhancedEquiNet_final.pth'

    @staticmethod
    def get_best_model_path():
        """获取最佳模型路径"""
        return f'./out/{ModelSaveConfig.BEST_MODEL_NAME}'

    @staticmethod
    def get_final_model_path(d_model):
        """获取最终模型路径"""
        return f'./out/EnhancedEquiNet_{d_model}.pth'

# ==================== 配置打印函数 ====================
def print_config_summary():
    """打印配置摘要"""
    print("=" * 50)
    print("EquiNet 模型配置摘要")
    print("=" * 50)

    print(f"模型架构:")
    print(f"  输入维度: {ModelConfig.INPUT_DIM}")
    print(f"  模型维度: {ModelConfig.D_MODEL}")
    print(f"  注意力头数: {ModelConfig.NHEAD}")
    print(f"  层数: {ModelConfig.NUM_LAYERS}")
    print(f"  输出维度: {ModelConfig.OUTPUT_DIM}")
    print(f"  序列长度: {ModelConfig.MAX_SEQ_LEN}")

    print(f"\n训练参数:")
    print(f"  训练轮数: {TrainingConfig.EPOCHS}")
    print(f"  学习率: {TrainingConfig.LEARNING_RATE}")
    print(f"  批处理大小: {TrainingConfig.BATCH_SIZE}")
    print(f"  每轮批次数: {TrainingConfig.BATCHES_PER_EPOCH}")

    print(f"\n数据参数:")
    print(f"  数据目录: {DataConfig.DATA_DIR}")
    print(f"  测试集比例: {DataConfig.TEST_RATIO}")
    print(f"  上下文长度: {DataConfig.CONTEXT_LENGTH}")
    print(f"  未来天数: {DataConfig.FUTURE_DAYS}")

    print(f"\n评估参数:")
    print(f"  评估样本数: {EvaluationConfig.EVAL_SAMPLES}")
    print(f"  评估批处理大小: {EvaluationConfig.EVAL_BATCH_SIZE}")

    print("=" * 50)

if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary() 