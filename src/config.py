"""
EquiNet 模型配置文件
统一管理模型参数、训练参数和评估参数
"""

import torch

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""
    # 基础模型参数
    INPUT_DIM = 5                    # 输入特征维度数（OHLCV）
    D_MODEL = 64                     # 模型维度（从128降到48，实验证明性能更好）
    NHEAD = 4                        # 注意力头数（从4降到3，匹配更小的模型）
    NUM_LAYERS = 4                   # Transformer层数
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1之间）
    MAX_SEQ_LEN = 60                 # 最大序列长度

    # 注意力机制参数（为小模型调整）
    DROPOUT_RATE = 0.2               # Dropout比率（从0.3降到0.2，小模型需要更少正则化）
    ATTENTION_DROPOUT = 0.1          # 注意力Dropout比率（从0.2降到0.1）

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""

    # 基础训练参数（优化训练策略）
    EPOCHS = 30                     # 训练轮数（增加轮数以充分训练小模型）
    LEARNING_RATE = 0.001            # 初始学习率（提高学习率）

    # 训练批处理
    BATCH_SIZE = 128                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 40           # 每轮训练的批次数（减少批次数）

    # 优化器参数
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # 学习率调度器参数
    SCHEDULER_STEP_SIZE = 10         # 学习率调度步长
    SCHEDULER_GAMMA = 0.5            # 学习率衰减因子
    
    # 余弦退火调度器参数
    USE_COSINE_ANNEALING = True      # 是否使用余弦退火调度器
    COSINE_T_MAX = EPOCHS - 5        # 余弦退火周期（总轮数-预热轮数）
    COSINE_ETA_MIN = 1e-6            # 余弦退火最小学习率
    
    # 自适应学习率参数
    PATIENCE = 3                     # 性能不提升的容忍轮数
    LR_REDUCE_FACTOR = 0.7           # 学习率衰减因子
    MIN_LR = 1e-7                    # 最小学习率
    
    # 学习率预热参数
    WARMUP_EPOCHS = 5                # 预热轮数（前5轮逐步达到最高学习率）
    WARMUP_START_LR = 1e-4           # 预热起始学习率（提高起始值，减少过于保守的预热）

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
    # 新评分规则（基于预测上涨和实际涨跌幅）
    UPRISE_CORRECT_HIGH_SCORE = 1      # 预测上涨且实际上涨≥2%
    UPRISE_CORRECT_LOW_SCORE = 0.5      # 预测上涨且实际涨0-2%
    UPRISE_FALSE_SMALL_PENALTY = -1     # 预测上涨但实际下跌<2%
    UPRISE_FALSE_LARGE_PENALTY = -2     # 预测上涨但实际下跌≥2%
    # 其余情况（预测不上涨）：不改变分数

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
    print(f"  预热轮数: {TrainingConfig.WARMUP_EPOCHS}")
    print(f"  预热起始学习率: {TrainingConfig.WARMUP_START_LR}")
    
    print(f"\n学习率调度:")
    if TrainingConfig.USE_COSINE_ANNEALING:
        print(f"  调度策略: 余弦退火")
        print(f"  退火周期: {TrainingConfig.COSINE_T_MAX}轮")
        print(f"  最小学习率: {TrainingConfig.COSINE_ETA_MIN}")
    else:
        print(f"  调度策略: 阶梯衰减")
        print(f"  衰减步长: {TrainingConfig.SCHEDULER_STEP_SIZE}轮")
        print(f"  衰减因子: {TrainingConfig.SCHEDULER_GAMMA}")
    print(f"  自适应调整: 容忍{TrainingConfig.PATIENCE}轮, 衰减因子{TrainingConfig.LR_REDUCE_FACTOR}")
    print(f"  最小学习率: {TrainingConfig.MIN_LR}")

    print(f"\n数据参数:")
    print(f"  数据目录: {DataConfig.DATA_DIR}")
    print(f"  测试集比例: {DataConfig.TEST_RATIO}")
    print(f"  上下文长度: {DataConfig.CONTEXT_LENGTH}")
    print(f"  未来天数: {DataConfig.FUTURE_DAYS}")

    print(f"\n评估参数:")
    print(f"  评估样本数: {DataConfig.EVAL_SAMPLES}")
    print(f"  评估批处理大小: {DataConfig.EVAL_BATCH_SIZE}")

    print("=" * 50)

if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary() 