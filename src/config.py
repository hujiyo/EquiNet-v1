"""
EquiNet 模型配置文件
统一管理模型参数、训练参数和评估参数
"""

import torch

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""

    # 基础模型参数
    INPUT_DIM = 8                    # 输入特征维度数（OHLCV + 市值相关特征）
    D_MODEL = 128                    # 模型维度（更高的维度通常能捕获更复杂的模式）
    NHEAD = 8                        # 注意力头数（会被分配给不同类型的专业化头）
    NUM_LAYERS = 4                   # Transformer层数
    OUTPUT_DIM = 3                   # 输出类别数（上涨/下跌/震荡）
    MAX_SEQ_LEN = 60                 # 最大序列长度
    DECAY_FACTOR = 0.1               # 时间衰减因子

    # 位置编码参数
    POSITIONAL_ENCODING_DECAY = 0.1  # 位置编码的时间衰减因子

    # 注意力机制参数
    DROPOUT_RATE = 0.1               # Dropout比率
    ATTENTION_DROPOUT = 0.1          # 注意力Dropout比率

    # 专业化注意力头分配
    PRICE_HEADS = 2                  # 价格趋势头数量
    VOLUME_HEADS = 2                 # 成交量头数量
    VOLATILITY_HEADS = 2             # 波动率头数量
    PATTERN_HEADS = 2                # 综合模式头数量

    # 多尺度注意力参数
    SHORT_TERM_WINDOW = 10           # 短期窗口大小
    MEDIUM_TERM_WINDOW = 30          # 中期窗口大小
    LONG_TERM_DECAY = 0.1            # 长期衰减因子

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""

    # 基础训练参数
    EPOCHS = 80                      # 训练轮数
    LEARNING_RATE = 0.001            # 初始学习率
    BATCH_SIZE = 50                  # GPU每次并行训练的样本数
    BATCHES_PER_EPOCH = 20           # 每轮训练的批次数

    # 优化器参数
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # 学习率调度器参数
    SCHEDULER_STEP_SIZE = 10         # 学习率调度步长
    SCHEDULER_GAMMA = 0.5            # 学习率衰减因子

    # 损失函数参数
    FOCAL_LOSS_ALPHA = [1.5, 2.0, 1.0]  # Focal Loss类别权重
    FOCAL_LOSS_GAMMA = 2.0               # Focal Loss聚焦参数

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

    # 类别阈值
    UPRISE_THRESHOLD = 0.03          # 大涨阈值（3%）
    DOWNFALL_THRESHOLD = -0.02       # 大跌阈值（-2%）

    # 评估参数
    EVAL_SAMPLES = 1000              # 评估样本数量
    EVAL_BATCH_SIZE = 100            # 评估批处理大小

# ==================== 评估参数 ====================
class EvaluationConfig:
    """评估相关参数"""

    # 评分规则
    CORRECT_PREDICTION_SCORE = 1     # 预测正确得分
    UPRISE_PREDICTED_DOWNFALL = -1   # 上涨预测为下跌扣分
    DOWNFALL_PREDICTED_UPRISE = -2   # 下跌预测为上涨扣分

    # 评估设置
    EVAL_SAMPLES = 1000              # 评估样本数量
    EVAL_BATCH_SIZE = 100            # 评估批处理大小

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