import os
os.environ["TORCH_FORCE_FLASH_ATTENTION"] = "0"
import torch, numpy as np
from train import EnhancedStockTransformer, load_and_preprocess_data, generate_single_sample
from config import (ModelConfig, DataConfig, EvaluationConfig, 
                   DeviceConfig, ModelSaveConfig)

def evaluate(model, data, device, num_samples=EvaluationConfig.EVAL_SAMPLES):
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_samples):
        try:
            input_seq, target = generate_single_sample(data)
        except Exception:
            continue
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
            prediction = torch.argmax(output, dim=1).item()
        if prediction == target:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, total

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取设备
    device = DeviceConfig.get_device()

    # 加载数据
    train_data, test_data = load_and_preprocess_data()

    # 创建模型（使用配置文件中的参数）
    model = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        nhead=ModelConfig.NHEAD,
        num_layers=ModelConfig.NUM_LAYERS,
        output_dim=ModelConfig.OUTPUT_DIM,
        max_seq_len=ModelConfig.MAX_SEQ_LEN,
        decay_factor=ModelConfig.DECAY_FACTOR
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(ModelSaveConfig.get_best_model_path(), map_location=device))

    # 评估
    acc_train, n_train = evaluate(model, train_data, device)
    acc_test, n_test = evaluate(model, test_data, device)
    acc_all, n_all = evaluate(model, train_data + test_data, device)

    print(f"训练集抽测{EvaluationConfig.EVAL_SAMPLES}组准确率: {acc_train*100:.2f}% (有效样本数: {n_train})")
    print(f"测试集抽测{EvaluationConfig.EVAL_SAMPLES}组准确率: {acc_test*100:.2f}% (有效样本数: {n_test})")
    print(f"全部数据抽测{EvaluationConfig.EVAL_SAMPLES}组准确率: {acc_all*100:.2f}% (有效样本数: {n_all})")