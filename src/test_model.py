import os
os.environ["TORCH_FORCE_FLASH_ATTENTION"] = "0"
import torch, numpy as np
from train import EnhancedStockTransformer, load_and_preprocess_data, evaluate_model_batch, create_fixed_evaluation_dataset, calculate_test_loss, WeightedFocalLoss
from config import (ModelConfig, DataConfig, EvaluationConfig, 
                   DeviceConfig, ModelSaveConfig)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取设备
    device = DeviceConfig.get_device()

    # 加载数据
    train_data, test_data = load_and_preprocess_data()

    # 创建模型
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

    # 创建固定评估数据集
    eval_inputs, eval_targets = create_fixed_evaluation_dataset(test_data, num_samples=DataConfig.EVAL_SAMPLES)
    
    # 创建损失函数
    criterion = WeightedFocalLoss(
        positive_weight=1.5,
        negative_weight=1.0,
        gamma=2.5,
        alpha=1.0
    )
    
    # 评估模型
    score, total, class_correct, class_total, pred_positive_correct, pred_positive_total = evaluate_model_batch(model, eval_inputs, eval_targets, device)
    
    # 计算测试集损失
    test_loss = calculate_test_loss(model, eval_inputs, eval_targets, criterion, device)
    
    # 打印结果
    class_names = ['不上涨', '上涨']
    print(f"\n模型评估结果:")
    for i in range(2):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f'  {class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.3f}')
    
    overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
    avg_score = score / total if total > 0 else 0
    print(f'  总体准确率: {overall_acc:.3f}')
    print(f'  评估得分: {score} / {total} = {avg_score:.3f}')
    print(f'  测试集损失: {test_loss:.4f}')