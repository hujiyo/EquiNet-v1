import os
os.environ["TORCH_FORCE_FLASH_ATTENTION"] = "0"
import torch
from train import (EnhancedStockTransformer, load_and_preprocess_data, 
                   evaluate_model_batch, create_fixed_evaluation_dataset, 
                   calculate_test_loss, DynamicWeightedBCE, 
                   calculate_stock_weights)
from config import (ModelConfig, DataConfig, EvaluationConfig, 
                   DeviceConfig, ModelSaveConfig)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取设备
    device = DeviceConfig.get_device()
    print(f"使用设备: {device}")

    # 加载数据（注意：load_and_preprocess_data 返回4个值）
    print("\n正在加载数据...")
    train_data, test_data, train_stock_info, test_stock_info = load_and_preprocess_data()
    print(f"训练数据: {len(train_data)} 只股票")
    print(f"测试数据: {len(test_data)} 只股票")

    # 创建模型
    print("\n正在创建模型...")
    model = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        nhead=ModelConfig.NHEAD,
        num_layers=ModelConfig.NUM_LAYERS,
        output_dim=ModelConfig.OUTPUT_DIM,
        max_seq_len=ModelConfig.MAX_SEQ_LEN
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 加载模型权重
    print(f"\n正在加载模型权重: {ModelSaveConfig.get_best_model_path()}")
    model.load_state_dict(torch.load(ModelSaveConfig.get_best_model_path(), map_location=device))
    print("模型加载成功！")

    # 创建固定评估数据集（注意：create_fixed_evaluation_dataset 返回3个值）
    print("\n正在创建评估数据集...")
    eval_inputs, eval_targets, eval_cumulative_returns = create_fixed_evaluation_dataset(
        test_data, num_samples=DataConfig.EVAL_SAMPLES
    )
    
    # 创建损失函数（使用 train.py 中的 DynamicWeightedBCE）
    criterion = DynamicWeightedBCE()
    criterion.update_weights(eval_targets)  # 根据评估数据更新权重
    
    # 评估模型（注意：evaluate_model_batch 返回8个值）
    print("\n正在评估模型...")
    score, total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score = evaluate_model_batch(
        model, eval_inputs, eval_targets, eval_cumulative_returns, device
    )
    
    # 计算测试集损失
    test_loss = calculate_test_loss(model, eval_inputs, eval_targets, criterion, device)
    
    # 打印详细结果
    print("\n" + "=" * 50)
    print("模型评估结果")
    print("=" * 50)
    
    class_names = ['不上涨', '上涨']
    for i in range(2):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.3f}')
        else:
            print(f'{class_names[i]}: 0/0 = 0.000 (无样本)')
    
    # 计算上涨准确率（预测上涨后真上涨的概率）
    if pred_positive_total > 0:
        precision = pred_positive_correct / pred_positive_total
        non_negative_rate = pred_non_negative / pred_positive_total
        print(f'上涨准确率: {pred_positive_correct}/{pred_positive_total} = {precision:.3f}')
        print(f'非负准确率: {pred_non_negative}/{pred_positive_total} = {non_negative_rate:.3f}')
    else:
        print(f'上涨准确率: 0/0 = 0.000 (无预测上涨)')
    
    overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
    avg_score = score / total if total > 0 else 0
    
    print(f'总体准确率: {overall_acc:.3f}')
    print(f'评估得分: {score} / {total} = {avg_score:.3f}')
    print(f'AUC得分: {auc_score:.4f}')
    print(f'测试集损失: {test_loss:.4f}')
    print("=" * 50)