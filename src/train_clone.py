'''
克隆模型训练脚本 v3

核心思想：
- 前10轮：只训练模型A（原始标签）
- 第10轮：克隆模型A为模型B（完全独立的参数）
- 第10轮起：
  - 模型A继续用原始标签训练
  - 模型B用A的高置信预测作为伪标签训练：
    - A预测 > 0.7 → B的标签 = 1（伪正标签）
    - A预测 < 0.2 → B的标签 = 0（伪负标签）
    - 其它 → 保持原始标签不变

这样模型B学习的是A"确信"的模式，同时过滤掉A不确定的噪声样本
'''

import os, torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np
import random
import math
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import (ModelConfig, TrainingConfig, DataConfig,
                   DeviceConfig, ModelSaveConfig,
                   print_config_summary)

from train import (
    WarmupScheduler, RMSNorm, PositionalEncoding, 
    MultiHeadAttention, TransformerLayer,
    load_and_preprocess_data, calculate_stock_weights,
    create_fixed_evaluation_dataset, precompute_training_dataset,
    EnhancedStockTransformer
)


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns, 
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name=""):
    """
    模型评估函数
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_returns = []
    
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                       dtype=torch.bfloat16).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            batch_returns = eval_cumulative_returns[start_idx:end_idx]
            
            preds = torch.sigmoid(model(batch_inputs))
            
            all_preds.extend(preds.float().cpu().numpy().flatten())
            all_targets.extend(batch_targets)
            all_returns.extend(batch_returns)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)
    
    # 计算 AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5
    
    # 计算 Top N% 收益
    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(all_preds) * percent / 100))
    
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]
    
    # Top K 的最低置信度阈值（用于实盘时判断是否入选Top1%）
    top_threshold = all_preds[sorted_indices[top_k - 1]]
    
    # 统计高置信样本
    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2
    
    stats = {
        'auc': auc,
        'top_return': np.mean(top_returns),
        'top_count': top_k,
        'top_threshold': top_threshold,  # Top1%的最低置信度
        'high_conf_count': np.sum(high_conf),
        'low_conf_count': np.sum(low_conf),
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
    }
    
    return stats


def train_clone_model(model_a, train_stock_info, test_stock_info, train_weights, 
                      epochs=TrainingConfig.EPOCHS, 
                      learning_rate=TrainingConfig.LEARNING_RATE, 
                      device=None, 
                      batch_size=TrainingConfig.BATCH_SIZE, 
                      batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
                      clone_epoch=10,
                      pseudo_pos_ratio=0.01,
                      pseudo_neg_ratio=0.05):
    """
    克隆模型训练函数 v5
    
    训练策略：
    - 前 clone_epoch 轮：只训练模型A
    - 第 clone_epoch 轮：克隆模型A为模型B
    - 之后：A继续原始训练，B用A的高置信预测作为伪标签
      - 按比例选取：A预测值前 pseudo_pos_ratio (1%) 的样本 → 伪正标签
      - 按比例选取：A预测值倒数 pseudo_neg_ratio (5%) 的样本 → 伪负标签
      - 其它样本 → 保持原始标签不变
    """
    print("\n" + "="*60)
    print("克隆模型训练 v5")
    print("="*60)
    print(f"训练策略：")
    print(f"  - 前{clone_epoch}轮：只训练模型A（原始标签）")
    print(f"  - 第{clone_epoch}轮：克隆模型A为模型B")
    print(f"  - 之后：")
    print(f"    - 模型A：继续用原始标签训练")
    print(f"    - 模型B：用A的高置信预测作为伪标签")
    print(f"      - A预测值前{pseudo_pos_ratio*100:.0f}% → 伪正标签")
    print(f"      - A预测值倒数{pseudo_neg_ratio*100:.0f}% → 伪负标签")
    print(f"      - 其它 → 保持原始标签不变")
    print("="*60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)
    
    # 创建评估数据集
    eval_inputs, eval_targets, eval_cumulative_returns = create_fixed_evaluation_dataset(test_stock_info)
    
    # 模型B初始化为None
    model_b = None
    optimizer_b = None
    
    # 模型A的优化器
    optimizer_a = optim.Adam(model_a.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # 学习率调度（模型A）
    warmup_scheduler_a = WarmupScheduler(
        optimizer_a, 
        warmup_epochs=TrainingConfig.WARMUP_EPOCHS,
        target_lr=learning_rate,
        start_lr=TrainingConfig.WARMUP_START_LR
    )
    
    for param_group in optimizer_a.param_groups:
        param_group['lr'] = learning_rate
    
    total_main_epochs = epochs - TrainingConfig.WARMUP_EPOCHS
    main_scheduler_a = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, 
        T_max=total_main_epochs,
        eta_min=TrainingConfig.COSINE_ETA_MIN
    )
    
    for param_group in optimizer_a.param_groups:
        param_group['lr'] = TrainingConfig.WARMUP_START_LR
    
    # 损失函数
    def bce_loss(pred, target):
        pred = pred.squeeze()
        eps = 1e-7
        pred_clamp = torch.clamp(pred, eps, 1 - eps)
        return (-target * torch.log(pred_clamp) - (1 - target) * torch.log(1 - pred_clamp)).mean()
    
    best_auc_a = 0.0
    best_auc_b = 0.0
    best_model_a_state = None
    best_model_b_state = None
    best_epoch_a = 0
    best_epoch_b = 0
    
    # 最佳模型A缓存（按Top1%收益率判断）
    # 用于给模型B生成伪标签，避免当前A过拟合影响B
    best_return_a = -float('inf')
    best_model_a_for_pseudo = None  # 用于生成伪标签的最佳A
    best_return_epoch_a = 0
    
    # 按收益率保存的最佳模型（真正要用的模型）
    best_return_b = -float('inf')
    best_return_epoch_b = 0
    best_model_a_by_return = None  # 按收益率保存的模型A
    best_model_b_by_return = None  # 按收益率保存的模型B
    best_auc_a_at_best_return = 0.0  # 最佳收益时的AUC
    best_auc_b_at_best_return = 0.0
    best_threshold_a = 0.0  # 最佳收益时的Top1%阈值
    best_threshold_b = 0.0
    
    # 早停机制：测试集损失或收益率任一刷新都算，连续patience轮无刷新则停止
    patience = 15
    no_improve_count = 0
    best_loss_a = float('inf')  # 用于早停的最佳损失
    
    for epoch in range(epochs):
        model_a.train()
        if model_b is not None:
            model_b.train()
        
        total_loss_a = 0
        total_loss_b = 0
        total_pseudo_pos = 0
        total_pseudo_neg = 0
        total_unchanged = 0
        
        # 是否已经克隆了模型B
        has_model_b = (epoch + 1) >= clone_epoch
        
        # 学习率更新
        if warmup_scheduler_a.is_warmup_phase():
            current_lr = warmup_scheduler_a.step(epoch)
            lr_status = f"预热阶段 ({epoch + 1}/{TrainingConfig.WARMUP_EPOCHS})"
        else:
            current_lr = main_scheduler_a.get_last_lr()[0]
            lr_status = "正常训练"
        
        status = "A+B训练" if has_model_b else "只训练A"
        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status}) [{status}]')
        
        # 第clone_epoch轮时克隆模型B
        if (epoch + 1) == clone_epoch and model_b is None:
            print(f"\n  >>> 第{clone_epoch}轮：克隆模型A为模型B <<<")
            model_b = copy.deepcopy(model_a)
            model_b = model_b.to(device)
            
            # 模型B的优化器（从头开始，使用较小的学习率）
            optimizer_b = optim.Adam(model_b.parameters(), lr=learning_rate * 0.5, 
                                     weight_decay=TrainingConfig.WEIGHT_DECAY)
            print(f"  模型B已创建，参数数: {sum(p.numel() for p in model_b.parameters()):,}")
            print()
        
        # 预计算训练数据
        epoch_seed = DataConfig.RANDOM_SEED + epoch
        epoch_inputs, epoch_targets = precompute_training_dataset(
            train_stock_info, train_weights, batch_size, batches_per_epoch, epoch_seed)
        
        # 打印标签分布（软标签：1.0=上涨, 0.4=边界, 0.0=不涨）
        count_positive = np.sum(epoch_targets >= 0.9)  # 1.0
        count_boundary = np.sum((epoch_targets > 0.1) & (epoch_targets < 0.9))  # 0.4
        count_negative = np.sum(epoch_targets <= 0.1)  # 0.0
        total_count = len(epoch_targets)
        print(f'  标签分布: 上涨={count_positive}({count_positive/total_count:.1%}), 边界={count_boundary}({count_boundary/total_count:.1%}), 不涨={count_negative}({count_negative/total_count:.1%})')
        
        # 转换为tensor
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.bfloat16).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.bfloat16).to(device)
        
        # 训练循环
        for step in range(batches_per_epoch):
            start_idx = step * batch_size
            end_idx = min((step + 1) * batch_size, len(epoch_inputs_tensor))
            
            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            
            # ========== 训练模型A ==========
            optimizer_a.zero_grad()
            pred_a = torch.sigmoid(model_a(batch_inputs))
            loss_a = bce_loss(pred_a, batch_targets)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer_a.step()
            total_loss_a += loss_a.item()
            
            # ========== 训练模型B（如果存在）==========
            if model_b is not None and best_model_a_for_pseudo is not None:
                optimizer_b.zero_grad()
                
                # 用【最佳模型A】的预测生成伪标签（而非当前A）
                # 这样即使当前A过拟合，B仍然能用之前表现最好的A来纠偏
                with torch.no_grad():
                    pred_a_for_pseudo = torch.sigmoid(best_model_a_for_pseudo(batch_inputs)).squeeze()
                
                # 生成伪标签（按比例选取前pseudo_pos_ratio作为伪正，倒数pseudo_neg_ratio作为伪负）
                pseudo_targets = batch_targets.clone()
                
                # 计算伪正阈值：取前pseudo_pos_ratio的预测值
                k_pos = max(1, int(len(pred_a_for_pseudo) * pseudo_pos_ratio))
                threshold_pos = torch.topk(pred_a_for_pseudo, k_pos).values[-1]
                
                # 计算伪负阈值：取倒数pseudo_neg_ratio的预测值
                k_neg = max(1, int(len(pred_a_for_pseudo) * pseudo_neg_ratio))
                threshold_neg = torch.topk(pred_a_for_pseudo, k_neg, largest=False).values[-1]
                
                # A预测值 >= threshold_pos → 伪正标签
                high_mask = pred_a_for_pseudo >= threshold_pos
                pseudo_targets[high_mask] = 1.0
                
                # A预测值 <= threshold_neg → 伪负标签
                low_mask = pred_a_for_pseudo <= threshold_neg
                pseudo_targets[low_mask] = 0.0
                
                # 统计
                total_pseudo_pos += high_mask.sum().item()
                total_pseudo_neg += low_mask.sum().item()
                total_unchanged += (~high_mask & ~low_mask).sum().item()
                
                # 训练模型B
                pred_b = torch.sigmoid(model_b(batch_inputs))
                loss_b = bce_loss(pred_b, pseudo_targets)
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
                optimizer_b.step()
                total_loss_b += loss_b.item()
            
            # 进度显示
            progress = (step + 1) / batches_per_epoch * 100
            avg_loss_a = total_loss_a / (step + 1)
            if model_b is not None:
                avg_loss_b = total_loss_b / (step + 1)
                print(f'\r  训练进度: {progress:.1f}%, Loss_A: {avg_loss_a:.4f}, Loss_B: {avg_loss_b:.4f}', end='', flush=True)
            else:
                print(f'\r  训练进度: {progress:.1f}%, Loss_A: {avg_loss_a:.4f}', end='', flush=True)
        
        print()
        print()
        
        # 清理内存
        del epoch_inputs_tensor, epoch_targets_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 更新学习率
        if not warmup_scheduler_a.is_warmup_phase():
            main_scheduler_a.step()
        
        # 评估模型A
        stats_a = evaluate_model(model_a, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="A")
        
        avg_loss_a = total_loss_a / batches_per_epoch
        
        # 打印模型A结果
        print(f'  [模型A] 损失: {avg_loss_a:.4f}, AUC: {stats_a["auc"]:.4f}')
        print(f'          预测均值: {stats_a["pred_mean"]:.3f}, 高置信(>0.7): {stats_a["high_conf_count"]}, 低置信(<0.2): {stats_a["low_conf_count"]}')
        print(f'          Top{DataConfig.TOP_PERCENT}%收益: {stats_a["top_return"]*100:+.2f}%')
        
        # 早停检测：损失或收益率任一刷新都算
        improved = False
        
        # 检查损失是否改善
        if avg_loss_a < best_loss_a:
            best_loss_a = avg_loss_a
            improved = True
            print(f'          ✓ 损失改善: {best_loss_a:.4f}')
        
        # 保存最佳模型A（按AUC）
        if stats_a['auc'] > best_auc_a:
            best_auc_a = stats_a['auc']
            best_epoch_a = epoch + 1
            best_model_a_state = copy.deepcopy(model_a.state_dict())
            improved = True
            print(f'          ✓ 新最佳模型A（AUC）！AUC: {best_auc_a:.4f}')
        
        # 更新用于生成伪标签的最佳模型A（按Top1%收益率判断）
        if stats_a['top_return'] > best_return_a:
            best_return_a = stats_a['top_return']
            best_return_epoch_a = epoch + 1
            # 创建一个独立的模型副本用于生成伪标签
            if best_model_a_for_pseudo is None:
                best_model_a_for_pseudo = copy.deepcopy(model_a)
            else:
                best_model_a_for_pseudo.load_state_dict(copy.deepcopy(model_a.state_dict()))
            best_model_a_for_pseudo.eval()  # 设为eval模式
            improved = True
            print(f'          ✓ 新最佳模型A（收益率）！Top1%收益: {best_return_a*100:+.2f}% (第{best_return_epoch_a}轮)')
            # 同时保存按收益率的模型A权重
            best_model_a_by_return = copy.deepcopy(model_a.state_dict())
            best_auc_a_at_best_return = stats_a['auc']
            best_threshold_a = stats_a['top_threshold']
        
        # 更新早停计数器
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f'          ⚠ 无改善 ({no_improve_count}/{patience})')
        
        # 评估模型B（如果存在）
        if model_b is not None:
            stats_b = evaluate_model(model_b, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="B")
            
            avg_loss_b = total_loss_b / batches_per_epoch if total_loss_b > 0 else 0
            
            print(f'  [模型B] 损失: {avg_loss_b:.4f}, AUC: {stats_b["auc"]:.4f}')
            print(f'          预测均值: {stats_b["pred_mean"]:.3f}, 高置信(>0.7): {stats_b["high_conf_count"]}, 低置信(<0.2): {stats_b["low_conf_count"]}')
            print(f'          Top{DataConfig.TOP_PERCENT}%收益: {stats_b["top_return"]*100:+.2f}%')
            print(f'          伪标签来源: 最佳A(第{best_return_epoch_a}轮, 收益{best_return_a*100:+.2f}%)')
            print(f'          伪标签统计: 伪正={total_pseudo_pos}, 伪负={total_pseudo_neg}, 不变={total_unchanged}')
            
            # 保存最佳模型B（按AUC）
            if stats_b['auc'] > best_auc_b:
                best_auc_b = stats_b['auc']
                best_epoch_b = epoch + 1
                best_model_b_state = copy.deepcopy(model_b.state_dict())
                print(f'          ✓ 新最佳模型B（AUC）！AUC: {best_auc_b:.4f}')
            
            # 保存最佳模型B（按收益率）- 这才是真正重要的！
            if stats_b['top_return'] > best_return_b:
                best_return_b = stats_b['top_return']
                best_return_epoch_b = epoch + 1
                best_model_b_by_return = copy.deepcopy(model_b.state_dict())
                best_auc_b_at_best_return = stats_b['auc']
                best_threshold_b = stats_b['top_threshold']
                print(f'          ✓ 新最佳模型B（收益率）！Top1%收益: {best_return_b*100:+.2f}% (第{best_return_epoch_b}轮)')
        
        print("-" * 60)
        
        # 早停检查
        if no_improve_count >= patience:
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break
    
    # 保存最佳模型（按收益率保存，文件名包含详细信息）
    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"最佳模型A（按AUC）: 第{best_epoch_a}轮, AUC: {best_auc_a:.4f}")
    print(f"最佳模型A（按收益率）: 第{best_return_epoch_a}轮, Top1%收益: {best_return_a*100:+.2f}%")
    if best_model_b_state is not None:
        print(f"最佳模型B（按AUC）: 第{best_epoch_b}轮, AUC: {best_auc_b:.4f}")
        print(f"最佳模型B（按收益率）: 第{best_return_epoch_b}轮, Top1%收益: {best_return_b*100:+.2f}%")
    
    # 生成带详细信息的文件名
    # 格式: model_top{k}_{return}pct_thr{threshold}_auc{auc}_ep{epoch}_{MMDD_HHMM}.pth
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # 保存模型A（按收益率）
    if best_model_a_by_return is not None:
        return_str = f"{best_return_a*100:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
        thr_str = f"{best_threshold_a:.3f}".replace('.', '_')
        auc_str = f"{best_auc_a_at_best_return:.4f}".replace('.', '_')
        filename_a = f"modelA_top{DataConfig.TOP_PERCENT}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{best_return_epoch_a}_{timestamp}.pth"
        save_path_a = os.path.join(DataConfig.OUTPUT_DIR, filename_a)
        torch.save(best_model_a_by_return, save_path_a)
        print(f"✓ 模型A（按收益率）已保存: {filename_a}")
        print(f"  Top1%阈值: {best_threshold_a:.4f} (预测值≥此值即入选Top1%)")
    
    # 保存模型B（按收益率）- 这是最重要的！
    if best_model_b_by_return is not None:
        return_str = f"{best_return_b*100:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
        thr_str = f"{best_threshold_b:.3f}".replace('.', '_')
        auc_str = f"{best_auc_b_at_best_return:.4f}".replace('.', '_')
        filename_b = f"modelB_top{DataConfig.TOP_PERCENT}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{best_return_epoch_b}_{timestamp}.pth"
        save_path_b = os.path.join(DataConfig.OUTPUT_DIR, filename_b)
        torch.save(best_model_b_by_return, save_path_b)
        print(f"✓ 模型B（按收益率）已保存: {filename_b}")
        print(f"  Top1%阈值: {best_threshold_b:.4f} (预测值≥此值即入选Top1%)")
    
    print("=" * 60)
    
    return best_auc_a, best_auc_b, best_return_a, best_return_b


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 打印配置摘要
    print_config_summary()
    
    # 获取设备
    device = DeviceConfig.print_device_info()
    
    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()
    
    # 计算权重
    train_weights = calculate_stock_weights(train_stock_info)
    
    # 打印数据集统计
    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    print("="*60)
    
    # 创建模型A（使用原始的EnhancedStockTransformer）
    print("\n正在创建模型A (BF16精度)...")
    model_a = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM, 
        d_model=ModelConfig.D_MODEL, 
        nhead=ModelConfig.NHEAD, 
        num_layers=ModelConfig.NUM_LAYERS, 
        output_dim=ModelConfig.OUTPUT_DIM, 
        max_seq_len=DataConfig.CONTEXT_LENGTH
    ).to(device)
    
    # 转换为BF16
    model_a = model_a.to(dtype=torch.bfloat16)
    
    # 打印参数数量
    total_params = sum(p.numel() for p in model_a.parameters())
    print(f"模型A参数数: {total_params:,}")
    
    # 开始训练
    print("\n开始克隆模型训练...")
    best_auc_a, best_auc_b, best_return_a, best_return_b = train_clone_model(
        model_a, train_stock_info, test_stock_info, train_weights, 
        device=device,
        clone_epoch=10,
        pseudo_pos_ratio=0.01,  # 前1%作为伪正标签
        pseudo_neg_ratio=0.05   # 倒数5%作为伪负标签
    )
    
    print(f"\n最终结果:")
    print(f"  模型A: 最佳AUC={best_auc_a:.4f}, 最佳Top1%收益={best_return_a*100:+.2f}%")
    print(f"  模型B: 最佳AUC={best_auc_b:.4f}, 最佳Top1%收益={best_return_b*100:+.2f}%")
