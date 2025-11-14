"""
today_analysis.py - 综合模型分析工具

功能：
1. 对out文件夹下的所有模型使用data_new/下的数据进行测评
2. 对每个模型使用today/下的数据进行预测
3. 生成综合对比表格，展示：
   - 每个模型在不同置信度区间的表现
   - 每个模型对today数据的预测结果
"""

import os
import torch
import pandas as pd
import numpy as np
from train import (EnhancedStockTransformer, load_and_preprocess_data, 
                   evaluate_model_batch, create_fixed_evaluation_dataset, 
                   calculate_test_loss, DynamicWeightedBCE)
from config import (ModelConfig, DataConfig, DeviceConfig)

def evaluate_model_on_data_new(model_path, device, data_dir='./data_new'):
    """
    使用data_new下的数据评估单个模型
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    print(f"\n正在评估模型: {os.path.basename(model_path)}")
    print(f"数据目录: {data_dir}")
    
    # 加载模型
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
    
    # 加载data_new数据
    original_data_dir = DataConfig.DATA_DIR
    DataConfig.DATA_DIR = data_dir
    
    try:
        train_stock_info, test_stock_info = load_and_preprocess_data(data_dir=data_dir)
        
        # 创建评估数据集（使用滚动窗口标准化）
        eval_inputs, eval_targets, eval_cumulative_returns = create_fixed_evaluation_dataset(
            test_stock_info, num_samples=DataConfig.EVAL_SAMPLES
        )
        
        # 评估模型
        score, total, class_correct, class_total, pred_positive_correct, pred_positive_total, \
        pred_non_negative, auc_score, confidence_stats, score_count = evaluate_model_batch(
            model, eval_inputs, eval_targets, eval_cumulative_returns, device
        )
        
        # 计算测试集损失
        criterion = DynamicWeightedBCE()
        criterion.update_weights(eval_targets)
        test_loss = calculate_test_loss(model, eval_inputs, eval_targets, criterion, device)
        
        # 整理结果
        results = {
            'model_name': os.path.basename(model_path),
            'auc': auc_score,
            'test_loss': test_loss,
            'overall_acc': sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0,
            'score_count': score_count,
            'cumulative_return': score,
            'avg_return': score / score_count if score_count > 0 else 0,
            'confidence_stats': confidence_stats,
            # 详细的置信度区间统计
            'conf_0.80-0.90': {
                'count': confidence_stats['0.80-0.90'][1],
                'correct': confidence_stats['0.80-0.90'][0],
                'non_negative': confidence_stats['0.80-0.90'][2],
                'precision': confidence_stats['0.80-0.90'][0] / confidence_stats['0.80-0.90'][1] if confidence_stats['0.80-0.90'][1] > 0 else 0,
                'non_neg_rate': confidence_stats['0.80-0.90'][2] / confidence_stats['0.80-0.90'][1] if confidence_stats['0.80-0.90'][1] > 0 else 0,
            },
            'conf_0.90-0.93': {
                'count': confidence_stats['0.90-0.93'][1],
                'correct': confidence_stats['0.90-0.93'][0],
                'non_negative': confidence_stats['0.90-0.93'][2],
                'precision': confidence_stats['0.90-0.93'][0] / confidence_stats['0.90-0.93'][1] if confidence_stats['0.90-0.93'][1] > 0 else 0,
                'non_neg_rate': confidence_stats['0.90-0.93'][2] / confidence_stats['0.90-0.93'][1] if confidence_stats['0.90-0.93'][1] > 0 else 0,
            },
            'conf_0.93-0.96': {
                'count': confidence_stats['0.93-0.96'][1],
                'correct': confidence_stats['0.93-0.96'][0],
                'non_negative': confidence_stats['0.93-0.96'][2],
                'precision': confidence_stats['0.93-0.96'][0] / confidence_stats['0.93-0.96'][1] if confidence_stats['0.93-0.96'][1] > 0 else 0,
                'non_neg_rate': confidence_stats['0.93-0.96'][2] / confidence_stats['0.93-0.96'][1] if confidence_stats['0.93-0.96'][1] > 0 else 0,
            },
            'conf_0.96-1.00': {
                'count': confidence_stats['0.96-1.00'][1],
                'correct': confidence_stats['0.96-1.00'][0],
                'non_negative': confidence_stats['0.96-1.00'][2],
                'precision': confidence_stats['0.96-1.00'][0] / confidence_stats['0.96-1.00'][1] if confidence_stats['0.96-1.00'][1] > 0 else 0,
                'non_neg_rate': confidence_stats['0.96-1.00'][2] / confidence_stats['0.96-1.00'][1] if confidence_stats['0.96-1.00'][1] > 0 else 0,
            }
        }
        
        return results
        
    finally:
        # 恢复原始数据目录
        DataConfig.DATA_DIR = original_data_dir

def predict_today_stocks(model_path, device, today_dir='./today'):
    """
    使用单个模型对today文件夹下的股票进行预测
    
    Returns:
        dict: 包含预测结果的字典，按置信度区间分类
    """
    print(f"\n正在使用模型预测today股票: {os.path.basename(model_path)}")
    
    # 获取所有股票文件
    if not os.path.exists(today_dir):
        print(f"⚠ 警告：{today_dir} 文件夹不存在")
        return {}
    
    stock_files = sorted([f for f in os.listdir(today_dir) if f.endswith('.xlsx')])
    
    if len(stock_files) == 0:
        print(f"⚠ 警告：{today_dir} 文件夹中没有xlsx文件")
        return {}
    
    # 准备数据字典
    stock_files_data = {}
    
    for stock_file in stock_files:
        try:
            file_path = os.path.join(today_dir, stock_file)
            
            # 读取股票数据
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # 提取OHLCV数据
            data = df[['start', 'max', 'min', 'end', 'volume']].values
            
            # 检查数据长度是否足够
            if len(data) < DataConfig.CONTEXT_LENGTH + 1:
                continue
            
            stock_files_data[stock_file] = data
            
        except Exception as e:
            print(f"处理 {stock_file} 时出错: {e}")
            continue
    
    # 使用统一预测函数
    from train import predict_multiple_stocks
    raw_predictions = predict_multiple_stocks(model_path, stock_files_data, device)
    
    # 按置信度区间分类存储预测结果
    predictions = {
        '0.80-0.90': [],
        '0.90-0.93': [],
        '0.93-0.96': [],
        '0.96-1.00': [],
        'all': raw_predictions  # 所有预测
    }
    
    # 按置信度区间分类
    for stock_file, probability in raw_predictions:
        if 0.80 <= probability < 0.90:
            predictions['0.80-0.90'].append((stock_file, probability))
        elif 0.90 <= probability < 0.93:
            predictions['0.90-0.93'].append((stock_file, probability))
        elif 0.93 <= probability < 0.96:
            predictions['0.93-0.96'].append((stock_file, probability))
        elif 0.96 <= probability <= 1.00:
            predictions['0.96-1.00'].append((stock_file, probability))
    
    # 对每个区间的预测按置信度排序
    for key in predictions:
        predictions[key].sort(key=lambda x: x[1], reverse=True)
    
    return predictions

def generate_comparison_table(all_results):
    """
    生成综合对比表格
    
    Args:
        all_results: list of dict, 每个dict包含一个模型的完整评估和预测结果
    """
    print("\n" + "=" * 120)
    print("综合模型分析报告")
    print("=" * 120)
    
    # 过滤出有评估结果的模型
    results_with_eval = [r for r in all_results if r['evaluation'] is not None]
    
    # 1. 模型性能对比表
    if results_with_eval:
        print("\n【第一部分：模型性能对比 - data_new测评结果】")
        print("=" * 120)
        
        # 表头
        header = f"{'模型名称':<30} {'AUC':<8} {'Loss':<8} {'总准确率':<10}"
        for interval in ['0.80-0.90', '0.90-0.93', '0.93-0.96', '0.96-1.00']:
            header += f" {interval}预测数/准确率/非负率"
        print(header)
        print("-" * 120)
        
        # 每个模型一行
        for result in results_with_eval:
            eval_results = result['evaluation']
            row = f"{eval_results['model_name']:<30} "
            row += f"{eval_results['auc']:<8.4f} "
            row += f"{eval_results['test_loss']:<8.4f} "
            row += f"{eval_results['overall_acc']:<10.3f} "
            
            for interval in ['0.80-0.90', '0.90-0.93', '0.93-0.96', '0.96-1.00']:
                conf_key = f'conf_{interval}'
                stats = eval_results[conf_key]
                row += f" {stats['count']:>3}/{stats['precision']:>5.3f}/{stats['non_neg_rate']:>5.3f} "
            
            print(row)
        
        print("-" * 120)
    else:
        print("\n【第一部分：模型性能对比 - data_new测评结果】")
        print("=" * 120)
        print("⚠ 无评估结果")
        print("-" * 120)
    
    # 2. 收益评估表
    if results_with_eval:
        print("\n【第二部分：收益评估 (置信度≥0.9)】")
        print("=" * 80)
        print(f"{'模型名称':<30} {'参与预测数':<12} {'累计收益率':<15} {'平均收益率':<15}")
        print("-" * 80)
        
        for result in results_with_eval:
            eval_results = result['evaluation']
            print(f"{eval_results['model_name']:<30} "
                  f"{eval_results['score_count']:<12} "
                  f"{eval_results['cumulative_return']*100:<15.2f}% "
                  f"{eval_results['avg_return']*100:<15.3f}%")
        
        print("-" * 80)
    else:
        print("\n【第二部分：收益评估 (置信度≥0.9)】")
        print("=" * 80)
        print("⚠ 无评估结果")
        print("-" * 80)
    
    # 3. today预测结果对比表
    results_with_pred = [r for r in all_results if r['predictions'] is not None]
    
    if results_with_pred:
        print("\n【第三部分：Today股票预测结果对比】")
        print("=" * 120)
        
        # 为每个置信度区间生成一个子表
        for interval in ['0.96-1.00', '0.93-0.96', '0.90-0.93', '0.80-0.90']:
            print(f"\n置信度区间: {interval}")
            print("-" * 120)
            
            # 收集所有模型在该区间的预测
            interval_predictions = {}
            for result in results_with_pred:
                # 获取模型名称（优先从evaluation，如果没有则从model_path）
                if result['evaluation'] is not None:
                    model_name = result['evaluation']['model_name']
                else:
                    model_name = os.path.basename(result['model_path'])
                predictions = result['predictions'][interval]
                interval_predictions[model_name] = predictions
            
            if not any(interval_predictions.values()):
                print(f"  该区间暂无预测")
                continue
            
            # 获取所有被预测的股票
            all_stocks = set()
            for predictions in interval_predictions.values():
                for stock_file, _ in predictions:
                    all_stocks.add(stock_file)
            
            all_stocks = sorted(all_stocks)
            
            # 打印表头（所有模型名称）
            header = f"{'股票名称':<20}"
            for model_name in interval_predictions.keys():
                # 截断模型名称以适应显示
                short_name = model_name[:18] if len(model_name) > 18 else model_name
                header += f" {short_name:<20}"
            print(header)
            print("-" * 120)
            
            # 打印每只股票的预测
            for stock in all_stocks:
                row = f"{stock:<20}"
                for model_name, predictions in interval_predictions.items():
                    # 查找该股票的预测
                    stock_pred = next((prob for s, prob in predictions if s == stock), None)
                    if stock_pred is not None:
                        row += f" {stock_pred:<20.4f}"
                    else:
                        row += f" {'-':<20}"
                print(row)
    else:
        print("\n【第三部分：Today股票预测结果对比】")
        print("=" * 120)
        print("⚠ 无预测结果")
        print("-" * 120)
    
    # 4. 详细预测统计
    if results_with_pred:
        print("\n【第四部分：Today预测统计】")
        print("=" * 80)
        print(f"{'模型名称':<30} {'0.80-0.90':<12} {'0.90-0.93':<12} {'0.93-0.96':<12} {'0.96-1.00':<12} {'总计':<10}")
        print("-" * 80)
        
        for result in results_with_pred:
            # 获取模型名称
            if result['evaluation'] is not None:
                model_name = result['evaluation']['model_name']
            else:
                model_name = os.path.basename(result['model_path'])
            predictions = result['predictions']
            row = f"{model_name:<30}"
            row += f" {len(predictions['0.80-0.90']):<12}"
            row += f" {len(predictions['0.90-0.93']):<12}"
            row += f" {len(predictions['0.93-0.96']):<12}"
            row += f" {len(predictions['0.96-1.00']):<12}"
            row += f" {len(predictions['all']):<10}"
            print(row)
        
        print("-" * 80)
    else:
        print("\n【第四部分：Today预测统计】")
        print("=" * 80)
        print("⚠ 无预测结果")
        print("-" * 80)
    
    # 5. 高置信度预测详情（≥0.9）
    if results_with_pred:
        print("\n【第五部分：高置信度预测详情 (≥0.9)】")
        print("=" * 120)
        
        has_high_conf = False
        for result in results_with_pred:
            # 获取模型名称
            if result['evaluation'] is not None:
                model_name = result['evaluation']['model_name']
            else:
                model_name = os.path.basename(result['model_path'])
            predictions = result['predictions']
            
            # 合并≥0.9的所有预测
            high_conf_predictions = []
            high_conf_predictions.extend(predictions['0.90-0.93'])
            high_conf_predictions.extend(predictions['0.93-0.96'])
            high_conf_predictions.extend(predictions['0.96-1.00'])
            high_conf_predictions.sort(key=lambda x: x[1], reverse=True)
            
            if len(high_conf_predictions) == 0:
                continue
            
            has_high_conf = True
            print(f"\n模型: {model_name}")
            print(f"  共 {len(high_conf_predictions)} 只股票")
            
            # 分类显示
            buy_list = [(s, p) for s, p in high_conf_predictions if p >= 0.9]
            caution_list = [(s, p) for s, p in high_conf_predictions if 0.8 <= p < 0.9]
            
            if buy_list:
                print(f"  建议购买 (≥0.9): {len(buy_list)} 只")
                for stock, prob in buy_list[:10]:  # 只显示前10只
                    status = "建议购买" if prob >= 0.9 else ""
                    print(f"    {stock:<20} >> {prob:.4f} {status}")
                if len(buy_list) > 10:
                    print(f"    ... (还有 {len(buy_list)-10} 只)")
        
        if not has_high_conf:
            print("⚠ 无高置信度预测")
    else:
        print("\n【第五部分：高置信度预测详情 (≥0.9)】")
        print("=" * 120)
        print("⚠ 无预测结果")
    
    print("\n" + "=" * 120)
    print("分析完成！")
    print("=" * 120)

def main():
    """主函数"""
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取设备
    device = DeviceConfig.get_device()
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 获取所有模型文件
    out_dir = './out'
    if not os.path.exists(out_dir):
        print(f"错误：{out_dir} 文件夹不存在")
        return
    
    model_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.pth')])
    
    if len(model_files) == 0:
        print(f"错误：{out_dir} 文件夹中没有模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for i, f in enumerate(model_files, 1):
        file_size = os.path.getsize(os.path.join(out_dir, f)) / (1024 * 1024)
        print(f"  {i}. {f} ({file_size:.2f} MB)")
    
    # 询问用户是否分析所有模型
    print("\n是否分析所有模型？")
    print("1. 是 - 分析所有模型")
    print("2. 否 - 选择特定模型")
    choice = input("请输入选择 (直接回车默认为1): ").strip()
    
    selected_models = []
    if choice == '' or choice == '1':
        selected_models = [os.path.join(out_dir, f) for f in model_files]
    else:
        print("\n请输入要分析的模型编号（用逗号分隔，如: 1,2,3）：")
        indices = input().strip().split(',')
        try:
            for idx in indices:
                idx = int(idx.strip())
                if 1 <= idx <= len(model_files):
                    selected_models.append(os.path.join(out_dir, model_files[idx-1]))
        except:
            print("输入格式错误，将分析所有模型")
            selected_models = [os.path.join(out_dir, f) for f in model_files]
    
    if len(selected_models) == 0:
        print("未选择任何模型")
        return
    
    print(f"\n将分析 {len(selected_models)} 个模型")
    print("=" * 80)
    
    # 检查data_new目录
    data_new_dir = './data_new'
    if not os.path.exists(data_new_dir):
        print(f"⚠ 警告：{data_new_dir} 文件夹不存在")
        # 检查是否有其他数据文件夹可用
        alternative_dirs = ['./data', './data_old']
        available_dirs = [d for d in alternative_dirs if os.path.exists(d)]
        
        if available_dirs:
            print(f"发现可用的数据文件夹: {', '.join(available_dirs)}")
            print("请选择要使用的数据文件夹进行评估：")
            for i, d in enumerate(available_dirs, 1):
                print(f"  {i}. {d}")
            print(f"  0. 跳过评估")
            
            choice = input("请输入选择 (直接回车跳过评估): ").strip()
            if choice and choice != '0':
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(available_dirs):
                        data_new_dir = available_dirs[idx-1]
                        print(f"将使用 {data_new_dir} 进行评估")
                        skip_evaluation = False
                    else:
                        print("无效选择，将跳过评估")
                        skip_evaluation = True
                except:
                    print("无效输入，将跳过评估")
                    skip_evaluation = True
            else:
                skip_evaluation = True
        else:
            print("未找到任何可用的数据文件夹，将跳过模型评估")
            skip_evaluation = True
    else:
        skip_evaluation = False
    
    # 检查today目录
    today_dir = './today'
    if not os.path.exists(today_dir):
        print(f"⚠ 警告：{today_dir} 文件夹不存在，将跳过today预测")
        skip_prediction = True
    else:
        skip_prediction = False
    
    if skip_evaluation and skip_prediction:
        print("错误：data_new 和 today 文件夹都不存在，无法进行分析")
        return
    
    # 对每个模型进行评估和预测
    all_results = []
    
    for i, model_path in enumerate(selected_models, 1):
        print(f"\n{'='*80}")
        print(f"处理模型 {i}/{len(selected_models)}: {os.path.basename(model_path)}")
        print(f"{'='*80}")
        
        result = {
            'model_path': model_path,
            'evaluation': None,
            'predictions': None
        }        
        try:
            # 1. 评估模型（使用data_new）
            if not skip_evaluation:
                result['evaluation'] = evaluate_model_on_data_new(model_path, device, data_new_dir)
                print(f"✓ 评估完成")
            
            # 2. 预测today股票
            if not skip_prediction:
                result['predictions'] = predict_today_stocks(model_path, device, today_dir)
                print(f"✓ 预测完成")
            
            # 只有成功获取到至少一项结果时才添加
            if result['evaluation'] is not None or result['predictions'] is not None:
                all_results.append(result)
            
        except Exception as e:
            print(f"✗ 处理模型时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成综合对比表格
    if len(all_results) > 0:
        generate_comparison_table(all_results)
    else:
        print("\n没有成功处理任何模型")

if __name__ == "__main__":
    main()

