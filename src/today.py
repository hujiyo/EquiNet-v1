"""
today.py - 对today文件夹下的所有股票进行预测

功能：读取today文件夹下的所有xlsx文件（每个文件代表一只股票），
     使用每个股票的最新历史数据（由CONTEXT_LENGTH配置）作为输入，预测未来涨跌概率。
"""

import os
import pandas as pd
import numpy as np
from train import predict_multiple_stocks
from config import DataConfig, DeviceConfig, ModelSaveConfig

def load_and_predict_today_stocks(today_dir='./today', model_path=None, device=None):
    """
    加载today文件夹下的所有股票文件，对最新历史数据进行预测
    
    Args:
        today_dir: 股票数据目录
        model_path: 模型文件路径，如果为None则使用最佳模型
        device: 计算设备，如果为None则自动选择
        
    Returns:
        predictions: list of (filename, probability)
    """
    if device is None:
        device = DeviceConfig.get_device()
    
    if model_path is None:
        model_path = ModelSaveConfig.get_best_model_path()
    
    print(f"使用模型: {model_path}")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return []
    
    # 检查today目录是否存在
    if not os.path.exists(today_dir):
        print(f"错误：目录不存在 {today_dir}")
        return []
    
    # 获取所有股票文件并读取数据
    stock_files = sorted([f for f in os.listdir(today_dir) if f.endswith('.xlsx')])
    print(f"找到 {len(stock_files)} 个股票文件")
    
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
                print(f"{stock_file} >> 数据不足（需要至少{DataConfig.CONTEXT_LENGTH + 1}天，实际{len(data)}天）")
                continue
            
            stock_files_data[stock_file] = data
            
        except Exception as e:
            print(f"{stock_file} >> 读取失败: {e}")
            continue
    
    # 使用统一预测函数
    predictions = predict_multiple_stocks(model_path, stock_files_data, device)
    
    # 打印预测结果
    for filename, probability in predictions:
        print(f"{filename} >> 预测概率: {probability:.4f}")
    
    # 打印汇总信息
    print("\n" + "=" * 50)
    print("预测汇总")
    print("=" * 50)
    print(f"总共预测: {len(predictions)} 只股票")
    
    if len(predictions) > 0:
        # 统计不同置信度区间的股票数量
        high_confidence = [p for p in predictions if p[1] >= 0.9]
        cautious_confidence = [p for p in predictions if 0.8 <= p[1] < 0.9]
        medium_confidence = [p for p in predictions if 0.7 <= p[1] < 0.8]
        low_confidence = [p for p in predictions if p[1] < 0.7]
        
        print(f"高置信度 (≥0.9): {len(high_confidence)} 只 - 建议购买")
        print(f"谨慎置信度 (0.8-0.9): {len(cautious_confidence)} 只 - 谨慎买入")
        print(f"中置信度 (0.7-0.8): {len(medium_confidence)} 只")
        print(f"低置信度 (<0.7): {len(low_confidence)} 只")
        
        # 打印建议购买的股票列表
        if len(high_confidence) > 0:
            print(f"\n建议购买列表 (置信度≥0.9):")
            # 按置信度从高到低排序
            high_confidence.sort(key=lambda x: x[1], reverse=True)
            for stock_file, prob in high_confidence:
                print(f"  {stock_file}: {prob:.3f}")
        else:
            print(f"\n暂无建议购买的股票（置信度≥0.9）")
        
        # 打印谨慎买入的股票列表
        if len(cautious_confidence) > 0:
            print(f"\n谨慎买入列表 (置信度0.8-0.9):")
            # 按置信度从高到低排序
            cautious_confidence.sort(key=lambda x: x[1], reverse=True)
            for stock_file, prob in cautious_confidence:
                print(f"  {stock_file}: {prob:.3f}")
        else:
            print(f"\n暂无谨慎买入的股票（置信度0.8-0.9）")
    
    print("=" * 50)

if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 打印设备信息
    device = DeviceConfig.get_device()
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 列出可用的模型文件
    out_dir = './out'
    if os.path.exists(out_dir):
        model_files = [f for f in os.listdir(out_dir) if f.endswith('.pth')]
        if len(model_files) > 0:
            print("=" * 50)
            print("可用的模型文件:")
            print("=" * 50)
            model_files.sort()
            for i, model_file in enumerate(model_files, 1):
                # 获取文件大小
                file_size = os.path.getsize(os.path.join(out_dir, model_file)) / (1024 * 1024)  # MB
                print(f"{i}. {model_file} ({file_size:.2f} MB)")
            
            print(f"\n默认使用: {ModelSaveConfig.BEST_MODEL_NAME}")
            print("=" * 50)
            
            # 让用户选择模型
            choice = input("\n请选择模型编号（直接回车使用默认模型）: ").strip()
            
            if choice:
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_files):
                        selected_model = os.path.join(out_dir, model_files[choice_idx])
                        print(f"\n已选择模型: {model_files[choice_idx]}\n")
                        load_and_predict_today_stocks(model_path=selected_model, device=device)
                    else:
                        print(f"\n输入无效，使用默认模型\n")
                        load_and_predict_today_stocks(device=device)
                except ValueError:
                    print(f"\n输入无效，使用默认模型\n")
                    load_and_predict_today_stocks(device=device)
            else:
                print(f"\n使用默认模型\n")
                load_and_predict_today_stocks(device=device)
        else:
            print(f"警告: {out_dir} 文件夹中没有找到任何模型文件(.pth)\n")
            load_and_predict_today_stocks(device=device)
    else:
        print(f"警告: {out_dir} 文件夹不存在\n")
        load_and_predict_today_stocks(device=device)

