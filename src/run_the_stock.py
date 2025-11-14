"""
run_the_stock.py - 单股票历史预测验证工具

功能：
1. 输入股票代码，找到对应的xlsx文件
2. 输入日期（如2025/01/17），以该日期为分水岭
3. 使用out_stable下的4个模型预测未来3天
4. 打印实际涨跌情况
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from train import EnhancedStockTransformer
from config import ModelConfig, DataConfig, DeviceConfig

def find_stock_file(stock_code, data_dirs=['./today', './data', './data_new']):
    """
    根据股票代码查找对应的xlsx文件
    
    Args:
        stock_code: 股票代码（如 000034 或 000034.xlsx）
        data_dirs: 要搜索的目录列表
        
    Returns:
        str: 文件完整路径，如果找不到返回None
    """
    # 标准化股票代码
    if not stock_code.endswith('.xlsx'):
        stock_code = stock_code + '.xlsx'
    
    # 在所有目录中搜索
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        
        file_path = os.path.join(data_dir, stock_code)
        if os.path.exists(file_path):
            return file_path
    
    return None

def load_stock_data(file_path):
    """
    加载股票数据
    
    Returns:
        tuple: (df, data, time_column)
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 检查必需的列
    required_columns = ['time', 'start', 'max', 'min', 'end', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必需的列: {col}")
    
    # 转换时间列 - 自动识别格式，不指定format参数
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # 删除时间列中的无效值（NaT）
    original_len = len(df)
    df = df.dropna(subset=['time']).reset_index(drop=True)
    if original_len > len(df):
        print(f"  ⚠ 已移除 {original_len - len(df)} 行无效日期数据")
    
    if len(df) == 0:
        raise ValueError("数据文件中没有有效的日期数据，请检查time列的格式")
    
    # 按时间排序（确保从旧到新）
    df = df.sort_values('time', ascending=True).reset_index(drop=True)
    
    # 提取OHLCV数据
    data = df[['start', 'max', 'min', 'end', 'volume']].values
    time_column = df['time'].values
    
    return df, data, time_column

def find_date_index(time_column, target_date):
    """
    查找目标日期在时间列中的索引
    
    Args:
        time_column: 时间列（numpy array of datetime64）
        target_date: 目标日期字符串（如 "2025/01/17" 或 "2025-01-17"）
        
    Returns:
        int: 索引位置，如果找不到返回-1
    """
    try:
        # 自动识别日期格式
        target = pd.to_datetime(target_date)
    except:
        return -1
    
    # 查找精确匹配
    for i, t in enumerate(time_column):
        if pd.Timestamp(t).date() == target.date():
            return i
    
    return -1

def predict_with_model(model_path, input_data, device):
    """
    使用单个模型进行预测（简化版，直接调用统一预测函数）
    
    Args:
        model_path: 模型文件路径
        input_data: 输入数据（标准化后的OHLCV，长度由CONTEXT_LENGTH配置）
        device: 计算设备
        
    Returns:
        float: 预测概率
    """
    # 注意：input_data已经是标准化后的数据，需要构造原始数据格式
    # 由于统一预测函数内部会重新标准化，这里需要特殊处理
    # 为了保持兼容性，这里保持原有实现
    
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
    
    # 预测
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.bfloat16).unsqueeze(0).to(device)
        output = model(input_tensor)
        probability = torch.sigmoid(output).float().cpu().item()
    
    return probability

def calculate_actual_return(data, start_idx, future_days=3):
    """
    计算实际收益率
    
    Args:
        data: 原始OHLCV数据
        start_idx: 起始索引（预测日的索引）
        future_days: 未来天数
        
    Returns:
        tuple: (actual_return, start_price, end_price)
    """
    if start_idx + future_days >= len(data):
        return None, None, None
    
    start_price = data[start_idx, 3]  # 当前收盘价
    end_price = data[start_idx + future_days, 3]  # 3天后收盘价
    
    if start_price == 0:
        return None, None, None
    
    actual_return = (end_price - start_price) / start_price
    
    return actual_return, start_price, end_price

def get_date_range_info(time_column, start_idx, context_length=None, future_days=None):
    """
    获取日期范围信息
    
    Returns:
        dict: 包含输入数据日期范围和预测目标日期范围的信息
    """
    # 使用配置文件中的默认值
    if context_length is None:
        context_length = DataConfig.CONTEXT_LENGTH
    if future_days is None:
        future_days = DataConfig.FUTURE_DAYS
    
    info = {}
    
    # 输入数据的日期范围（历史数据）
    if start_idx - context_length + 1 >= 0:
        input_start_date = pd.Timestamp(time_column[start_idx - context_length + 1]).strftime('%Y/%m/%d')
        input_end_date = pd.Timestamp(time_column[start_idx]).strftime('%Y/%m/%d')
        info['input_range'] = f"{input_start_date} 至 {input_end_date}"
    else:
        info['input_range'] = "数据不足"
    
    # 预测目标的日期范围（未来3天）
    if start_idx + future_days < len(time_column):
        predict_start_date = pd.Timestamp(time_column[start_idx]).strftime('%Y/%m/%d')
        predict_end_date = pd.Timestamp(time_column[start_idx + future_days]).strftime('%Y/%m/%d')
        info['predict_range'] = f"{predict_start_date} → {predict_end_date}"
        info['predict_end_date'] = predict_end_date
    else:
        info['predict_range'] = "未来数据不足"
        info['predict_end_date'] = "未知"
    
    return info

def print_day_by_day_details(df, start_idx, future_days=3):
    """
    打印逐日详细数据
    
    Args:
        df: 完整的DataFrame（包含time列）
        start_idx: 起始索引
        future_days: 未来天数
    """
    print(f"\n{'='*80}")
    print("逐日详细数据")
    print(f"{'='*80}")
    
    # 打印表头
    print(f"{'日期':<12} {'开盘':<10} {'最高':<10} {'最低':<10} {'收盘':<10} {'成交量':<12} {'涨跌幅':<10}")
    print("-" * 80)
    
    # 当前日（预测起点）
    current_row = df.iloc[start_idx]
    print(f"{pd.Timestamp(current_row['time']).strftime('%Y/%m/%d'):<12} "
          f"{current_row['start']:<10.2f} "
          f"{current_row['max']:<10.2f} "
          f"{current_row['min']:<10.2f} "
          f"{current_row['end']:<10.2f} "
          f"{int(current_row['volume']):<12} "
          f"{'(基准日)':<10}")
    
    base_price = current_row['end']
    
    # 未来每一天
    for i in range(1, future_days + 1):
        if start_idx + i >= len(df):
            print(f"第{i}天: 数据不足")
            continue
        
        row = df.iloc[start_idx + i]
        day_return = (row['end'] - base_price) / base_price * 100
        
        # 颜色标记（用符号表示）
        if day_return > 0:
            change_str = f"+{day_return:.2f}%"
        elif day_return < 0:
            change_str = f"{day_return:.2f}%"
        else:
            change_str = f"{day_return:.2f}%"
        
        print(f"{pd.Timestamp(row['time']).strftime('%Y/%m/%d'):<12} "
              f"{row['start']:<10.2f} "
              f"{row['max']:<10.2f} "
              f"{row['min']:<10.2f} "
              f"{row['end']:<10.2f} "
              f"{int(row['volume']):<12} "
              f"{change_str:<10}")
    
    print("-" * 80)

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
    
    # 检查out_stable目录
    model_dir = './out_stable'
    if not os.path.exists(model_dir):
        print(f"错误：{model_dir} 文件夹不存在")
        print("请确保out_stable文件夹存在并包含模型文件")
        return
    
    # 获取模型文件
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    
    if len(model_files) == 0:
        print(f"错误：{model_dir} 文件夹中没有模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型:")
    for i, f in enumerate(model_files, 1):
        print(f"  {i}. {f}")
    print()
    
    # 输入股票代码
    print("=" * 80)
    stock_code = input("请输入股票代码（如 000034 或 000034.xlsx）: ").strip()
    
    if not stock_code:
        print("错误：股票代码不能为空")
        return
    
    # 查找股票文件
    print(f"\n正在查找股票文件...")
    file_path = find_stock_file(stock_code)
    
    if file_path is None:
        print(f"错误：找不到股票 {stock_code} 的数据文件")
        print("已搜索目录: ./today, ./data, ./data_new")
        return
    
    print(f"✓ 找到文件: {file_path}")
    
    # 加载股票数据
    print(f"\n正在加载股票数据...")
    try:
        df, data, time_column = load_stock_data(file_path)
        print(f"✓ 数据加载成功")
        print(f"  数据长度: {len(data)} 天")
        
        # 安全地打印日期范围
        try:
            start_date = pd.Timestamp(time_column[0]).strftime('%Y/%m/%d')
            end_date = pd.Timestamp(time_column[-1]).strftime('%Y/%m/%d')
            print(f"  日期范围: {start_date} 至 {end_date}")
        except Exception as date_error:
            print(f"  ⚠ 无法显示日期范围: {date_error}")
    except Exception as e:
        print(f"错误：加载数据失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 输入目标日期
    print("\n" + "=" * 80)
    target_date = input("请输入目标日期（格式: 2025/01/17）: ").strip()
    
    if not target_date:
        print("错误：日期不能为空")
        return
    
    # 查找日期索引
    print(f"\n正在查找日期 {target_date}...")
    date_idx = find_date_index(time_column, target_date)
    
    if date_idx == -1:
        print(f"错误：找不到日期 {target_date}")
        print(f"可用日期范围: {pd.Timestamp(time_column[0]).strftime('%Y/%m/%d')} 至 {pd.Timestamp(time_column[-1]).strftime('%Y/%m/%d')}")
        return
    
    print(f"✓ 找到日期，索引位置: {date_idx}")
    
    # 检查是否有足够的历史数据
    context_length = DataConfig.CONTEXT_LENGTH
    if date_idx < context_length - 1:
        print(f"错误：历史数据不足（需要至少{context_length}天，当前只有{date_idx + 1}天）")
        return
    
    # 检查是否有未来数据（用于验证）
    future_days = DataConfig.FUTURE_DAYS  # 3天
    if date_idx + future_days >= len(data):
        print(f"警告：未来数据不足（需要{future_days}天，只有{len(data) - date_idx - 1}天），无法验证实际结果")
        has_future_data = False
    else:
        has_future_data = True
    
    # 获取日期范围信息
    date_info = get_date_range_info(time_column, date_idx, context_length, future_days)
    
    # 准备输入数据（历史数据）
    print(f"\n正在准备输入数据...")
    input_start_idx = date_idx - context_length + 1
    input_data_raw = data[input_start_idx:date_idx + 1]  # context_length天
    
    print(f"  输入数据范围: {date_info['input_range']}")
    print(f"  预测目标范围: {date_info['predict_range']}")
    
    # 🔑 修复：使用与训练时相同的滚动窗口标准化
    # 避免训练-预测不一致的问题
    input_data_normalized = np.zeros_like(input_data_raw, dtype=np.float64)
    
    if len(input_data_raw) < 2:
        print(f"错误：数据不足，无法进行滚动窗口标准化")
        return
    
    # 滚动窗口标准化：每天相对于前一天的涨跌幅
    valid_data = True
    for i in range(1, len(input_data_raw)):
        yesterday_close = input_data_raw[i-1, 3]  # 前一天的收盘价
        yesterday_volume = input_data_raw[i-1, 4]  # 前一天的成交量
        
        if yesterday_close == 0 or yesterday_volume == 0:
            print(f"错误：第{i}天数据异常（价格或成交量为0）")
            valid_data = False
            break
        
        # 价格特征：相对于前一天收盘价的涨跌幅
        input_data_normalized[i, :4] = (input_data_raw[i, :4] - yesterday_close) / yesterday_close
        # 成交量特征：相对于前一天成交量的变化比例
        input_data_normalized[i, 4] = (input_data_raw[i, 4] - yesterday_volume) / yesterday_volume
    
    if not valid_data:
        return
    
    # 只使用标准化后的数据（去掉第0天基准数据）
    input_data_normalized = input_data_normalized[1:]
    
    # 使用所有模型进行预测
    print(f"\n{'='*80}")
    print("模型预测结果")
    print(f"{'='*80}")
    
    predictions = []
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        try:
            print(f"\n正在使用模型: {model_file}")
            probability = predict_with_model(model_path, input_data_normalized, device)
            predictions.append((model_file, probability))
            
            # 判断预测结果
            if probability >= 0.9:
                suggestion = "建议购买"
            elif probability >= 0.8:
                suggestion = "谨慎买入"
            else:
                suggestion = ""
            
            print(f"  预测概率: {probability:.4f} {suggestion}")
            
        except Exception as e:
            print(f"  ✗ 预测失败: {e}")
            continue
    
    # 计算预测统计
    if len(predictions) > 0:
        avg_prob = np.mean([p[1] for p in predictions])
        max_prob = max([p[1] for p in predictions])
        min_prob = min([p[1] for p in predictions])
        std_prob = np.std([p[1] for p in predictions])
        
        print(f"\n{'='*80}")
        print("预测统计")
        print(f"{'='*80}")
        print(f"参与模型数: {len(predictions)}")
        print(f"平均概率: {avg_prob:.4f}")
        print(f"最高概率: {max_prob:.4f} ({[p[0] for p in predictions if p[1] == max_prob][0]})")
        print(f"最低概率: {min_prob:.4f} ({[p[0] for p in predictions if p[1] == min_prob][0]})")
        print(f"标准差: {std_prob:.4f}")
        
        # 综合建议
        if avg_prob >= 0.9:
            consensus = "强烈建议购买（多模型一致看好）"
        elif avg_prob >= 0.8:
            consensus = "谨慎买入（多模型较为看好）"
        elif avg_prob >= 0.7:
            consensus = "观望（模型意见不一致）"
        else:
            consensus = "不建议购买"
        
        print(f"\n综合建议: {consensus}")
    
    # 打印实际情况
    if has_future_data:
        print(f"\n{'='*80}")
        print("实际情况验证")
        print(f"{'='*80}")
        
        actual_return, start_price, end_price = calculate_actual_return(data, date_idx, future_days)
        
        if actual_return is not None:
            print(f"基准日期 ({target_date}): 收盘价 = {start_price:.2f}")
            print(f"目标日期 ({date_info['predict_end_date']}): 收盘价 = {end_price:.2f}")
            print(f"实际涨跌幅: {actual_return * 100:.2f}%")
            
            # 判断实际结果
            uprise_threshold = DataConfig.UPRISE_THRESHOLD
            if actual_return >= uprise_threshold:
                actual_label = f"上涨 (≥{uprise_threshold*100}%)"
                result_icon = "✓"
            else:
                actual_label = f"未达标 (<{uprise_threshold*100}%)"
                result_icon = "✗"
            
            print(f"实际结果: {result_icon} {actual_label}")
            
            # 模型预测验证
            if len(predictions) > 0:
                correct_count = 0
                for model_file, probability in predictions:
                    predicted_up = probability >= 0.5
                    actual_up = actual_return >= uprise_threshold
                    
                    if predicted_up == actual_up:
                        correct_count += 1
                
                accuracy = correct_count / len(predictions)
                print(f"\n模型预测准确度: {correct_count}/{len(predictions)} = {accuracy:.1%}")
            
            # 打印逐日详细数据
            print_day_by_day_details(df, date_idx, future_days)
        else:
            print("无法计算实际收益率（数据异常）")
    else:
        print(f"\n{'='*80}")
        print("实际情况验证")
        print(f"{'='*80}")
        print("未来数据不足，无法验证实际结果")
        print(f"需要日期 {target_date} 之后至少 {future_days} 天的数据")
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

