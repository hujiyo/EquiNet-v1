"""Compute first-order state transition statistics for stock data.

本脚本根据 `KNOWLEDGE.md` 中的状态编码方案，统计所有股票在某日处于
{A_i, B_j} 状态后，次日出现“大幅上涨 / 上涨 / 下跌”三种情况的概率分布，
并输出到 `first_order.xlsx`。
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from config import DataConfig
except ImportError:  # pragma: no cover
    DataConfig = None  # type: ignore


# 次日涨停/上涨阈值来自 KNOWLEDGE.md 的固定定义
LIMIT_UP_THRESHOLD = 0.08  # 涨幅 ≥ 8% 视为涨停（B6）
UP_THRESHOLD = 0.0        # 0% ≤ 涨幅 < 8% 视为上涨


VOLUME_RULES: Tuple[Tuple[str, str], ...] = (
    ("A1", "<0.3"),
    ("A2", "0.3-0.7"),
    ("A3", "0.7-1.0"),
    ("A4", "1.0-1.3"),
    ("A5", "1.3-1.7"),
    ("A6", ">=1.7"),
)

PRICE_RULES: Tuple[Tuple[str, str], ...] = (
    ("B1", "<-8%"),
    ("B2", "-8%~-3%"),
    ("B3", "-3%~0%"),
    ("B4", "0%~3%"),
    ("B5", "3%~8%"),
    ("B6", ">8%"),  # 修改为 >8% 以保持一致性
)

STATE_KEYS = [f"{a}{b}" for a, _ in VOLUME_RULES for b, _ in PRICE_RULES]


def classify_volume_ratio(ratio: float) -> Optional[str]:
    """Return the volume bucket label given the ratio versus前一日."""
    if not math.isfinite(ratio) or ratio <= 0:
        return None
    for label, upper in VOLUME_RULES:
        if upper.startswith("<"):
            threshold = float(upper[1:])
            if ratio < threshold:
                return label
        elif "-" in upper:
            low_str, high_str = upper.split("-")
            low = float(low_str)
            high = float(high_str)
            if low <= ratio < high:
                return label
        else:  # >= 格式
            threshold = float(upper.replace(">=", ""))
            if ratio >= threshold:
                return label
    return None


def classify_price_change(change: float) -> Optional[str]:
    """Return the price bucket label based on当日涨跌幅。"""
    if not math.isfinite(change):
        print(f"警告: 无效的价格变化值: {change}")
        return None
    
    # 将涨跌幅转换为百分比形式（例如：0.05 表示 5%）
    percent = change * 100  # 转换为百分比
    
    for label, interval in PRICE_RULES:
        if interval.startswith('<'):
            # 处理 <X% 格式
            threshold = float(interval[1:-1])
            if percent < threshold:
                return label
        elif interval.startswith('>'):
            # 处理 >X% 格式
            threshold = float(interval[1:-1])
            if percent > threshold:
                return label
        else:
            # 处理 X%~Y% 格式
            low, high = map(float, interval.replace('%', '').split('~'))
            if low <= percent < high:
                return label
    return None


def classify_next_day(change: float) -> Optional[str]:
    if not math.isfinite(change):
        return None
    if change >= LIMIT_UP_THRESHOLD:
        return "limit_up"
    if change >= UP_THRESHOLD:
        return "up"
    return "down"


def load_stock_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    required_cols = {"time", "start", "max", "min", "end", "volume"}
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as exc:  # pragma: no cover
        print(f"读取 {file_path.name} 失败: {exc}")
        return None

    df.columns = [str(col).strip().lower() for col in df.columns]
    missing = required_cols - set(df.columns)
    if missing:
        print(f"{file_path.name} 缺少列: {sorted(missing)}，已跳过。")
        return None

    df = df.sort_values("time").reset_index(drop=True)
    df = df[['time', 'start', 'max', 'min', 'end', 'volume']].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['end', 'volume'], inplace=True)
    return df


def iter_stock_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*.xlsx")):
        if path.is_file():
            yield path


def compute_first_order_stats(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "total": 0,
        "limit_up": 0,
        "up": 0,
        "down": 0,
        "sum_next_return": 0.0,
    })
    skipped_files = 0
    processed_rows = 0

    for file_path in iter_stock_files(data_dir):
        df = load_stock_dataframe(file_path)
        if df is None or len(df) < 3:
            skipped_files += 1
            continue

        closes = df['end'].to_numpy(dtype=float)
        volumes = df['volume'].to_numpy(dtype=float)

        for idx in range(1, len(df) - 1):
            prev_close = closes[idx - 1]
            curr_close = closes[idx]
            next_close = closes[idx + 1]
            prev_volume = volumes[idx - 1]
            curr_volume = volumes[idx]

            if min(prev_close, curr_close, prev_volume, curr_volume, next_close) <= 0:
                continue

            volume_ratio = curr_volume / prev_volume
            price_change = (curr_close - prev_close) / prev_close
            next_change = (next_close - curr_close) / curr_close

            volume_bucket = classify_volume_ratio(volume_ratio)
            price_bucket = classify_price_change(price_change)
            next_bucket = classify_next_day(next_change)

            if not (volume_bucket and price_bucket and next_bucket):
                continue

            state_key = f"{volume_bucket}{price_bucket}"
            state_stats = counts[state_key]
            state_stats["total"] += 1
            state_stats[next_bucket] += 1
            state_stats["sum_next_return"] += next_change
            processed_rows += 1

    records = []
    for volume_label, volume_desc in VOLUME_RULES:
        for price_label, price_desc in PRICE_RULES:
            state = f"{volume_label}{price_label}"
            stats = counts[state]
            total = int(stats["total"])
            record = {
                "状态": state,
                "成交量区间": volume_label,
                "成交量范围": volume_desc,
                "价格区间": price_label,
                "价格范围": price_desc,
                "样本数量": total,
                "涨停概率": stats["limit_up"] / total if total else np.nan,
                "上涨概率": stats["up"] / total if total else np.nan,
                "下跌概率": stats["down"] / total if total else np.nan,
                "平均次日收益": stats["sum_next_return"] / total if total else np.nan,
            }
            records.append(record)

    df_result = pd.DataFrame(records)

    # 汇总行
    total_samples = df_result["样本数量"].sum()
    if total_samples > 0:
        df_result = pd.concat([
            df_result,
            pd.DataFrame([
                {
                    "状态": "总计",
                    "成交量区间": "-",
                    "成交量范围": "-",
                    "价格区间": "-",
                    "价格范围": "-",
                    "样本数量": total_samples,
                    "涨停概率": (
                        (df_result["涨停概率"].fillna(0) * df_result["样本数量"]).sum() / total_samples
                    ),
                    "上涨概率": (
                        (df_result["上涨概率"].fillna(0) * df_result["样本数量"]).sum() / total_samples
                    ),
                    "下跌概率": (
                        (df_result["下跌概率"].fillna(0) * df_result["样本数量"]).sum() / total_samples
                    ),
                    "平均次日收益": (
                        df_result["平均次日收益"].fillna(0)
                        .mul(df_result["样本数量"])  # type: ignore[arg-type]
                        .sum()
                        / total_samples
                    ),
                }
            ]),
        ], ignore_index=True)

    meta = {
        "processed_samples": processed_rows,
        "total_states_with_data": int((df_result["样本数量"] > 0).sum()),
        "skipped_files": skipped_files,
    }
    return df_result, meta


def resolve_data_dir(cli_data_dir: Optional[str]) -> Path:
    if cli_data_dir:
        data_dir = Path(cli_data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"指定的 data_dir 不存在: {data_dir}")
        return data_dir

    if DataConfig is not None:
        candidate = Path(DataConfig.DATA_DIR).expanduser().resolve()
        if candidate.exists():
            return candidate

    fallback = Path(__file__).resolve().parent / "today"
    if fallback.exists():
        print(f"提示：使用备用数据目录 {fallback}")
        return fallback

    raise FileNotFoundError("无法找到有效的数据目录，请通过 --data-dir 指定。")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计一阶状态转移概率，并写入 first_order.xlsx")
    parser.add_argument("--data-dir", type=str, default=None, help="包含股票数据的目录，递归搜索 .xlsx")
    parser.add_argument("--output", type=str, default="first_order.xlsx", help="输出文件路径")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {args.output}")

    print(f"次日涨停阈值: {LIMIT_UP_THRESHOLD*100:.2f}%")
    print(f"次日上涨阈值: {UP_THRESHOLD*100:.2f}%")

    df_result, meta = compute_first_order_stats(data_dir)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings_df = pd.DataFrame([
        ("data_dir", str(data_dir)),
        ("limit_up_threshold", LIMIT_UP_THRESHOLD),
        ("up_threshold", UP_THRESHOLD),
        ("processed_samples", meta["processed_samples"]),
        ("states_with_data", meta["total_states_with_data"]),
        ("skipped_files", meta["skipped_files"]),
    ], columns=["key", "value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, index=False, sheet_name="first_order")
        settings_df.to_excel(writer, index=False, sheet_name="meta")

    print("统计完成，已写入:", output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
