"""Compute second-order state transition statistics for stock data.

本脚本按照 `KNOWLEDGE.md` 中的状态编码方案，统计所有股票在连续两日分别处于
{A_i, B_j} 状态后，次日出现「涨停 / 上涨 / 下跌」三种情况的概率分布，并输出到
`second_order.xlsx`。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from first_order_stats import (
    LIMIT_UP_THRESHOLD,
    PRICE_RULES,
    STATE_KEYS,
    UP_THRESHOLD,
    VOLUME_RULES,
    classify_next_day,
    classify_price_change,
    classify_volume_ratio,
    iter_stock_files,
    load_stock_dataframe,
    resolve_data_dir,
)

VOLUME_TEXT = {
    "A1": "极度缩量",
    "A2": "大幅缩量",
    "A3": "缩量",
    "A4": "放量",
    "A5": "大幅放量",
    "A6": "极度放量",
}

PRICE_TEXT = {
    "B1": "跌停",
    "B2": "大幅下跌",
    "B3": "下跌",
    "B4": "上涨",
    "B5": "大幅上涨",
    "B6": "涨停",
}


def split_state_label(state: str) -> Tuple[str, str]:
    """将形如 A1B2 的状态拆解为 (A1, B2)。"""
    return state[:2], state[2:]


def compute_second_order_stats(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    counts: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "limit_up": 0,
            "up": 0,
            "down": 0,
            "sum_next_return": 0.0,
        }
    )

    processed_pairs = 0
    skipped_files = 0
    total_files = 0
    processed_files = 0

    volume_range_map = {label: desc for label, desc in VOLUME_RULES}
    price_range_map = {label: desc for label, desc in PRICE_RULES}

    for file_path in iter_stock_files(data_dir):
        total_files += 1
        df = load_stock_dataframe(file_path)
        if df is None or len(df) < 4:
            skipped_files += 1
            continue

        closes = df["end"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)

        # 先为每一天计算状态（与 first_order_stats 中逻辑保持一致）
        daily_states: list[Optional[str]] = [None] * len(df)
        for idx in range(1, len(df)):
            prev_close = closes[idx - 1]
            curr_close = closes[idx]
            prev_volume = volumes[idx - 1]
            curr_volume = volumes[idx]

            if min(prev_close, curr_close, prev_volume, curr_volume) <= 0:
                continue

            volume_ratio = curr_volume / prev_volume
            price_change = (curr_close - prev_close) / prev_close

            volume_bucket = classify_volume_ratio(volume_ratio)
            price_bucket = classify_price_change(price_change)

            if volume_bucket and price_bucket:
                daily_states[idx] = f"{volume_bucket}{price_bucket}"

        file_pairs = 0

        # 遍历连续两日状态（索引 idx 表示当前日）
        for idx in range(2, len(df) - 1):
            prev_state = daily_states[idx - 1]
            curr_state = daily_states[idx]
            curr_close = closes[idx]
            next_close = closes[idx + 1]

            if not (prev_state and curr_state):
                continue
            if min(curr_close, next_close) <= 0:
                continue

            next_change = (next_close - curr_close) / curr_close
            next_bucket = classify_next_day(next_change)
            if not next_bucket:
                continue

            key = (prev_state, curr_state)
            stats = counts[key]
            stats["total"] += 1
            stats[next_bucket] += 1
            stats["sum_next_return"] += next_change

            processed_pairs += 1
            file_pairs += 1

        if file_pairs > 0:
            processed_files += 1

    records = []
    for prev_state in STATE_KEYS:
        prev_vol_label, prev_price_label = split_state_label(prev_state)
        for curr_state in STATE_KEYS:
            curr_vol_label, curr_price_label = split_state_label(curr_state)
            stats = counts[(prev_state, curr_state)]
            total = int(stats["total"])

            record = {
                "状态组合": f"{prev_state}->{curr_state}",
                "前一日状态": prev_state,
                "当日状态": curr_state,
                "前一日成交量区间": prev_vol_label,
                "前一日成交量范围": volume_range_map.get(prev_vol_label, "-"),
                "前一日成交量说明": VOLUME_TEXT.get(prev_vol_label, "-"),
                "前一日价格区间": prev_price_label,
                "前一日价格范围": price_range_map.get(prev_price_label, "-"),
                "前一日价格说明": PRICE_TEXT.get(prev_price_label, "-"),
                "当日成交量区间": curr_vol_label,
                "当日成交量范围": volume_range_map.get(curr_vol_label, "-"),
                "当日成交量说明": VOLUME_TEXT.get(curr_vol_label, "-"),
                "当日价格区间": curr_price_label,
                "当日价格范围": price_range_map.get(curr_price_label, "-"),
                "当日价格说明": PRICE_TEXT.get(curr_price_label, "-"),
                "样本数量": total,
                "涨停概率": stats["limit_up"] / total if total else np.nan,
                "上涨概率": stats["up"] / total if total else np.nan,
                "下跌概率": stats["down"] / total if total else np.nan,
                "平均次日收益": stats["sum_next_return"] / total if total else np.nan,
            }
            records.append(record)

    df_result = pd.DataFrame(records)
    base_df = df_result.copy()

    # 汇总行
    total_samples = base_df["样本数量"].sum()
    if total_samples and total_samples > 0:
        total_row = {
            "状态组合": "总计",
            "前一日状态": "-",
            "当日状态": "-",
            "前一日成交量区间": "-",
            "前一日成交量范围": "-",
            "前一日成交量说明": "-",
            "前一日价格区间": "-",
            "前一日价格范围": "-",
            "前一日价格说明": "-",
            "当日成交量区间": "-",
            "当日成交量范围": "-",
            "当日成交量说明": "-",
            "当日价格区间": "-",
            "当日价格范围": "-",
            "当日价格说明": "-",
            "样本数量": total_samples,
            "涨停概率": (
                base_df["涨停概率"].fillna(0).mul(base_df["样本数量"]).sum() / total_samples
            ),
            "上涨概率": (
                base_df["上涨概率"].fillna(0).mul(base_df["样本数量"]).sum() / total_samples
            ),
            "下跌概率": (
                base_df["下跌概率"].fillna(0).mul(base_df["样本数量"]).sum() / total_samples
            ),
            "平均次日收益": (
                base_df["平均次日收益"].fillna(0).mul(base_df["样本数量"]).sum() / total_samples
            ),
        }
        df_result = pd.concat([df_result, pd.DataFrame([total_row])], ignore_index=True)

    pairs_with_data = int((base_df["样本数量"] > 0).sum())

    meta = {
        "processed_pairs": processed_pairs,
        "pairs_with_data": pairs_with_data,
        "total_files": total_files,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
    }

    return df_result, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计二阶状态转移概率，并写入 second_order.xlsx"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="包含股票数据的目录，递归搜索 .xlsx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="second_order.xlsx",
        help="输出文件路径",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {args.output}")
    print(f"次日涨停阈值: {LIMIT_UP_THRESHOLD*100:.2f}%")
    print(f"次日上涨阈值: {UP_THRESHOLD*100:.2f}%")

    df_result, meta = compute_second_order_stats(data_dir)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_df = pd.DataFrame(
        [
            ("数据目录", str(data_dir)),
            ("涨停阈值", f"{LIMIT_UP_THRESHOLD:.2%}"),
            ("上涨阈值", f"{UP_THRESHOLD:.2%}"),
            ("遍历文件数", meta["total_files"]),
            ("成功处理文件数", meta["processed_files"]),
            ("跳过文件数", meta["skipped_files"]),
            ("有效状态组合数", meta["pairs_with_data"]),
            ("有效样本对数", meta["processed_pairs"]),
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ],
        columns=["参数", "值"],
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, index=False, sheet_name="second_order")
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    print("统计完成，已写入:", output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
