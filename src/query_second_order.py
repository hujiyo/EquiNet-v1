"""Interactive lookup tool for second-order state statistics.

运行脚本后先展示 `KNOWLEDGE.md` 中 A/B 维度的含义，并提示用户输入两日状态
（例如 `A1B1A2B2` 或 `A1B1->A2B2`）。脚本会读取 `second_order.xlsx`（默认位于
`out/second_order.xlsx`），查找对应的统计结果并输出。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from first_order_stats import PRICE_RULES, VOLUME_RULES

SEPARATOR = "-" * 60


def render_dimension_info() -> str:
    volume_lines = ["量能维度 A (相对前日成交量):"]
    for label, desc in VOLUME_RULES:
        volume_lines.append(f"  {label}: {desc}")

    price_lines = ["价格维度 B (当日涨跌幅):"]
    for label, desc in PRICE_RULES:
        price_lines.append(f"  {label}: {desc}")

    return "\n".join(volume_lines + ["", *price_lines])


def parse_state_pair(user_input: str) -> Tuple[str, str]:
    text = user_input.strip().upper()
    if not text:
        raise ValueError("输入为空")

    # 支持 "A1B1->A2B2" / "A1B1 A2B2" / "A1B1,A2B2" / "A1B1A2B2"
    if "->" in text:
        parts = [p.strip() for p in text.split("->") if p.strip()]
    else:
        normalized = re.sub(r"[\s,\/]+", " ", text)
        tokens = normalized.split()
        if len(tokens) == 2:
            parts = tokens
        else:
            compact = re.sub(r"\s", "", text)
            if len(compact) == 8:
                parts = [compact[:4], compact[4:]]
            else:
                raise ValueError("输入格式不正确，请参考示例 A1B1A2B2")

    if len(parts) != 2:
        raise ValueError("需要提供连续两天的状态，请检查输入")

    pattern = re.compile(r"A[1-6]B[1-6]")
    for idx, part in enumerate(parts, start=1):
        if not pattern.fullmatch(part):
            raise ValueError(f"第 {idx} 天状态 '{part}' 不合法，应为 A1~A6 + B1~B6 组合")

    return parts[0], parts[1]


def load_statistics(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"统计文件不存在: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="second_order")
    if "状态组合" not in df.columns:
        raise ValueError("统计表缺少 '状态组合' 列，请确认文件格式正确")

    df = df.set_index("状态组合")
    return df


def format_result(row: pd.Series) -> str:
    items = []
    for col, value in row.items():
        if pd.isna(value):
            display = "NaN"
        elif isinstance(value, float):
            if col.endswith("概率"):
                display = f"{value:.2%}"
            else:
                display = f"{value:.6f}" if abs(value) < 1 else f"{value:.4f}"
        else:
            display = str(value)
        items.append(f"{col}: {display}")
    return "\n".join(items)


def interactive_lookup(df: pd.DataFrame) -> None:
    print(SEPARATOR)
    print(render_dimension_info())
    print(SEPARATOR)
    print("输入格式示例: A1B1A2B2 或 A1B1->A2B2 (输入 q 退出)")

    while True:
        try:
            user_input = input("请输入连续两日状态: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break

        try:
            prev_state, curr_state = parse_state_pair(user_input)
        except ValueError as exc:
            print(f"输入错误: {exc}")
            continue

        key = f"{prev_state}->{curr_state}"
        if key not in df.index:
            print(f"未在统计结果中找到 {key} 的记录。")
            continue

        row = df.loc[key]
        print(SEPARATOR)
        print(f"查询结果 ({key}):")
        print(format_result(row))
        print(SEPARATOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="二阶状态统计查询工具")
    parser.add_argument(
        "--excel",
        type=str,
        default="out/second_order.xlsx",
        help="second_order.xlsx 文件路径",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    df_stats = load_statistics(excel_path)

    print(f"已加载统计文件: {excel_path}")
    interactive_lookup(df_stats)


if __name__ == "__main__":  # pragma: no cover
    main()
