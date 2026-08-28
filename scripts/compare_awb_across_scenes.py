from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "results"

indoor_csv_path = (
    results_dir
    / "colorchecker_delta_e_summary.csv"
)

outdoor_csv_path = (
    results_dir
    / "outdoor1_delta_e_summary.csv"
)

output_csv_path = (
    results_dir
    / "awb_cross_scene_summary.csv"
)

output_figure_path = (
    results_dir
    / "19_awb_cross_scene_delta_e.png"
)

method_order = [
    "Fixed WB",
    "Norm2 AWB",
    "Gray World",
    "PCA AWB",
]


def load_mean_delta_e(csv_path):
    """读取一个场景中各AWB方法的平均Delta E00。"""

    results = {}

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            method_name = row["method"]
            results[method_name] = float(
                row["mean_delta_e00"]
            )

    return results


indoor_results = load_mean_delta_e(
    indoor_csv_path
)

outdoor_results = load_mean_delta_e(
    outdoor_csv_path
)

# 检查两个场景是否都有全部方法
for method_name in method_order:
    if method_name not in indoor_results:
        raise ValueError(
            f"Indoor1 missing method: {method_name}"
        )

    if method_name not in outdoor_results:
        raise ValueError(
            f"Outdoor1 missing method: {method_name}"
        )


# 每一行对应一种方法：
# 第0列是Indoor1，第1列是Outdoor1
scene_matrix = np.array(
    [
        [
            indoor_results[method_name],
            outdoor_results[method_name],
        ]
        for method_name in method_order
    ],
    dtype=np.float64
)

# axis=1表示平均掉两个场景，保留四种方法
two_scene_mean = np.mean(
    scene_matrix,
    axis=1
)

# 每种方法在两个场景中的较差结果
worst_scene_score = np.max(
    scene_matrix,
    axis=1
)

worst_scene_index = np.argmax(
    scene_matrix,
    axis=1
)

scene_names = np.array(
    ["Indoor1", "Outdoor1"]
)

worst_scene_names = scene_names[
    worst_scene_index
]

fixed_mean = two_scene_mean[0]

improvement_vs_fixed = (
    fixed_mean
    - two_scene_mean
) / fixed_mean * 100


# 保存跨场景CSV
with output_csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    field_names = [
        "method",
        "indoor1_mean_delta_e00",
        "outdoor1_mean_delta_e00",
        "two_scene_mean_delta_e00",
        "worst_scene_mean_delta_e00",
        "worst_scene",
        "improvement_vs_fixed_percent",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=field_names
    )

    writer.writeheader()

    for method_index, method_name in enumerate(
        method_order
    ):
        writer.writerow(
            {
                "method": method_name,
                "indoor1_mean_delta_e00":
                    scene_matrix[method_index, 0],
                "outdoor1_mean_delta_e00":
                    scene_matrix[method_index, 1],
                "two_scene_mean_delta_e00":
                    two_scene_mean[method_index],
                "worst_scene_mean_delta_e00":
                    worst_scene_score[method_index],
                "worst_scene":
                    worst_scene_names[method_index],
                "improvement_vs_fixed_percent":
                    improvement_vs_fixed[method_index],
            }
        )


# 制作跨场景对比图
fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)

x_positions = np.arange(
    len(method_order)
)

bar_width = 0.36

indoor_bars = axes[0].bar(
    x_positions - bar_width / 2,
    scene_matrix[:, 0],
    width=bar_width,
    label="Indoor1",
    color="#4C78A8"
)

outdoor_bars = axes[0].bar(
    x_positions + bar_width / 2,
    scene_matrix[:, 1],
    width=bar_width,
    label="Outdoor1",
    color="#F58518"
)

axes[0].set_title(
    "AWB Performance Across Scenes"
)
axes[0].set_ylabel("Mean Delta E00")
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(method_order)
axes[0].legend()
axes[0].grid(
    axis="y",
    alpha=0.3
)

axes[0].bar_label(
    indoor_bars,
    fmt="%.2f",
    padding=3
)

axes[0].bar_label(
    outdoor_bars,
    fmt="%.2f",
    padding=3
)


method_colors = [
    "#8C8C8C",
    "#4C78A8",
    "#54A24B",
    "#F58518",
]

mean_bars = axes[1].bar(
    method_order,
    two_scene_mean,
    color=method_colors
)

axes[1].set_title(
    "Two-Scene Mean Delta E00"
)
axes[1].set_ylabel("Mean Delta E00")
axes[1].grid(
    axis="y",
    alpha=0.3
)

axes[1].text(
    0.5,
    0.95,
    "Lower is better",
    transform=axes[1].transAxes,
    ha="center",
    va="top"
)

axes[1].bar_label(
    mean_bars,
    fmt="%.2f",
    padding=3
)

fig.tight_layout()

fig.savefig(
    output_figure_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


print("\nCross-scene AWB summary:")

for method_index, method_name in enumerate(
    method_order
):
    print(f"\n{method_name}:")
    print(
        "  Indoor1:",
        f"{scene_matrix[method_index, 0]:.3f}"
    )
    print(
        "  Outdoor1:",
        f"{scene_matrix[method_index, 1]:.3f}"
    )
    print(
        "  Two-scene mean:",
        f"{two_scene_mean[method_index]:.3f}"
    )
    print(
        "  Worst scene:",
        worst_scene_names[method_index],
        f"({worst_scene_score[method_index]:.3f})"
    )
    print(
        "  Improvement vs Fixed:",
        f"{improvement_vs_fixed[method_index]:.2f}%"
    )

print("\nSaved:", output_csv_path)
print("Saved:", output_figure_path)