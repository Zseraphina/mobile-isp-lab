from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 当前mobile-isp-lab项目的根目录
project_root = Path(__file__).resolve().parents[1]

# 四张Infinite-ISP输出图的路径
fixed_wb_path = Path(
    "/workspace/infinite-isp-baseline/out_frames/"
    "Out_Indoor1_2592x1536_12bit_RGGB_20260822_183551.png"
)

auto_awb_path = Path(
    "/workspace/infinite-isp-baseline/out_frames/"
    "Out_Indoor1_2592x1536_12bit_RGGB_20260825_200548.png"
)

gray_world_path = Path(
    "/workspace/infinite-isp-baseline/out_frames/"
    "Out_Indoor1_2592x1536_12bit_RGGB_20260826_234953.png"
)

pca_awb_path = Path(
    "/workspace/infinite-isp-baseline/out_frames/"
    "Out_Indoor1_2592x1536_12bit_RGGB_20260827_000854.png"
)

# 对比图的保存路径
output_path = (
    project_root
    / "results"
    / "08_infinite_isp_awb_comparison.png"
)


def load_rgb_image(image_path):
    """读取PNG，并保证输出为uint8 RGB数组。"""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.array(rgb_image, dtype=np.uint8)


# 读取四张图片
fixed_wb_uint8 = load_rgb_image(fixed_wb_path)
auto_awb_uint8 = load_rgb_image(auto_awb_path)
gray_world_uint8 = load_rgb_image(gray_world_path)
pca_awb_uint8 = load_rgb_image(pca_awb_path)


# 检查四张图片尺寸是否一致
if not (
    fixed_wb_uint8.shape
    == auto_awb_uint8.shape
    == gray_world_uint8.shape
    == pca_awb_uint8.shape
):
    raise ValueError(
        "Image shapes do not match: "
        f"Fixed={fixed_wb_uint8.shape}, "
        f"Norm2={auto_awb_uint8.shape}, "
        f"Gray World={gray_world_uint8.shape}, "
        f"PCA={pca_awb_uint8.shape}"
    )

print("Fixed WB shape:", fixed_wb_uint8.shape)
print("Norm2 AWB shape:", auto_awb_uint8.shape)
print("Gray World shape:", gray_world_uint8.shape)
print("PCA AWB shape:", pca_awb_uint8.shape)

# 转为float32，避免uint8减法发生回绕
fixed_wb = fixed_wb_uint8.astype(np.float32)
norm2_awb = auto_awb_uint8.astype(np.float32)
gray_world_awb = gray_world_uint8.astype(np.float32)
pca_awb = pca_awb_uint8.astype(np.float32)

def calculate_difference_metrics(reference, comparison):
    """计算comparison相对于reference的图像差异。"""

    difference = comparison - reference
    absolute_difference = np.abs(difference)

    mean_shift = np.mean(
        difference,
        axis=(0, 1),
        dtype=np.float64
    )

    channel_mae = np.mean(
        absolute_difference,
        axis=(0, 1),
        dtype=np.float64
    )

    overall_mae = np.mean(
        absolute_difference,
        dtype=np.float64
    )

    difference_vis = np.clip(
        absolute_difference * 4,
        0,
        255
    ).astype(np.uint8)

    return {
        "mean_shift": mean_shift,
        "channel_mae": channel_mae,
        "overall_mae": overall_mae,
        "difference_vis": difference_vis,
    }


# 分别计算两种AWB相对于Fixed WB的差异
norm2_difference_metrics = calculate_difference_metrics(
    fixed_wb,
    norm2_awb
)

gray_world_difference_metrics = calculate_difference_metrics(
    fixed_wb,
    gray_world_awb
)

pca_difference_metrics = calculate_difference_metrics(
    fixed_wb,
    pca_awb
)


# 第一行显示四张结果图
# 第二行显示三种AWB相对于Fixed WB的差异
fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 9)
)

axes[0, 0].imshow(fixed_wb_uint8)
axes[0, 0].set_title("Fixed WB")

axes[0, 1].imshow(auto_awb_uint8)
axes[0, 1].set_title("Norm2 AWB")

axes[0, 2].imshow(gray_world_uint8)
axes[0, 2].set_title("Gray World AWB")

axes[0, 3].imshow(pca_awb_uint8)
axes[0, 3].set_title("PCA AWB")

axes[1, 0].text(
    0.5,
    0.5,
    "Difference maps\nrelative to Fixed WB",
    ha="center",
    va="center",
    fontsize=14
)

axes[1, 1].imshow(
    norm2_difference_metrics["difference_vis"]
)
axes[1, 1].set_title("Norm2 Absolute Difference x4")

axes[1, 2].imshow(
    gray_world_difference_metrics["difference_vis"]
)
axes[1, 2].set_title("Gray World Absolute Difference x4")

axes[1, 3].imshow(
    pca_difference_metrics["difference_vis"]
)
axes[1, 3].set_title("PCA Absolute Difference x4")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

# 确保results目录存在
output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


def print_difference_metrics(method_name, metrics):
    """打印一种AWB相对于Fixed WB的差异指标。"""

    channel_names = ["R", "G", "B"]

    print(f"\n{method_name} vs Fixed WB:")
    print("Overall MAE:", metrics["overall_mae"])

    for channel_index, channel_name in enumerate(channel_names):
        print(
            f"{channel_name} mean shift:",
            metrics["mean_shift"][channel_index]
        )
        print(
            f"{channel_name} MAE:",
            metrics["channel_mae"][channel_index]
        )


print("Image shape:", fixed_wb_uint8.shape)

print_difference_metrics(
    "Norm2 AWB",
    norm2_difference_metrics
)

print_difference_metrics(
    "Gray World AWB",
    gray_world_difference_metrics
)

print_difference_metrics(
    "PCA AWB",
    pca_difference_metrics
)

print("Saved:", output_path)


# ColorChecker大致所在的原图区域
x_start = 500
x_end = 1200
y_start = 900
y_end = 1536

# NumPy图像索引顺序是[行, 列]，也就是[y, x]
fixed_colorchecker = fixed_wb_uint8[
    y_start:y_end,
    x_start:x_end
]

norm2_colorchecker = auto_awb_uint8[
    y_start:y_end,
    x_start:x_end
]

gray_world_colorchecker = gray_world_uint8[
    y_start:y_end,
    x_start:x_end
]

pca_colorchecker = pca_awb_uint8[
    y_start:y_end,
    x_start:x_end
]

# 创建三种白平衡的ColorChecker局部对比图
crop_fig, crop_axes = plt.subplots(
    1,
    4,
    figsize=(24, 6)
)

# extent让坐标轴继续显示原图坐标
image_extent = [
    x_start,
    x_end,
    y_end,
    y_start
]

crop_axes[0].imshow(
    fixed_colorchecker,
    extent=image_extent
)
crop_axes[0].set_title("Fixed WB ColorChecker")

crop_axes[1].imshow(
    norm2_colorchecker,
    extent=image_extent
)
crop_axes[1].set_title("Norm2 AWB ColorChecker")

crop_axes[2].imshow(
    gray_world_colorchecker,
    extent=image_extent
)
crop_axes[2].set_title("Gray World ColorChecker")

crop_axes[3].imshow(
    pca_colorchecker,
    extent=image_extent
)
crop_axes[3].set_title("PCA AWB ColorChecker")

for ax in crop_axes:
    ax.set_xlabel("Original image x")
    ax.set_ylabel("Original image y")
    ax.grid(
        color="yellow",
        alpha=0.4
    )

plt.tight_layout()

crop_output_path = (
    project_root
    / "results"
    / "09_colorchecker_crop.png"
)

crop_fig.savefig(
    crop_output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(crop_fig)

print("Saved:", crop_output_path)
# 六个中性灰阶块的区域
# 格式：(名称, x起点, x终点, y起点, y终点)
neutral_patch_rois = [
    ("White",       605, 650, 1415, 1460),
    ("Light Gray",  685, 735, 1415, 1460),
    ("Gray",        770, 820, 1415, 1460),
    ("Dark Gray",   855, 905, 1415, 1460),
    ("Darker Gray", 940, 990, 1415, 1460),
    ("Black",      1025, 1075, 1415, 1460),
]


def calculate_neutral_metrics(image, patch_rois):
    """计算每个中性块的平均RGB和通道分离程度。"""

    metrics = []

    for patch_name, x1, x2, y1, y2 in patch_rois:
        patch = image[
            y1:y2,
            x1:x2
        ].astype(np.float64)

        mean_rgb = np.mean(
            patch,
            axis=(0, 1)
        )

        mean_level = np.mean(mean_rgb)

        channel_spread = (
            np.max(mean_rgb)
            - np.min(mean_rgb)
        )

        spread_percent = (
            channel_spread
            / mean_level
            * 100
        )

        metrics.append(
            {
                "name": patch_name,
                "mean_rgb": mean_rgb,
                "spread_percent": spread_percent,
            }
        )

    return metrics


fixed_neutral_metrics = calculate_neutral_metrics(
    fixed_wb_uint8,
    neutral_patch_rois
)

norm2_neutral_metrics = calculate_neutral_metrics(
    auto_awb_uint8,
    neutral_patch_rois
)

gray_world_neutral_metrics = calculate_neutral_metrics(
    gray_world_uint8,
    neutral_patch_rois
)

pca_neutral_metrics = calculate_neutral_metrics(
    pca_awb_uint8,
    neutral_patch_rois
)

print("\nNeutral patch evaluation:")

for (
    fixed_result,
    norm2_result,
    gray_world_result,
    pca_result
) in zip(
    fixed_neutral_metrics,
    norm2_neutral_metrics,
    gray_world_neutral_metrics,
    pca_neutral_metrics
):
    fixed_rgb = fixed_result["mean_rgb"]
    norm2_rgb = norm2_result["mean_rgb"]
    gray_world_rgb = gray_world_result["mean_rgb"]
    pca_rgb = pca_result["mean_rgb"]

    print(f"\n{fixed_result['name']}:")

    print(
        "  Fixed WB RGB:",
        np.round(fixed_rgb, 2),
        "spread:",
        f"{fixed_result['spread_percent']:.2f}%"
    )

    print(
        "  Norm2 AWB RGB:",
        np.round(norm2_rgb, 2),
        "spread:",
        f"{norm2_result['spread_percent']:.2f}%"
    )

    print(
        "  Gray World RGB:",
        np.round(gray_world_rgb, 2),
        "spread:",
        f"{gray_world_result['spread_percent']:.2f}%"
    )
    
    print(
        "  PCA AWB RGB:",
        np.round(pca_rgb, 2),
        "spread:",
        f"{pca_result['spread_percent']:.2f}%"
    )

def calculate_neutral_score(neutral_metrics):
    """计算除黑色块之外的平均中性色通道分离程度。"""

    return np.mean(
        [
            result["spread_percent"]
            for result in neutral_metrics[:-1]
        ]
    )


# 黑色块信号很弱，暂不计入整体中性色评分
fixed_neutral_score = calculate_neutral_score(
    fixed_neutral_metrics
)

norm2_neutral_score = calculate_neutral_score(
    norm2_neutral_metrics
)

gray_world_neutral_score = calculate_neutral_score(
    gray_world_neutral_metrics
)

pca_neutral_score = calculate_neutral_score(
    pca_neutral_metrics
)
print("\nFixed WB neutral score:", fixed_neutral_score)
print("Norm2 AWB neutral score:", norm2_neutral_score)
print(
    "Gray World neutral score:",
    gray_world_neutral_score
)
print(
    "PCA AWB neutral score:",
    pca_neutral_score
)

def calculate_relative_improvement(
    reference_score,
    test_score
):
    """计算相对于参考方法的评分下降百分比。"""

    return (
        reference_score
        - test_score
    ) / reference_score * 100


# 增益来自各次Infinite-ISP输出YAML
algorithm_results = [
    {
        "method": "Fixed WB",
        "r_gain": 1.24609375,
        "b_gain": 2.80859375,
        "overall_mae_vs_fixed": 0.0,
        "neutral_score_percent": fixed_neutral_score,
        "neutral_improvement_vs_fixed_percent": 0.0,
        "ae_feedback": "Correct Exposure",
    },
    {
        "method": "Norm2 AWB",
        "r_gain": 1.3877439116600867,
        "b_gain": 2.955442980094861,
        "overall_mae_vs_fixed": (
            norm2_difference_metrics["overall_mae"]
        ),
        "neutral_score_percent": norm2_neutral_score,
        "neutral_improvement_vs_fixed_percent": (
            calculate_relative_improvement(
                fixed_neutral_score,
                norm2_neutral_score
            )
        ),
        "ae_feedback": "Overexposed",
    },
    {
        "method": "Gray World",
        "r_gain": 1.3960070859486422,
        "b_gain": 2.9887926761019323,
        "overall_mae_vs_fixed": (
            gray_world_difference_metrics["overall_mae"]
        ),
        "neutral_score_percent": gray_world_neutral_score,
        "neutral_improvement_vs_fixed_percent": (
            calculate_relative_improvement(
                fixed_neutral_score,
                gray_world_neutral_score
            )
        ),
        "ae_feedback": "Overexposed",
    },
    {
        "method": "PCA AWB",
        "r_gain": 1.3558077812194824,
        "b_gain": 2.8518218994140625,
        "overall_mae_vs_fixed": (
            pca_difference_metrics["overall_mae"]
        ),
        "neutral_score_percent": pca_neutral_score,
        "neutral_improvement_vs_fixed_percent": (
            calculate_relative_improvement(
                fixed_neutral_score,
                pca_neutral_score
            )
        ),
        "ae_feedback": "Correct Exposure",
    },
]

# 保存CSV实验汇总表
csv_output_path = (
    project_root
    / "results"
    / "awb_baseline_summary.csv"
)

csv_field_names = [
    "method",
    "r_gain",
    "b_gain",
    "overall_mae_vs_fixed",
    "neutral_score_percent",
    "neutral_improvement_vs_fixed_percent",
    "ae_feedback",
]

with csv_output_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_field_names
    )
    writer.writeheader()
    writer.writerows(algorithm_results)

print("Saved:", csv_output_path)

# 提取柱状图需要的方法名和评分
method_names = [
    result["method"]
    for result in algorithm_results
]

neutral_scores = [
    result["neutral_score_percent"]
    for result in algorithm_results
]

score_fig, score_ax = plt.subplots(
    figsize=(9, 6)
)

bars = score_ax.bar(
    method_names,
    neutral_scores,
    color=[
        "#7f7f7f",
        "#4c78a8",
        "#59a14f",
        "#f28e2b",
    ]
)

# 在每根柱子上显示具体分数
for bar, score in zip(bars, neutral_scores):
    score_ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.2,
        f"{score:.2f}%",
        ha="center",
        va="bottom"
    )

score_ax.set_ylabel(
    "Neutral channel spread (%)"
)
score_ax.set_title(
    "Indoor1 AWB Neutral-Patch Evaluation"
)
score_ax.text(
    0.5,
    0.97,
    "Lower is better",
    transform=score_ax.transAxes,
    ha="center",
    va="top"
)
score_ax.grid(
    axis="y",
    alpha=0.3
)

score_ax.set_ylim(
    0,
    max(neutral_scores) * 1.18
)

plt.tight_layout()

score_chart_path = (
    project_root
    / "results"
    / "10_awb_neutral_score_comparison.png"
)

score_fig.savefig(
    score_chart_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(score_fig)

print("Saved:", score_chart_path)