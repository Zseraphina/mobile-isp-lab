from pathlib import Path
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
import csv

# 项目根目录
project_root = Path(__file__).resolve().parents[1]

# 先使用Fixed WB图检查24个色块的位置
image_path = Path(
    "/workspace/infinite-isp-baseline/out_frames/"
    "Out_Indoor1_2592x1536_12bit_RGGB_20260822_183551.png"
)

output_path = (
    project_root
    / "results"
    / "11_colorchecker_roi_layout.png"
)


def load_rgb_image(path):
    """读取图片并转换成uint8 RGB数组。"""

    with Image.open(path) as image:
        return np.array(
            image.convert("RGB"),
            dtype=np.uint8
        )


image_rgb = load_rgb_image(image_path)

# 六列色块的中心x坐标
x_centers = [
    628,
    710,
    795,
    880,
    965,
    1050,
]

# 四行色块的中心y坐标
y_centers = [
    1175,
    1260,
    1345,
    1438,
]

# 每个采样框从中心向四周延伸18像素
patch_half_size = 18

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.imshow(image_rgb)

patch_number = 1

for row_index, center_y in enumerate(y_centers):
    for column_index, center_x in enumerate(x_centers):
        x_start = center_x - patch_half_size
        y_start = center_y - patch_half_size

        rectangle = Rectangle(
            (x_start, y_start),
            width=patch_half_size * 2,
            height=patch_half_size * 2,
            fill=False,
            edgecolor="yellow",
            linewidth=2
        )

        ax.add_patch(rectangle)

        ax.text(
            center_x,
            center_y,
            str(patch_number),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "facecolor": "black",
                "alpha": 0.6,
                "pad": 1,
            }
        )

        patch_number += 1

# 只显示ColorChecker附近
ax.set_xlim(550, 1150)

# 图像y坐标向下增加，所以较大的数写在前面
ax.set_ylim(1510, 1090)

ax.set_xlabel("Original image x")
ax.set_ylabel("Original image y")
ax.set_title("ColorChecker 24-Patch ROI Layout")

plt.tight_layout()

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("Image shape:", image_rgb.shape)
print("Number of patches:", patch_number - 1)
print("Saved:", output_path)

# 四种白平衡方法对应的输出图片
evaluation_image_paths = {
    "Fixed WB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Out_Indoor1_2592x1536_12bit_RGGB_20260822_183551.png"
    ),
    "Norm2 AWB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Out_Indoor1_2592x1536_12bit_RGGB_20260825_200548.png"
    ),
    "Gray World": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Out_Indoor1_2592x1536_12bit_RGGB_20260826_234953.png"
    ),
    "PCA AWB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Out_Indoor1_2592x1536_12bit_RGGB_20260827_000854.png"
    ),
}


# ColorChecker Classic 24个色块的参考Lab数值
# 光源：D50；观察者：2°
reference_lab = np.array(
    [
        [37.99,  13.56,  14.06],
        [65.71,  18.13,  17.81],
        [49.93,  -4.88, -21.93],
        [43.14, -13.10,  21.91],
        [55.11,   8.84, -25.40],
        [70.72, -33.40,  -0.20],

        [62.66,  36.07,  57.10],
        [40.02,  10.41, -45.96],
        [51.12,  48.24,  16.25],
        [30.33,  22.98, -21.59],
        [72.53, -23.71,  57.26],
        [71.94,  19.36,  67.86],

        [28.78,  14.18, -50.30],
        [55.26, -38.34,  31.37],
        [42.10,  53.38,  28.19],
        [81.73,   4.04,  79.82],
        [51.94,  49.99, -14.57],
        [51.04, -28.63, -28.64],

        [96.54,  -0.43,   1.19],
        [81.26,  -0.64,  -0.34],
        [66.77,  -0.73,  -0.50],
        [50.87,  -0.15,  -0.27],
        [35.66,  -0.42,  -1.23],
        [20.46,  -0.08,  -0.97],
    ],
    dtype=np.float64
)


patch_names = [
    "Dark Skin",
    "Light Skin",
    "Blue Sky",
    "Foliage",
    "Blue Flower",
    "Bluish Green",
    "Orange",
    "Purplish Blue",
    "Moderate Red",
    "Purple",
    "Yellow Green",
    "Orange Yellow",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Magenta",
    "Cyan",
    "White",
    "Neutral 8",
    "Neutral 6.5",
    "Neutral 5",
    "Neutral 3.5",
    "Black",
]


# 使用已经验证过的24个ROI位置
evaluation_x_centers = [628, 710, 795, 880, 965, 1050]
evaluation_y_centers = [1175, 1260, 1345, 1438]
evaluation_half_size = 18


def load_evaluation_image(image_path):
    """读取一张图片，并转换成uint8 RGB数组。"""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.array(rgb_image, dtype=np.uint8)


def extract_patch_mean_lab(
    image,
    x_centers,
    y_centers,
    half_size
):
    """提取24个色块，并计算每个色块的平均Lab。"""

    patch_mean_lab_values = []

    for y_center in y_centers:
        for x_center in x_centers:
            patch = image[
                y_center - half_size:y_center + half_size,
                x_center - half_size:x_center + half_size
            ]

            # uint8的0～255转换为浮点数0～1
            patch_rgb = (
                patch.astype(np.float64)
                / 255.0
            )

            # 把每个像素从sRGB转换到Lab(D50)
            patch_lab = rgb2lab(
                patch_rgb,
                illuminant="D50",
                observer="2"
            )

            # 对色块的高度和宽度求平均，保留L、a、b
            mean_lab = np.mean(
                patch_lab,
                axis=(0, 1)
            )

            patch_mean_lab_values.append(mean_lab)

    return np.array(
        patch_mean_lab_values,
        dtype=np.float64
    )


delta_e_results = {}

print("\nColorChecker Delta E 2000 evaluation:")

for method_name, image_path in evaluation_image_paths.items():
    evaluation_image = load_evaluation_image(image_path)

    measured_lab = extract_patch_mean_lab(
        evaluation_image,
        evaluation_x_centers,
        evaluation_y_centers,
        evaluation_half_size
    )

    # 对24个色块分别计算CIEDE2000色差
    delta_e = deltaE_ciede2000(
        measured_lab,
        reference_lab
    )

    delta_e_results[method_name] = delta_e

    worst_patch_index = int(np.argmax(delta_e))

    print(f"\n{method_name}:")
    print(
        "  Mean Delta E00:",
        f"{np.mean(delta_e):.3f}"
    )
    print(
        "  Median Delta E00:",
        f"{np.median(delta_e):.3f}"
    )
    print(
        "  Chromatic patches 1-18:",
        f"{np.mean(delta_e[:18]):.3f}"
    )
    print(
        "  Neutral patches 19-24:",
        f"{np.mean(delta_e[18:]):.3f}"
    )
    print(
        "  Worst patch:",
        f"{worst_patch_index + 1} "
        f"{patch_names[worst_patch_index]}"
    )
    print(
        "  Maximum Delta E00:",
        f"{delta_e[worst_patch_index]:.3f}"
    )
  
# 当前项目的results目录
evaluation_results_dir = (
    Path(__file__).resolve().parents[1]
    / "results"
)

evaluation_results_dir.mkdir(
    parents=True,
    exist_ok=True
)

# 保持四种方法当前的排列顺序
method_order = list(delta_e_results.keys())


# ============================================================
# 1. 保存每种方法的汇总指标
# ============================================================

summary_csv_path = (
    evaluation_results_dir
    / "colorchecker_delta_e_summary.csv"
)

summary_fieldnames = [
    "method",
    "mean_delta_e00",
    "median_delta_e00",
    "chromatic_mean_delta_e00",
    "neutral_mean_delta_e00",
    "maximum_delta_e00",
    "worst_patch_number",
    "worst_patch_name",
]

with open(
    summary_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=summary_fieldnames
    )

    writer.writeheader()

    for method_name in method_order:
        delta_e = delta_e_results[method_name]
        worst_index = int(np.argmax(delta_e))

        writer.writerow(
            {
                "method": method_name,
                "mean_delta_e00": np.mean(delta_e),
                "median_delta_e00": np.median(delta_e),
                "chromatic_mean_delta_e00": np.mean(
                    delta_e[:18]
                ),
                "neutral_mean_delta_e00": np.mean(
                    delta_e[18:]
                ),
                "maximum_delta_e00": np.max(delta_e),
                "worst_patch_number": worst_index + 1,
                "worst_patch_name": patch_names[
                    worst_index
                ],
            }
        )

print("Saved:", summary_csv_path)


# ============================================================
# 2. 保存每一个色块的ΔE00
# ============================================================

patch_csv_path = (
    evaluation_results_dir
    / "colorchecker_delta_e_by_patch.csv"
)

with open(
    patch_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "method",
            "patch_number",
            "patch_name",
            "delta_e00",
        ]
    )

    for method_name in method_order:
        delta_e = delta_e_results[method_name]

        for patch_index, patch_delta_e in enumerate(
            delta_e
        ):
            writer.writerow(
                [
                    method_name,
                    patch_index + 1,
                    patch_names[patch_index],
                    patch_delta_e,
                ]
            )

print("Saved:", patch_csv_path)


# ============================================================
# 3. 绘制四种方法的平均ΔE00柱状图
# ============================================================

mean_delta_e_values = np.array(
    [
        np.mean(delta_e_results[method_name])
        for method_name in method_order
    ]
)

summary_fig, summary_ax = plt.subplots(
    figsize=(10, 6)
)

bars = summary_ax.bar(
    method_order,
    mean_delta_e_values,
    color=[
        "gray",
        "steelblue",
        "seagreen",
        "darkorange",
    ]
)

summary_ax.bar_label(
    bars,
    labels=[
        f"{value:.2f}"
        for value in mean_delta_e_values
    ],
    padding=3
)

summary_ax.set_title(
    "Indoor1 ColorChecker Mean Delta E00"
)

summary_ax.set_ylabel(
    "Mean Delta E00"
)

summary_ax.text(
    0.5,
    0.95,
    "Lower is better",
    transform=summary_ax.transAxes,
    horizontalalignment="center"
)

summary_ax.grid(
    axis="y",
    alpha=0.3
)

summary_ax.set_ylim(
    0,
    np.max(mean_delta_e_values) * 1.2
)

summary_fig.tight_layout()

summary_figure_path = (
    evaluation_results_dir
    / "12_colorchecker_delta_e_summary.png"
)

summary_fig.savefig(
    summary_figure_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(summary_fig)

print("Saved:", summary_figure_path)


# ============================================================
# 4. 绘制四种方法×24色块的ΔE00热力图
# ============================================================

delta_e_matrix = np.vstack(
    [
        delta_e_results[method_name]
        for method_name in method_order
    ]
)

heatmap_fig, heatmap_ax = plt.subplots(
    figsize=(16, 4.5)
)

heatmap_image = heatmap_ax.imshow(
    delta_e_matrix,
    cmap="magma",
    aspect="auto",
    interpolation="nearest"
)

heatmap_ax.set_title(
    "ColorChecker Delta E00 by Patch"
)

heatmap_ax.set_xlabel(
    "ColorChecker patch number"
)

heatmap_ax.set_ylabel(
    "AWB method"
)

heatmap_ax.set_xticks(
    np.arange(24)
)

heatmap_ax.set_xticklabels(
    np.arange(1, 25)
)

heatmap_ax.set_yticks(
    np.arange(len(method_order))
)

heatmap_ax.set_yticklabels(
    method_order
)

colorbar = heatmap_fig.colorbar(
    heatmap_image,
    ax=heatmap_ax,
    shrink=0.85
)

colorbar.set_label(
    "Delta E00"
)

heatmap_fig.tight_layout()

heatmap_figure_path = (
    evaluation_results_dir
    / "13_colorchecker_delta_e_by_patch.png"
)

heatmap_fig.savefig(
    heatmap_figure_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(heatmap_fig)

print("Saved:", heatmap_figure_path)