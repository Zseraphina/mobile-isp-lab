from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from skimage.color import rgb2lab, deltaE_ciede2000

# 当前mobile-isp-lab项目根目录
project_root = Path(__file__).resolve().parents[1]

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

# 四种方法的稳定输出路径
image_paths = {
    "Fixed WB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Outdoor1_fixed_wb.png"
    ),
    "Norm2 AWB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Outdoor1_norm2_awb.png"
    ),
    "Gray World": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Outdoor1_gray_world_awb.png"
    ),
    "PCA AWB": Path(
        "/workspace/infinite-isp-baseline/out_frames/"
        "Outdoor1_pca_awb.png"
    ),
}

output_path = (
    project_root
    / "results"
    / "14_outdoor1_awb_comparison.png"
)


def load_rgb_image(image_path):
    """读取PNG，并返回uint8 RGB数组。"""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.array(
            rgb_image,
            dtype=np.uint8
        )


# 使用字典保存四张图片
images = {}

expected_shape = None

for method_name, image_path in image_paths.items():
    image = load_rgb_image(image_path)

    if expected_shape is None:
        expected_shape = image.shape
    elif image.shape != expected_shape:
        raise ValueError(
            "Image shape mismatch: "
            f"{method_name} has {image.shape}, "
            f"expected {expected_shape}"
        )

    images[method_name] = image

    print(
        method_name,
        "shape:",
        image.shape
    )


# 创建2行2列的四图对比
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8)
)

for ax, method_name in zip(
    axes.flat,
    image_paths.keys()
):
    ax.imshow(images[method_name])
    ax.set_title(method_name)
    ax.axis("off")

fig.suptitle(
    "Outdoor1 AWB Comparison",
    fontsize=16
)

plt.tight_layout(
    rect=(0, 0, 1, 0.96)
)

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

print("Saved:", output_path)

# ============================================================
# 放大Outdoor1中的ColorChecker区域
# ============================================================

# 根据原始2592×1536图片估计的色卡附近区域
colorchecker_x_start = 1550
colorchecker_x_end = 1920

colorchecker_y_start = 600
colorchecker_y_end = 860


fixed_wb_image = images["Fixed WB"]

outdoor_colorchecker_crop = fixed_wb_image[
    colorchecker_y_start:colorchecker_y_end,
    colorchecker_x_start:colorchecker_x_end
]


crop_fig, crop_ax = plt.subplots(
    figsize=(12, 7)
)

# extent让坐标显示原始2592×1536图片中的坐标
crop_ax.imshow(
    outdoor_colorchecker_crop,
    extent=[
        colorchecker_x_start,
        colorchecker_x_end,
        colorchecker_y_end,
        colorchecker_y_start,
    ]
)

crop_ax.set_title(
    "Outdoor1 ColorChecker Coordinate Crop"
)

crop_ax.set_xlabel(
    "Original image x"
)

crop_ax.set_ylabel(
    "Original image y"
)

# 每20个像素显示一个坐标刻度
crop_ax.set_xticks(
    np.arange(
        colorchecker_x_start,
        colorchecker_x_end + 1,
        20
    )
)

crop_ax.set_yticks(
    np.arange(
        colorchecker_y_start,
        colorchecker_y_end + 1,
        20
    )
)

crop_ax.grid(
    color="yellow",
    alpha=0.35
)

crop_fig.tight_layout()

crop_output_path = (
    project_root
    / "results"
    / "15_outdoor1_colorchecker_crop.png"
)

crop_fig.savefig(
    crop_output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(crop_fig)

print("Saved:", crop_output_path)

# ============================================================
# 绘制Outdoor1的24个ColorChecker ROI
# ============================================================

outdoor_roi_x_centers = [
    1647,
    1684,
    1719,
    1753,
    1787,
    1818,
]

outdoor_roi_y_centers = [
    682,
    716,
    749,
    781,
]

outdoor_roi_half_size = 6



roi_fig, roi_ax = plt.subplots(
    figsize=(12, 7)
)

roi_ax.imshow(
    outdoor_colorchecker_crop,
    extent=[
        colorchecker_x_start,
        colorchecker_x_end,
        colorchecker_y_end,
        colorchecker_y_start,
    ]
)

patch_number = 1

for y_center in outdoor_roi_y_centers:
    for x_center in outdoor_roi_x_centers:
        rectangle = Rectangle(
            (
                x_center - outdoor_roi_half_size,
                y_center - outdoor_roi_half_size,
            ),
            2 * outdoor_roi_half_size,
            2 * outdoor_roi_half_size,
            fill=False,
            edgecolor="yellow",
            linewidth=2
        )

        roi_ax.add_patch(rectangle)

        roi_ax.text(
            x_center,
            y_center,
            str(patch_number),
            color="white",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "pad": 1,
            }
        )

        patch_number += 1


roi_ax.set_title(
    "Outdoor1 ColorChecker 24-Patch ROI Layout"
)

roi_ax.set_xlabel(
    "Original image x"
)

roi_ax.set_ylabel(
    "Original image y"
)

roi_fig.tight_layout()

roi_output_path = (
    project_root
    / "results"
    / "16_outdoor1_colorchecker_roi_layout.png"
)

roi_fig.savefig(
    roi_output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(roi_fig)

print(
    "Number of Outdoor1 patches:",
    patch_number - 1
)

print("Saved:", roi_output_path)

# ============================================================
# 计算Outdoor1 ColorChecker的Delta E00
# ============================================================


def extract_outdoor_patch_mean_lab(
    image,
    x_centers,
    y_centers,
    half_size
):
    """提取Outdoor1的24个ROI，并计算平均Lab。"""

    patch_mean_lab_values = []

    expected_patch_shape = (
        2 * half_size,
        2 * half_size,
        3
    )

    for y_center in y_centers:
        for x_center in x_centers:
            patch = image[
                y_center - half_size:y_center + half_size,
                x_center - half_size:x_center + half_size
            ]

            # 检查每个ROI是否确实为12×12×3
            if patch.shape != expected_patch_shape:
                raise ValueError(
                    "Unexpected ROI shape: "
                    f"{patch.shape}, "
                    f"expected {expected_patch_shape}"
                )

            # uint8的0～255转换为浮点数0～1
            patch_rgb = (
                patch.astype(np.float64)
                / 255.0
            )

            # 每个像素由sRGB转换成Lab(D50)
            patch_lab = rgb2lab(
                patch_rgb,
                illuminant="D50",
                observer="2"
            )

            # 平均高度和宽度，保留L、a、b
            mean_lab = np.mean(
                patch_lab,
                axis=(0, 1)
            )

            patch_mean_lab_values.append(
                mean_lab
            )

    return np.array(
        patch_mean_lab_values,
        dtype=np.float64
    )


outdoor_delta_e_results = {}

print(
    "\nOutdoor1 ColorChecker "
    "Delta E 2000 evaluation:"
)

for method_name, image in images.items():
    measured_lab = extract_outdoor_patch_mean_lab(
        image,
        outdoor_roi_x_centers,
        outdoor_roi_y_centers,
        outdoor_roi_half_size
    )

    delta_e = deltaE_ciede2000(
        measured_lab,
        reference_lab
    )

    outdoor_delta_e_results[
        method_name
    ] = delta_e

    worst_patch_index = int(
        np.argmax(delta_e)
    )

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

# ============================================================
# 保存Outdoor1 Delta E00结果
# ============================================================

outdoor_results_dir = (
    project_root
    / "results"
)

outdoor_method_order = list(
    outdoor_delta_e_results.keys()
)


# 汇总CSV
outdoor_summary_csv_path = (
    outdoor_results_dir
    / "outdoor1_delta_e_summary.csv"
)

summary_fields = [
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
    outdoor_summary_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=summary_fields
    )

    writer.writeheader()

    for method_name in outdoor_method_order:
        delta_e = outdoor_delta_e_results[
            method_name
        ]

        worst_index = int(
            np.argmax(delta_e)
        )

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

print("Saved:", outdoor_summary_csv_path)


# 逐色块CSV
outdoor_patch_csv_path = (
    outdoor_results_dir
    / "outdoor1_delta_e_by_patch.csv"
)

with open(
    outdoor_patch_csv_path,
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

    for method_name in outdoor_method_order:
        delta_e = outdoor_delta_e_results[
            method_name
        ]

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

print("Saved:", outdoor_patch_csv_path)


# 平均Delta E00柱状图
outdoor_mean_values = np.array(
    [
        np.mean(
            outdoor_delta_e_results[
                method_name
            ]
        )
        for method_name in outdoor_method_order
    ]
)

summary_fig, summary_ax = plt.subplots(
    figsize=(10, 6)
)

bars = summary_ax.bar(
    outdoor_method_order,
    outdoor_mean_values,
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
        for value in outdoor_mean_values
    ],
    padding=3
)

summary_ax.set_title(
    "Outdoor1 ColorChecker Mean Delta E00"
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
    np.max(outdoor_mean_values) * 1.2
)

summary_fig.tight_layout()

summary_image_path = (
    outdoor_results_dir
    / "17_outdoor1_delta_e_summary.png"
)

summary_fig.savefig(
    summary_image_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(summary_fig)

print("Saved:", summary_image_path)


# 逐色块热力图
outdoor_delta_e_matrix = np.vstack(
    [
        outdoor_delta_e_results[
            method_name
        ]
        for method_name in outdoor_method_order
    ]
)

heatmap_fig, heatmap_ax = plt.subplots(
    figsize=(16, 4.5)
)

heatmap_image = heatmap_ax.imshow(
    outdoor_delta_e_matrix,
    cmap="magma",
    aspect="auto",
    interpolation="nearest"
)

heatmap_ax.set_title(
    "Outdoor1 ColorChecker Delta E00 by Patch"
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
    np.arange(len(outdoor_method_order))
)

heatmap_ax.set_yticklabels(
    outdoor_method_order
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

heatmap_image_path = (
    outdoor_results_dir
    / "18_outdoor1_delta_e_by_patch.png"
)

heatmap_fig.savefig(
    heatmap_image_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(heatmap_fig)

print("Saved:", heatmap_image_path)