import argparse
import sys
from pathlib import Path

import numpy as np


infinite_isp_root = Path(
    "/workspace/infinite-isp-baseline"
)

if str(infinite_isp_root) not in sys.path:
    sys.path.insert(
        0,
        str(infinite_isp_root)
    )

from modules.bayer_noise_reduction.joint_bf import JointBF

# ============================================================
# Scene configuration
# ============================================================

baseline_data_dir = (
    infinite_isp_root
    / "in_frames"
    / "normal"
    / "data"
)

bnr_parameters = {
    "is_enable": True,
    "filter_window": 9,
    "r_std_dev_s": 1,
    "r_std_dev_r": 0.1,
    "g_std_dev_s": 1,
    "g_std_dev_r": 0.08,
    "b_std_dev_s": 1,
    "b_std_dev_r": 0.1,
    "is_save": False,
}

bnr_platform = {
    "disable_progress_bar": True,
    "leave_pbar_string": False,
}

scene_configs = {
    "Outdoor1": {
        "raw_filename":
            "Outdoor1_2592x1536_12bit_RGGB.raw",
        "width": 2592,
        "height": 1536,
        "bit_depth": 12,
        "bayer_pattern": "rggb",
        "black_level_offsets": {
            "r": 200,
            "gr": 200,
            "gb": 200,
            "b": 200,
        },
        "underexposed_percentage": 5.0,
        "overexposed_percentage": 5.0,
        "pca_percentage": 3.5,
        "infinite_gains": {
            "gray_world": (
                1.834927908903037,
                2.2641020224020894,
            ),
            "norm2": (
                1.7971737331944668,
                2.1764551340395033,
            ),
            "pca": (
                1.8180660009384155,
                1.8342368602752686,
            ),
        },
    },

    "Indoor1": {
        "raw_filename":
            "Indoor1_2592x1536_12bit_RGGB.raw",
        "width": 2592,
        "height": 1536,
        "bit_depth": 12,
        "bayer_pattern": "rggb",
        "black_level_offsets": {
            "r": 200,
            "gr": 200,
            "gb": 200,
            "b": 200,
        },
        "underexposed_percentage": 5.0,
        "overexposed_percentage": 5.0,
        "pca_percentage": 3.5,
        "infinite_gains": {
            "gray_world": (
                1.3960070859486422,
                2.9887926761019323,
            ),
            "norm2": (
                1.3877439116600867,
                2.955442980094861,
            ),
            "pca": (
                1.3558077812194824,
                2.8518218994140625,
            ),
        },
    },
}


parser = argparse.ArgumentParser(
    description=(
        "Analyze reference-free AWB candidates "
        "for an Infinite-ISP RAW scene."
    )
)

parser.add_argument(
    "--scene",
    choices=scene_configs.keys(),
    default="Outdoor1",
    help="RAW scene to analyze.",
)

args = parser.parse_args()

scene_name = args.scene
scene_config = scene_configs[scene_name]

raw_path = (
    baseline_data_dir
    / scene_config["raw_filename"]
)

width = scene_config["width"]
height = scene_config["height"]
bit_depth = scene_config["bit_depth"]
bayer_pattern = scene_config["bayer_pattern"]

underexposed_percentage = (
    scene_config["underexposed_percentage"]
)

overexposed_percentage = (
    scene_config["overexposed_percentage"]
)

pca_percentage = (
    scene_config["pca_percentage"]
)

print("Scene:", scene_name)
# 读取RAW文件
raw_flat = np.fromfile(
    raw_path,
    dtype=np.uint16
)

# RAW中应该包含的像素数量
expected_pixel_count = (
    width
    * height
)

# 在reshape之前先检查像素数量
if raw_flat.size != expected_pixel_count:
    raise ValueError(
        "Unexpected RAW pixel count: "
        f"expected {expected_pixel_count}, "
        f"got {raw_flat.size}"
    )

# 从一维数组恢复成二维RAW图像
raw = raw_flat.reshape(
    height,
    width
)

# 12-bit数据能够表示的最大值
max_sensor_value = (
    2 ** bit_depth
    - 1
)

# 检查是否存在超过12-bit范围的数值
out_of_range_count = np.count_nonzero(
    raw > max_sensor_value
)

# 计算几个有代表性的分位数
percentiles = np.percentile(
    raw,
    [1, 5, 50, 95, 99]
)


print("RAW path:", raw_path)
print("RAW dtype:", raw.dtype)
print("RAW shape:", raw.shape)
print("RAW pixel count:", raw.size)
print("12-bit maximum:", max_sensor_value)
print("RAW minimum:", np.min(raw))
print("RAW maximum:", np.max(raw))
print("RAW mean:", np.mean(raw))
print("Out-of-range pixels:", out_of_range_count)

print(
    "Percentiles [1%, 5%, 50%, 95%, 99%]:",
    np.round(percentiles, 2)
)

black_level_offsets = (
    scene_config["black_level_offsets"]
)

r_offset = black_level_offsets["r"]
gr_offset = black_level_offsets["gr"]
gb_offset = black_level_offsets["gb"]
b_offset = black_level_offsets["b"]
# 先转成float32，避免uint16减法产生回绕
raw_blc_float = raw.astype(
    np.float32
)

if bayer_pattern != "rggb":
    raise NotImplementedError(
        "The current analyzer only supports RGGB."
    )
# Outdoor1的Bayer排列是RGGB
raw_blc_float[0::2, 0::2] -= r_offset
raw_blc_float[0::2, 1::2] -= gr_offset
raw_blc_float[1::2, 0::2] -= gb_offset
raw_blc_float[1::2, 1::2] -= b_offset

# 限制到12-bit合法范围，再转回uint16
raw_blc = np.clip(
    raw_blc_float,
    0,
    max_sensor_value
).astype(np.uint16)

# 计算黑电平校正后的分位数
blc_percentiles = np.percentile(
    raw_blc,
    [1, 5, 50, 95, 99]
)

print("\nAfter black level correction:")
print("BLC dtype:", raw_blc.dtype)
print("BLC shape:", raw_blc.shape)
print("BLC minimum:", np.min(raw_blc))
print("BLC maximum:", np.max(raw_blc))
print("BLC mean:", np.mean(raw_blc))

print(
    "BLC percentiles [1%, 5%, 50%, 95%, 99%]:",
    np.round(blc_percentiles, 2)
)

print(
    "Mean black level removed:",
    np.mean(raw)
    - np.mean(raw_blc)
)

# 使用Infinite-ISP原版JointBF执行Bayer域降噪
bnr_sensor_info = {
    "width": width,
    "height": height,
    "bit_depth": bit_depth,
    "bayer_pattern": bayer_pattern,
}

joint_bilateral_filter = JointBF(
    raw_blc,
    bnr_sensor_info,
    bnr_parameters,
    bnr_platform,
)

raw_bnr = joint_bilateral_filter.apply_jbf()

bnr_percentiles = np.percentile(
    raw_bnr,
    [1, 5, 50, 95, 99]
)

print("\nAfter Bayer noise reduction:")
print("BNR dtype:", raw_bnr.dtype)
print("BNR shape:", raw_bnr.shape)
print("BNR minimum:", np.min(raw_bnr))
print("BNR maximum:", np.max(raw_bnr))
print("BNR mean:", np.mean(raw_bnr))

print(
    "BNR percentiles [1%, 5%, 50%, 95%, 99%]:",
    np.round(bnr_percentiles, 2)
)
# 从RGGB Bayer RAW中分别提取四类采样点
r_blocks = raw_bnr[
    0::2,
    0::2
].astype(np.float32)

gr_blocks = raw_bnr[
    0::2,
    1::2
].astype(np.float32)

gb_blocks = raw_bnr[
    1::2,
    0::2
].astype(np.float32)

b_blocks = raw_bnr[
    1::2,
    1::2
].astype(np.float32)

# 将同一个2×2 Bayer单元中的两个绿色采样取平均
g_blocks = (
    gr_blocks
    + gb_blocks
) / 2.0

# 检查四类采样点的尺寸是否完全一致
if not (
    r_blocks.shape
    == gr_blocks.shape
    == gb_blocks.shape
    == b_blocks.shape
):
    raise ValueError(
        "RGGB channel shapes do not match"
    )

# 沿新的第三维组合成统计用RGB数组
rgb_blocks = np.stack(
    [
        r_blocks,
        g_blocks,
        b_blocks,
    ],
    axis=2
)

# 对所有Bayer单元求通道平均值
channel_means = np.mean(
    rgb_blocks,
    axis=(0, 1),
    dtype=np.float64
)

# 对所有Bayer单元求通道中位数
channel_medians = np.median(
    rgb_blocks,
    axis=(0, 1)
)

print("\nRGGB block statistics:")
print("R shape:", r_blocks.shape)
print("Gr shape:", gr_blocks.shape)
print("Gb shape:", gb_blocks.shape)
print("B shape:", b_blocks.shape)
print("RGB block shape:", rgb_blocks.shape)
print(
    "Number of Bayer blocks:",
    r_blocks.size
)

print(
    "Channel means [R, G, B]:",
    np.round(channel_means, 2)
)

print(
    "Channel medians [R, G, B]:",
    np.round(channel_medians, 2)
)

# Infinite-ISP AWB配置中的曝光过滤比例

# 12-bit共有4096个数值等级
pixel_level_count = (
    2 ** bit_depth
)

one_percent_level = (
    pixel_level_count
    / 100.0
)

underexposed_limit = (
    underexposed_percentage
    * one_percent_level
)

overexposed_limit = (
    pixel_level_count
    - overexposed_percentage
    * one_percent_level
)

# 任一通道低于下限，则整个Bayer单元属于过暗样本
underexposed_mask = np.any(
    rgb_blocks < underexposed_limit,
    axis=2
)

# 任一通道高于上限，则整个Bayer单元属于过曝样本
overexposed_mask = np.any(
    rgb_blocks > overexposed_limit,
    axis=2
)

# 过暗或过曝都属于无效样本
invalid_mask = (
    underexposed_mask
    | overexposed_mask
)

valid_mask = ~invalid_mask

# 布尔索引后形状为：有效单元数量 × 3
valid_rgb_blocks = rgb_blocks[
    valid_mask
]

total_block_count = rgb_blocks.shape[0] * rgb_blocks.shape[1]

underexposed_count = np.count_nonzero(
    underexposed_mask
)

overexposed_count = np.count_nonzero(
    overexposed_mask
)

invalid_count = np.count_nonzero(
    invalid_mask
)

valid_count = valid_rgb_blocks.shape[0]

valid_percentage = (
    valid_count
    / total_block_count
    * 100.0
)

# 过滤后的通道统计
valid_channel_means = np.mean(
    valid_rgb_blocks,
    axis=0,
    dtype=np.float64
)

valid_channel_medians = np.median(
    valid_rgb_blocks,
    axis=0
)


print("\nExposure filtering:")
print(
    "Underexposed limit:",
    underexposed_limit
)
print(
    "Overexposed limit:",
    overexposed_limit
)
print(
    "Total Bayer blocks:",
    total_block_count
)
print(
    "Underexposed blocks:",
    underexposed_count
)
print(
    "Overexposed blocks:",
    overexposed_count
)
print(
    "Invalid blocks:",
    invalid_count
)
print(
    "Valid blocks:",
    valid_count
)
print(
    "Valid percentage:",
    f"{valid_percentage:.2f}%"
)

print(
    "Valid channel means [R, G, B]:",
    np.round(valid_channel_means, 2)
)

print(
    "Valid channel medians [R, G, B]:",
    np.round(valid_channel_medians, 2)
)

# ============================================================
# Candidate 1: Gray World AWB
# ============================================================

# 与Infinite-ISP Gray World源码保持相同计算方式
gray_world_avg_rgb = np.mean(
    valid_rgb_blocks,
    axis=0
)

gray_world_r_gain = np.nan_to_num(
    gray_world_avg_rgb[1]
    / gray_world_avg_rgb[0]
)

gray_world_b_gain = np.nan_to_num(
    gray_world_avg_rgb[1]
    / gray_world_avg_rgb[2]
)

#Gray World增益
(
    infinite_gray_world_r_gain,
    infinite_gray_world_b_gain,
) = scene_config["infinite_gains"]["gray_world"]

# 与Infinite-ISP结果的绝对差值
r_gain_difference = abs(
    gray_world_r_gain
    - infinite_gray_world_r_gain
)

b_gain_difference = abs(
    gray_world_b_gain
    - infinite_gray_world_b_gain
)

# 相对误差百分比
r_gain_difference_percent = (
    r_gain_difference
    / infinite_gray_world_r_gain
    * 100.0
)

b_gain_difference_percent = (
    b_gain_difference
    / infinite_gray_world_b_gain
    * 100.0
)


print("\nGray World candidate:")

print(
    "Average RGB:",
    np.round(
        gray_world_avg_rgb,
        2
    )
)

print(
    "Our R gain:",
    f"{gray_world_r_gain:.6f}"
)

print(
    "Infinite-ISP R gain:",
    f"{infinite_gray_world_r_gain:.6f}"
)

print(
    "R gain difference:",
    f"{r_gain_difference:.6f}",
    f"({r_gain_difference_percent:.2f}%)"
)

print(
    "Our B gain:",
    f"{gray_world_b_gain:.6f}"
)

print(
    "Infinite-ISP B gain:",
    f"{infinite_gray_world_b_gain:.6f}"
)

print(
    "B gain difference:",
    f"{b_gain_difference:.6f}",
    f"({b_gain_difference_percent:.2f}%)"
)

# ============================================================
# Candidate 2: Norm2 Gray World AWB
# ============================================================

# 转成float64，提高大量平方累加时的数值稳定性
norm2_input = valid_rgb_blocks.astype(
    np.float64
)

# 分别计算R、G、B三个通道的二范数
norm2_rgb = np.linalg.norm(
    norm2_input,
    axis=0
)

norm2_r_gain = np.nan_to_num(
    norm2_rgb[1]
    / norm2_rgb[0]
)

norm2_b_gain = np.nan_to_num(
    norm2_rgb[1]
    / norm2_rgb[2]
)

# 转换成RMS，只用于更直观地显示
channel_rms = (
    norm2_rgb
    / np.sqrt(valid_count)
)

# Infinite-ISP完整流水线输出的Norm2增益
(
    infinite_norm2_r_gain,
    infinite_norm2_b_gain,
) = scene_config["infinite_gains"]["norm2"]

norm2_r_difference = abs(
    norm2_r_gain
    - infinite_norm2_r_gain
)

norm2_b_difference = abs(
    norm2_b_gain
    - infinite_norm2_b_gain
)

norm2_r_difference_percent = (
    norm2_r_difference
    / infinite_norm2_r_gain
    * 100.0
)

norm2_b_difference_percent = (
    norm2_b_difference
    / infinite_norm2_b_gain
    * 100.0
)


print("\nNorm2 candidate:")

print(
    "Channel RMS [R, G, B]:",
    np.round(
        channel_rms,
        2
    )
)

print(
    "Our R gain:",
    f"{norm2_r_gain:.6f}"
)

print(
    "Infinite-ISP R gain:",
    f"{infinite_norm2_r_gain:.6f}"
)

print(
    "R gain difference:",
    f"{norm2_r_difference:.6f}",
    f"({norm2_r_difference_percent:.2f}%)"
)

print(
    "Our B gain:",
    f"{norm2_b_gain:.6f}"
)

print(
    "Infinite-ISP B gain:",
    f"{infinite_norm2_b_gain:.6f}"
)

print(
    "B gain difference:",
    f"{norm2_b_difference:.6f}",
    f"({norm2_b_difference_percent:.2f}%)"
)

# ============================================================
# Candidate 3: PCA AWB
# Part 1: projection and tail-pixel selection
# ============================================================

pca_pixel_percentage = pca_percentage

pca_flat_img = valid_rgb_blocks.astype(
    np.float64
)

pca_sample_count = len(
    pca_flat_img
)

# 计算有效RGB样本的平均值
pca_mean_rgb = np.mean(
    pca_flat_img,
    axis=0
)

# 计算平均RGB向量的长度
pca_mean_magnitude = np.linalg.norm(
    pca_mean_rgb
)

if pca_mean_magnitude == 0:
    raise ValueError(
        "PCA mean RGB vector has zero magnitude"
    )

# 将平均RGB归一化成长度为1的方向向量
pca_mean_vector = (
    pca_mean_rgb
    / pca_mean_magnitude
)

# 计算每个RGB样本在平均颜色方向上的投影
pca_projection = np.sum(
    pca_flat_img
    * pca_mean_vector,
    axis=1
)

# 返回按投影值从小到大排列的样本下标
pca_sorted_indices = np.argsort(
    pca_projection
)

# 每一端选择的样本数量
pca_tail_count = int(
    np.ceil(
        pca_sample_count
        * pca_pixel_percentage
        / 100.0
    )
)

# 投影值最小的一端
pca_dark_indices = pca_sorted_indices[
    :pca_tail_count
]

# 投影值最大的一端
pca_bright_indices = pca_sorted_indices[
    -pca_tail_count:
]

# 合并暗端和亮端样本下标
pca_selected_indices = np.concatenate(
    [
        pca_dark_indices,
        pca_bright_indices,
    ]
)

# 与Infinite-ISP一致，选中后转成float32
pca_selected_data = pca_flat_img[
    pca_selected_indices
].astype(np.float32)

pca_dark_data = pca_flat_img[
    pca_dark_indices
]

pca_bright_data = pca_flat_img[
    pca_bright_indices
]

pca_projection_percentiles = np.percentile(
    pca_projection,
    [0, 3.5, 50, 96.5, 100]
)


print("\nPCA sample selection:")

print(
    "Valid sample count:",
    pca_sample_count
)

print(
    "Pixel percentage per side:",
    pca_pixel_percentage
)

print(
    "Tail sample count per side:",
    pca_tail_count
)

print(
    "Total selected samples:",
    len(pca_selected_data)
)

print(
    "Mean RGB:",
    np.round(
        pca_mean_rgb,
        2
    )
)

print(
    "Mean direction vector:",
    np.round(
        pca_mean_vector,
        6
    )
)

print(
    "Mean-vector magnitude:",
    np.linalg.norm(
        pca_mean_vector
    )
)

print(
    "Projection percentiles "
    "[min, 3.5%, 50%, 96.5%, max]:",
    np.round(
        pca_projection_percentiles,
        2
    )
)

print(
    "Dark-tail mean RGB:",
    np.round(
        np.mean(
            pca_dark_data,
            axis=0
        ),
        2
    )
)

print(
    "Bright-tail mean RGB:",
    np.round(
        np.mean(
            pca_bright_data,
            axis=0
        ),
        2
    )
)

# ============================================================
# Candidate 3: PCA AWB
# Part 2: principal direction and gains
# ============================================================

# X.T @ X，得到3×3颜色关系矩阵
pca_sigma = np.dot(
    pca_selected_data.transpose(),
    pca_selected_data
)

# 计算三个特征值及其对应的特征向量
pca_eigenvalues, pca_eigenvectors = np.linalg.eig(
    pca_sigma
)

# 按特征值从小到大排列
pca_eigenvalue_order = np.argsort(
    pca_eigenvalues
)

pca_sorted_eigenvalues = pca_eigenvalues[
    pca_eigenvalue_order
]

pca_sorted_eigenvectors = pca_eigenvectors[
    :,
    pca_eigenvalue_order
]

# 最后一列对应最大特征值
pca_principal_direction = np.abs(
    pca_sorted_eigenvectors[
        :,
        -1
    ]
)

# 使用G作为参考计算R、B增益
pca_r_gain = np.nan_to_num(
    pca_principal_direction[1]
    / pca_principal_direction[0]
)

pca_b_gain = np.nan_to_num(
    pca_principal_direction[1]
    / pca_principal_direction[2]
)

# Infinite-ISP完整流水线中的PCA结果
(
    infinite_pca_r_gain,
    infinite_pca_b_gain,
) = scene_config["infinite_gains"]["pca"]

pca_r_difference = abs(
    pca_r_gain
    - infinite_pca_r_gain
)

pca_b_difference = abs(
    pca_b_gain
    - infinite_pca_b_gain
)

pca_r_difference_percent = (
    pca_r_difference
    / infinite_pca_r_gain
    * 100.0
)

pca_b_difference_percent = (
    pca_b_difference
    / infinite_pca_b_gain
    * 100.0
)


print("\nPCA candidate:")

# 数值很大，除以10^9仅用于方便显示
print(
    "Sigma matrix / 1e9:"
)

print(
    np.round(
        pca_sigma / 1e9,
        3
    )
)

print(
    "Eigenvalues / 1e9:",
    np.round(
        pca_sorted_eigenvalues / 1e9,
        3
    )
)

print(
    "Principal direction [R, G, B]:",
    np.round(
        pca_principal_direction,
        6
    )
)

print(
    "Our R gain:",
    f"{pca_r_gain:.6f}"
)

print(
    "Infinite-ISP R gain:",
    f"{infinite_pca_r_gain:.6f}"
)

print(
    "R gain difference:",
    f"{pca_r_difference:.6f}",
    f"({pca_r_difference_percent:.2f}%)"
)

print(
    "Our B gain:",
    f"{pca_b_gain:.6f}"
)

print(
    "Infinite-ISP B gain:",
    f"{infinite_pca_b_gain:.6f}"
)

print(
    "B gain difference:",
    f"{pca_b_difference:.6f}",
    f"({pca_b_difference_percent:.2f}%)"
)

# ============================================================
# Reusable AWB candidate functions
# ============================================================

def apply_awb_gain_floor(gain):
    """与Infinite-ISP一致，AWB增益最低限制为1。"""

    gain = float(
        np.nan_to_num(gain)
    )

    if gain <= 1.0:
        return 1.0

    return gain


def calculate_gray_world_gains(rgb_samples):
    """使用通道平均值计算Gray World增益。"""

    average_rgb = np.mean(
        rgb_samples,
        axis=0
    )

    r_gain = apply_awb_gain_floor(
        average_rgb[1]
        / average_rgb[0]
    )

    b_gain = apply_awb_gain_floor(
        average_rgb[1]
        / average_rgb[2]
    )

    return np.array(
        [r_gain, b_gain],
        dtype=np.float64
    )


def calculate_norm2_gains(rgb_samples):
    """使用通道二范数计算Norm2增益。"""

    norm_rgb = np.linalg.norm(
        rgb_samples.astype(np.float64),
        axis=0
    )

    r_gain = apply_awb_gain_floor(
        norm_rgb[1]
        / norm_rgb[0]
    )

    b_gain = apply_awb_gain_floor(
        norm_rgb[1]
        / norm_rgb[2]
    )

    return np.array(
        [r_gain, b_gain],
        dtype=np.float64
    )


def calculate_pca_gains(
    rgb_samples,
    pixel_percentage
):
    """使用Infinite-ISP相同的PCA流程计算增益。"""

    flat_img = rgb_samples.astype(
        np.float64
    )

    sample_count = len(flat_img)

    if sample_count == 0:
        raise ValueError(
            "PCA received no RGB samples"
        )

    mean_rgb = np.mean(
        flat_img,
        axis=0
    )

    mean_magnitude = np.linalg.norm(
        mean_rgb
    )

    if mean_magnitude == 0:
        raise ValueError(
            "PCA mean RGB vector has zero magnitude"
        )

    mean_vector = (
        mean_rgb
        / mean_magnitude
    )

    projection = np.sum(
        flat_img * mean_vector,
        axis=1
    )

    sorted_indices = np.argsort(
        projection
    )

    tail_count = int(
        np.ceil(
            sample_count
            * pixel_percentage
            / 100.0
        )
    )

    tail_count = max(
        1,
        tail_count
    )

    selected_indices = np.concatenate(
        [
            sorted_indices[:tail_count],
            sorted_indices[-tail_count:],
        ]
    )

    selected_data = flat_img[
        selected_indices
    ].astype(np.float32)

    sigma = np.dot(
        selected_data.transpose(),
        selected_data
    )

    eigenvalues, eigenvectors = np.linalg.eig(
        sigma
    )

    eigenvalue_order = np.argsort(
        eigenvalues
    )

    sorted_eigenvectors = eigenvectors[
        :,
        eigenvalue_order
    ]

    principal_direction = np.abs(
        sorted_eigenvectors[:, -1]
    )

    r_gain = apply_awb_gain_floor(
        principal_direction[1]
        / principal_direction[0]
    )

    b_gain = apply_awb_gain_floor(
        principal_direction[1]
        / principal_direction[2]
    )

    return np.array(
        [r_gain, b_gain],
        dtype=np.float64
    )

# --------------------------------------------------
# Candidate disagreement
# --------------------------------------------------

candidate_names = [
    "Gray World",
    "Norm2",
    "PCA",
]

candidate_gains = np.array(
    [
        [gray_world_r_gain, gray_world_b_gain],
        [norm2_r_gain, norm2_b_gain],
        [pca_r_gain, pca_b_gain],
    ],
    dtype=np.float64,
)

helper_candidate_gains = np.vstack(
    [
        calculate_gray_world_gains(
            valid_rgb_blocks
        ),
        calculate_norm2_gains(
            valid_rgb_blocks
        ),
        calculate_pca_gains(
            valid_rgb_blocks,
            pca_percentage
        ),
    ]
)

if not np.allclose(
    helper_candidate_gains,
    candidate_gains,
    rtol=1e-7,
    atol=1e-9
):
    raise ValueError(
        "Reusable candidate functions do not "
        "match the original calculations: "
        f"{helper_candidate_gains} vs "
        f"{candidate_gains}"
    )

print("\nCandidate helper self-check:")
print(
    "Maximum absolute gain difference:",
    np.max(
        np.abs(
            helper_candidate_gains
            - candidate_gains
        )
    )
)

# ============================================================
# Spatial candidate stability diagnostic
# ============================================================

# 4×6网格使每个区域在原图中接近方形
stability_grid_rows = 4
stability_grid_columns = 6
minimum_tile_valid_samples = 2000

row_edges = np.linspace(
    0,
    rgb_blocks.shape[0],
    stability_grid_rows + 1,
    dtype=int
)

column_edges = np.linspace(
    0,
    rgb_blocks.shape[1],
    stability_grid_columns + 1,
    dtype=int
)

tile_candidate_gain_lists = {
    name: []
    for name in candidate_names
}

valid_stability_tile_count = 0
skipped_stability_tile_count = 0

for row_index in range(stability_grid_rows):
    row_start = row_edges[row_index]
    row_end = row_edges[row_index + 1]

    for column_index in range(
        stability_grid_columns
    ):
        column_start = column_edges[
            column_index
        ]
        column_end = column_edges[
            column_index + 1
        ]

        tile_rgb_blocks = rgb_blocks[
            row_start:row_end,
            column_start:column_end,
        ]

        tile_valid_mask = valid_mask[
            row_start:row_end,
            column_start:column_end,
        ]

        tile_valid_rgb = tile_rgb_blocks[
            tile_valid_mask
        ]

        if (
            len(tile_valid_rgb)
            < minimum_tile_valid_samples
        ):
            skipped_stability_tile_count += 1
            continue

        tile_candidate_gains = np.vstack(
            [
                calculate_gray_world_gains(
                    tile_valid_rgb
                ),
                calculate_norm2_gains(
                    tile_valid_rgb
                ),
                calculate_pca_gains(
                    tile_valid_rgb,
                    pca_percentage
                ),
            ]
        )

        for candidate_index, name in enumerate(
            candidate_names
        ):
            tile_candidate_gain_lists[
                name
            ].append(
                tile_candidate_gains[
                    candidate_index
                ]
            )

        valid_stability_tile_count += 1

if valid_stability_tile_count == 0:
    raise ValueError(
        "No spatial tiles contained enough "
        "valid AWB samples"
    )

spatial_stability_metrics = {}

print("\nSpatial candidate stability:")
print(
    "Grid:",
    f"{stability_grid_rows}x"
    f"{stability_grid_columns}"
)
print(
    "Minimum valid samples per tile:",
    minimum_tile_valid_samples
)
print(
    "Valid tiles:",
    valid_stability_tile_count
)
print(
    "Skipped tiles:",
    skipped_stability_tile_count
)

for candidate_index, name in enumerate(
    candidate_names
):
    tile_gains = np.asarray(
        tile_candidate_gain_lists[name],
        dtype=np.float64
    )

    global_gains = candidate_gains[
        candidate_index
    ]

    # 对数比值能对称处理增益变大和变小
    tile_log_gain_ratio = np.log(
        tile_gains
        / global_gains
    )

    # 同时综合R、B两个增益的相对波动
    tile_log_distance_percent = (
        np.sqrt(
            np.mean(
                tile_log_gain_ratio ** 2,
                axis=1
            )
        )
        * 100.0
    )

    median_distance_percent = np.median(
        tile_log_distance_percent
    )

    p90_distance_percent = np.percentile(
        tile_log_distance_percent,
        90
    )

    tile_gain_percentiles = np.percentile(
        tile_gains,
        [10, 50, 90],
        axis=0
    )

    spatial_stability_metrics[name] = {
        "median_distance_percent":
            median_distance_percent,
        "p90_distance_percent":
            p90_distance_percent,
    }

    print(f"\n{name}:")
    print(
        "  Tile R gain [P10, median, P90]:",
        np.round(
            tile_gain_percentiles[:, 0],
            6
        )
    )
    print(
        "  Tile B gain [P10, median, P90]:",
        np.round(
            tile_gain_percentiles[:, 1],
            6
        )
    )
    print(
        "  Median log-gain distance:",
        f"{median_distance_percent:.2f}%"
    )
    print(
        "  P90 log-gain distance:",
        f"{p90_distance_percent:.2f}%"
    )


# ============================================================
# Fixed bright-tail neutrality diagnostic
# ============================================================

# 使用PCA投影阶段已经选出的固定亮端样本。
# 这批样本不随候选增益变化。
fixed_bright_samples = pca_bright_data.astype(
    np.float64
)

# 为保证三种候选评价完全相同的像素，
# 只保留在所有候选增益下都不会超过12-bit上限的样本。
maximum_candidate_r_gain = np.max(
    candidate_gains[:, 0]
)

maximum_candidate_b_gain = np.max(
    candidate_gains[:, 1]
)

common_bright_safe_mask = (
    (
        fixed_bright_samples[:, 0]
        * maximum_candidate_r_gain
        <= max_sensor_value
    )
    & (
        fixed_bright_samples[:, 1]
        <= max_sensor_value
    )
    & (
        fixed_bright_samples[:, 2]
        * maximum_candidate_b_gain
        <= max_sensor_value
    )
)

common_bright_samples = fixed_bright_samples[
    common_bright_safe_mask
]

if len(common_bright_samples) == 0:
    raise ValueError(
        "No common unclipped bright-tail samples"
    )

fixed_bright_tail_metrics = {}

print("\nFixed bright-tail neutrality:")
print(
    "Original bright-tail samples:",
    len(fixed_bright_samples)
)
print(
    "Common unclipped samples:",
    len(common_bright_samples)
)

common_bright_sample_percentage = (
    len(common_bright_samples)
    / len(fixed_bright_samples)
    * 100.0
)

print(
    "Common sample percentage:",
    f"{common_bright_sample_percentage:.2f}%"
)

for candidate_index, name in enumerate(
    candidate_names
):
    gains = candidate_gains[
        candidate_index
    ]

    # 使用完整的固定亮端，并保持线性值不裁剪
    corrected_bright_samples = (
        fixed_bright_samples.copy()
    )

    corrected_bright_samples[:, 0] *= gains[0]
    corrected_bright_samples[:, 2] *= gains[1]

    bright_clipped_sample_mask = np.any(
       corrected_bright_samples
       > max_sensor_value,
       axis=1
    )

    bright_clipped_sample_percentage = (
        np.count_nonzero(
        bright_clipped_sample_mask
        )
        / len(corrected_bright_samples)
        * 100.0
    )

    corrected_mean_rgb = np.mean(
        corrected_bright_samples,
        axis=0
    )

    corrected_mean_level = np.mean(
        corrected_mean_rgb
    )

    aggregate_spread_percent = (
        (
            np.max(corrected_mean_rgb)
            - np.min(corrected_mean_rgb)
        )
        / corrected_mean_level
        * 100.0
    )

    per_pixel_mean_level = np.mean(
        corrected_bright_samples,
        axis=1
    )

    per_pixel_residual_percent = (
        (
            np.max(
                corrected_bright_samples,
                axis=1
            )
            - np.min(
                corrected_bright_samples,
                axis=1
            )
        )
        / np.maximum(
            per_pixel_mean_level,
            np.finfo(np.float64).eps
        )
        * 100.0
    )

    median_residual_percent = np.median(
        per_pixel_residual_percent
    )

    p90_residual_percent = np.percentile(
        per_pixel_residual_percent,
        90
    )

    fixed_bright_tail_metrics[name] = {
        "aggregate_spread_percent":
            aggregate_spread_percent,
        "median_residual_percent":
            median_residual_percent,
        "p90_residual_percent":
            p90_residual_percent,
        "clipped_sample_percentage":
            bright_clipped_sample_percentage,
    }

    print(f"\n{name}:")
    print(
        "  Corrected mean RGB:",
        np.round(
            corrected_mean_rgb,
            2
        )
    )
    print(
        "  Aggregate channel spread:",
        f"{aggregate_spread_percent:.2f}%"
    )
    print(
        "  Median per-pixel residual:",
        f"{median_residual_percent:.2f}%"
    )
    print(
        "  P90 per-pixel residual:",
        f"{p90_residual_percent:.2f}%"
    )
    print(
        "  Bright samples that would clip:",
        f"{bright_clipped_sample_percentage:.2f}%"
    )


# 三种算法的平均增益
candidate_mean_gains = np.mean(
    candidate_gains,
    axis=0,
)

# ============================================================
# Fixed bright-tail scene evidence
# ============================================================

# 使用三种候选的平均增益进行临时校正。
# 它不代表最终白平衡，只用于判断固定亮端中
# 是否存在中性表面，以及彩色表面是否足够多样。
bright_consensus_samples = fixed_bright_samples.copy()

bright_consensus_samples[:, 0] *= candidate_mean_gains[0]
bright_consensus_samples[:, 2] *= candidate_mean_gains[1]

# 每个亮端样本的平均通道值，用于消除亮度尺度。
bright_consensus_levels = np.mean(
    bright_consensus_samples,
    axis=1,
)

# 每个样本三个通道之间的相对分离程度。
bright_consensus_residual = (
    np.max(bright_consensus_samples, axis=1)
    - np.min(bright_consensus_samples, axis=1)
) / np.maximum(
    bright_consensus_levels,
    np.finfo(np.float64).eps,
)

# 通道分离不超过10%的亮端样本，
# 暂时视为候选中性亮端。
bright_neutral_threshold = 0.10

bright_neutral_mask = (
    bright_consensus_residual
    <= bright_neutral_threshold
)
bright_chromatic_mask = ~bright_neutral_mask

bright_neutral_percentage = (
    np.mean(bright_neutral_mask) * 100.0
)
bright_chromatic_percentage = (
    np.mean(bright_chromatic_mask) * 100.0
)

bright_chromatic_samples = bright_consensus_samples[
    bright_chromatic_mask
]

if len(bright_chromatic_samples) > 0:
    # 对明显有色的亮端样本，记录其最大通道。
    bright_dominant_channels = np.argmax(
        bright_chromatic_samples,
        axis=1,
    )
    bright_dominant_counts = np.bincount(
        bright_dominant_channels,
        minlength=3,
    )
    bright_dominant_shares = (
        bright_dominant_counts
        / np.sum(bright_dominant_counts)
    )

    nonzero_share_mask = bright_dominant_shares > 0

    # 除以log(3)，将熵归一化到0至1。
    bright_color_diversity = (
        -np.sum(
            bright_dominant_shares[nonzero_share_mask]
            * np.log(
                bright_dominant_shares[
                    nonzero_share_mask
                ]
            )
        )
        / np.log(3.0)
    )
else:
    bright_dominant_shares = np.zeros(
        3,
        dtype=np.float64,
    )
    bright_color_diversity = 0.0

print("\nFixed bright-tail scene evidence:")
print(
    "Neutral bright samples:",
    f"{bright_neutral_percentage:.2f}%"
)
print(
    "Chromatic bright samples:",
    f"{bright_chromatic_percentage:.2f}%"
)
print(
    "Chromatic dominant shares [R, G, B]:",
    np.round(bright_dominant_shares * 100.0, 2)
)
print(
    "Bright color diversity:",
    f"{bright_color_diversity * 100.0:.2f}%"
)

candidate_min_gains = np.min(
    candidate_gains,
    axis=0,
)

candidate_max_gains = np.max(
    candidate_gains,
    axis=0,
)

# 增益最大值和最小值的差距，相对于平均增益有多大
relative_gain_range = (
    candidate_max_gains - candidate_min_gains
) / candidate_mean_gains

# R、B两个通道分歧的平均值
overall_disagreement = np.mean(
    relative_gain_range
)

# 每个候选与三者平均增益的相对距离
relative_distance_to_mean = (
    np.abs(candidate_gains - candidate_mean_gains)
    / candidate_mean_gains
)

candidate_mean_distance = np.mean(
    relative_distance_to_mean,
    axis=1,
)

print("\nCandidate disagreement:")
print("Candidate gains [R gain, B gain]:")

for name, gains in zip(
    candidate_names,
    candidate_gains,
):
    print(
        f"  {name}:",
        np.round(gains, 6),
    )

print(
    "Mean gains [R, B]:",
    np.round(candidate_mean_gains, 6),
)

print(
    f"R gain relative range: "
    f"{relative_gain_range[0] * 100:.2f}%"
)

print(
    f"B gain relative range: "
    f"{relative_gain_range[1] * 100:.2f}%"
)

print(
    f"Overall candidate disagreement: "
    f"{overall_disagreement * 100:.2f}%"
)

print("Candidate distance from mean:")

for name, distance in zip(
    candidate_names,
    candidate_mean_distance,
):
    print(
        f"  {name}: {distance * 100:.2f}%"
    )
  
# --------------------------------------------------
# Highlight clipping risk
# --------------------------------------------------

sensor_max = (2**bit_depth) - 1

total_blocks = (
    rgb_blocks.shape[0]
    * rgb_blocks.shape[1]
)

total_raw_samples = raw_bnr.size

clipping_results = []

for method_name, gains in zip(
    candidate_names,
    candidate_gains,
):
    r_gain = gains[0]
    b_gain = gains[1]

    # 模拟Infinite-ISP应用白平衡增益
    r_after_wb = (
        rgb_blocks[:, :, 0].astype(np.float64)
        * r_gain
    )

    b_after_wb = (
        rgb_blocks[:, :, 2].astype(np.float64)
        * b_gain
    )

    # 恰好等于4095仍可保存，只有大于4095才会被clip
    r_clipped = r_after_wb > sensor_max
    b_clipped = b_after_wb > sensor_max

    # 同一个2×2 Bayer块中，R或B任意一个发生裁剪
    any_clipped = r_clipped | b_clipped

    r_clipped_count = np.count_nonzero(
        r_clipped
    )

    b_clipped_count = np.count_nonzero(
        b_clipped
    )

    any_clipped_count = np.count_nonzero(
        any_clipped
    )

    r_clipped_percent = (
        r_clipped_count
        / total_blocks
        * 100
    )

    b_clipped_percent = (
        b_clipped_count
        / total_blocks
        * 100
    )

    block_clipped_percent = (
        any_clipped_count
        / total_blocks
        * 100
    )

    # 每个Bayer块共有R、Gr、Gb、B四个RAW采样点
    raw_sample_clipped_percent = (
        (r_clipped_count + b_clipped_count)
        / total_raw_samples
        * 100
    )

    clipping_results.append(
        {
            "name": method_name,
            "r_clipped_percent": r_clipped_percent,
            "b_clipped_percent": b_clipped_percent,
            "block_clipped_percent": block_clipped_percent,
            "raw_sample_clipped_percent":
                raw_sample_clipped_percent,
        }
    )


print("\nHighlight clipping risk:")
print("Sensor maximum:", sensor_max)

for result in clipping_results:
    print(f"\n{result['name']}:")

    print(
        "  R samples clipped:",
        f"{result['r_clipped_percent']:.3f}%"
    )

    print(
        "  B samples clipped:",
        f"{result['b_clipped_percent']:.3f}%"
    )

    print(
        "  Bayer blocks with any clipping:",
        f"{result['block_clipped_percent']:.3f}%"
    )

    print(
        "  All RAW samples clipped:",
        f"{result['raw_sample_clipped_percent']:.3f}%"
    )

# --------------------------------------------------
# Neutral-pixel support
# --------------------------------------------------

neutral_threshold = 0.10
neutral_tail_fraction = 0.05

neutral_support_results = []

for method_name, gains in zip(
    candidate_names,
    candidate_gains,
):
    r_gain = gains[0]
    b_gain = gains[1]

    # 对曝光筛选后的有效RGB样本应用候选增益
    corrected_valid_rgb = (
        valid_rgb_blocks.astype(np.float64).copy()
    )

    corrected_valid_rgb[:, 0] *= r_gain
    corrected_valid_rgb[:, 2] *= b_gain

    # 每个RGB样本的平均亮度水平
    sample_mean = np.mean(
        corrected_valid_rgb,
        axis=1,
    )

    # 每个样本三个通道的最大值与最小值之差
    sample_channel_range = (
        np.max(corrected_valid_rgb, axis=1)
        - np.min(corrected_valid_rgb, axis=1)
    )

    # 相对于该样本平均值的通道分离程度
    neutral_residual = (
        sample_channel_range
        / (sample_mean + 1e-12)
    )

    # 通道分离不超过10%，暂时认为是候选中性像素
    neutral_mask = (
        neutral_residual <= neutral_threshold
    )

    neutral_support_percent = (
        np.mean(neutral_mask)
        * 100
    )

    # 取残差最低的5%，观察最可能的中性像素有多中性
    tail_count = int(
        np.ceil(
            len(neutral_residual)
            * neutral_tail_fraction
        )
    )

    sorted_residual = np.sort(
        neutral_residual
    )

    best_tail_mean_percent = (
        np.mean(sorted_residual[:tail_count])
        * 100
    )

    median_residual_percent = (
        np.median(neutral_residual)
        * 100
    )

    neutral_support_results.append(
        {
            "name": method_name,
            "neutral_support_percent":
                neutral_support_percent,
            "best_tail_mean_percent":
                best_tail_mean_percent,
            "median_residual_percent":
                median_residual_percent,
        }
    )


print("\nNeutral-pixel support:")
print(
    "Neutral residual threshold:",
    f"{neutral_threshold * 100:.1f}%"
)

for result in neutral_support_results:
    print(f"\n{result['name']}:")

    print(
        "  Neutral-support pixels:",
        f"{result['neutral_support_percent']:.3f}%"
    )

    print(
        "  Best 5% mean residual:",
        f"{result['best_tail_mean_percent']:.3f}%"
    )

    print(
        "  Full valid-set median residual:",
        f"{result['median_residual_percent']:.3f}%"
    )

# --------------------------------------------------
# Dominant-color concentration
# --------------------------------------------------

# 使用三种候选的平均增益做临时共识校正
# 不提前指定Gray World、Norm2或PCA谁是正确答案
consensus_rgb = (
    valid_rgb_blocks.astype(np.float64).copy()
)

consensus_rgb[:, 0] *= candidate_mean_gains[0]
consensus_rgb[:, 2] *= candidate_mean_gains[1]

consensus_sample_mean = np.mean(
    consensus_rgb,
    axis=1,
)

consensus_channel_range = (
    np.max(consensus_rgb, axis=1)
    - np.min(consensus_rgb, axis=1)
)

consensus_color_residual = (
    consensus_channel_range
    / (consensus_sample_mean + 1e-12)
)

# 通道分离超过20%的样本，暂时认为是明显有颜色的样本
chromatic_threshold = 0.20

chromatic_mask = (
    consensus_color_residual
    > chromatic_threshold
)

chromatic_rgb = consensus_rgb[
    chromatic_mask
]

chromatic_count = len(chromatic_rgb)

if chromatic_count == 0:
    raise ValueError(
        "No chromatic samples were found."
    )

# 每个彩色样本中，找出数值最大的通道
# 0代表R，1代表G，2代表B
dominant_channel_indices = np.argmax(
    chromatic_rgb,
    axis=1,
)

dominant_channel_counts = np.bincount(
    dominant_channel_indices,
    minlength=3,
)

dominant_channel_shares = (
    dominant_channel_counts
    / chromatic_count
)

channel_names = ["R", "G", "B"]

most_common_channel_index = int(
    np.argmax(dominant_channel_shares)
)

most_common_channel = channel_names[
    most_common_channel_index
]

maximum_channel_share = dominant_channel_shares[
    most_common_channel_index
]

# 使用归一化熵衡量分布集中程度
# 三个通道同样常见时接近0%
# 全部集中在一个通道时接近100%
nonzero_shares = dominant_channel_shares[
    dominant_channel_shares > 0
]

normalized_entropy = (
    -np.sum(
        nonzero_shares
        * np.log(nonzero_shares)
    )
    / np.log(3)
)

entropy_concentration = (
    1 - normalized_entropy
)

print("\nDominant-color concentration:")

print(
    "Consensus gains [R, B]:",
    np.round(candidate_mean_gains, 6)
)

print(
    "Consensus-corrected channel means [R, G, B]:",
    np.round(
        np.mean(consensus_rgb, axis=0),
        2
    )
)

print(
    "Chromatic residual threshold:",
    f"{chromatic_threshold * 100:.1f}%"
)

print(
    "Chromatic samples:",
    chromatic_count
)

print(
    "Chromatic sample percentage:",
    f"{np.mean(chromatic_mask) * 100:.2f}%"
)

for channel_name, channel_share in zip(
    channel_names,
    dominant_channel_shares,
):
    print(
        f"{channel_name}-dominant share:",
        f"{channel_share * 100:.2f}%"
    )

print(
    "Most common dominant channel:",
    most_common_channel
)

print(
    "Maximum dominant-channel share:",
    f"{maximum_channel_share * 100:.2f}%"
)

print(
    "Entropy concentration:",
    f"{entropy_concentration * 100:.2f}%"
)

# --------------------------------------------------
# Prototype reliability score
# --------------------------------------------------

clipping_by_name = {
    result["name"]: result
    for result in clipping_results
}

neutral_by_name = {
    result["name"]: result
    for result in neutral_support_results
}

neutral_support_values = np.array(
    [
        neutral_by_name[name][
            "neutral_support_percent"
        ]
        for name in candidate_names
    ],
    dtype=np.float64,
)

neutral_tail_residual_values = np.array(
    [
        neutral_by_name[name][
            "best_tail_mean_percent"
        ]
        for name in candidate_names
    ],
    dtype=np.float64,
)

clipping_values = np.array(
    [
        clipping_by_name[name][
            "block_clipped_percent"
        ]
        for name in candidate_names
    ],
    dtype=np.float64,
)


def normalize_higher_is_better(values):
    """把越高越好的指标归一化到0～1。"""

    value_min = np.min(values)
    value_max = np.max(values)
    value_range = value_max - value_min

    if value_range < 1e-12:
        return np.full_like(
            values,
            0.5,
            dtype=np.float64,
        )

    return (
        values - value_min
    ) / value_range


def normalize_lower_is_better(values):
    """把越低越好的指标归一化到0～1。"""

    value_min = np.min(values)
    value_max = np.max(values)
    value_range = value_max - value_min

    if value_range < 1e-12:
        return np.full_like(
            values,
            0.5,
            dtype=np.float64,
        )

    return (
        value_max - values
    ) / value_range


# --------------------------------------------------
# Fixed-scale reliability components
# --------------------------------------------------

# 中性支持度本来就是0～100%，直接转换成0～1
neutral_support_scores = np.clip(
    neutral_support_values / 100.0,
    0.0,
    1.0,
)

# 最佳中性像素残差：
# 0%残差得1分，达到10%残差时得0分
neutral_tail_limit_percent = (
    neutral_threshold * 100.0
)

neutral_tail_scores = np.clip(
    1.0
    - (
        neutral_tail_residual_values
        / neutral_tail_limit_percent
    ),
    0.0,
    1.0,
)

# Bayer块裁剪率：
# 0%裁剪得1分，达到5%裁剪时得0分
clipping_limit_percent = 5.0

clipping_scores = np.clip(
    1.0
    - (
        clipping_values
        / clipping_limit_percent
    ),
    0.0,
    1.0,
)

neutral_support_weight = 0.50
neutral_tail_weight = 0.30
clipping_weight = 0.20

reliability_scores = (
    neutral_support_weight
    * neutral_support_scores

    + neutral_tail_weight
    * neutral_tail_scores

    + clipping_weight
    * clipping_scores
)

selected_candidate_index = int(
    np.argmax(reliability_scores)
)

selected_candidate_name = candidate_names[
    selected_candidate_index
]

selected_candidate_gains = candidate_gains[
    selected_candidate_index
]

score_order = np.argsort(
    reliability_scores
)[::-1]

second_candidate_index = int(
    score_order[1]
)

reliability_score_margin = (
    reliability_scores[selected_candidate_index]
    - reliability_scores[second_candidate_index]
)

print("\nFixed-scale reliability score:")

print(
    "Reliability score margin:",
    f"{reliability_score_margin:.4f}"
)

print(
    "Weights [neutral support, neutral tail, clipping]:",
    [
        neutral_support_weight,
        neutral_tail_weight,
        clipping_weight,
    ]
)

for index, method_name in enumerate(
    candidate_names
):
    print(f"\n{method_name}:")

    print(
        "  Neutral-support component:",
        f"{neutral_support_scores[index]:.4f}"
    )

    print(
        "  Neutral-tail component:",
        f"{neutral_tail_scores[index]:.4f}"
    )

    print(
        "  Clipping component:",
        f"{clipping_scores[index]:.4f}"
    )

    print(
        "  Reliability score:",
        f"{reliability_scores[index]:.4f}"
    )

print(
    "\nSelected AWB candidate:",
    selected_candidate_name
)

print(
    "Selected gains [R, B]:",
    np.round(
        selected_candidate_gains,
        6
    )
)

print(
    "Scene-level candidate disagreement:",
    f"{overall_disagreement * 100:.2f}%"
)

# --------------------------------------------------
# Reliability-aware hybrid AWB
# --------------------------------------------------

# 候选增益平均分歧达到5%，认为算法意见明显不同
hard_disagreement_threshold = 0.05

# 第一名比第二名至少高0.02，认为赢家比较明确
hard_margin_threshold = 0.02

# Softmax温度越低，越偏向最高分候选
softmax_temperature = 0.05

use_hard_selection = (
    overall_disagreement
    >= hard_disagreement_threshold
    and reliability_score_margin
    >= hard_margin_threshold
)

if use_hard_selection:
    decision_mode = "Hard selection"

    fusion_weights = np.zeros(
        len(candidate_names),
        dtype=np.float64,
    )

    fusion_weights[
        selected_candidate_index
    ] = 1.0

else:
    decision_mode = "Soft fusion"

    # 减去最大值，避免指数运算数值过大
    shifted_scores = (
        reliability_scores
        - np.max(reliability_scores)
    )

    softmax_values = np.exp(
        shifted_scores
        / softmax_temperature
    )

    fusion_weights = (
        softmax_values
        / np.sum(softmax_values)
    )

# candidate_gains形状是(3, 2)
# fusion_weights[:, np.newaxis]形状是(3, 1)
# 广播后，每组[R gain, B gain]乘以自己的权重
hybrid_gains = np.sum(
    candidate_gains
    * fusion_weights[:, np.newaxis],
    axis=0,
)

hybrid_r_gain = hybrid_gains[0]
hybrid_b_gain = hybrid_gains[1]


print("\nReliability-aware Hybrid AWB:")

print(
    "Decision mode:",
    decision_mode
)

print(
    "Disagreement threshold:",
    f"{hard_disagreement_threshold * 100:.2f}%"
)

print(
    "Score-margin threshold:",
    f"{hard_margin_threshold:.4f}"
)

print(
    "Softmax temperature:",
    softmax_temperature
)

print("Fusion weights:")

for method_name, weight in zip(
    candidate_names,
    fusion_weights,
):
    print(
        f"  {method_name}:",
        f"{weight:.4f}"
    )

print(
    "Hybrid gains [R, B]:",
    np.round(hybrid_gains, 6)
)

# ============================================================
# Hybrid V3: bright-tail evidence score diagnostic
# ============================================================

# 按candidate_names的顺序从亮端指标字典中取值，
# 保证与candidate_gains和reliability_scores逐项对应。
bright_aggregate_spread_values = np.array(
    [
        fixed_bright_tail_metrics[name][
            "aggregate_spread_percent"
        ]
        for name in candidate_names
    ],
    dtype=np.float64,
)

bright_clipping_values = np.array(
    [
        fixed_bright_tail_metrics[name][
            "clipped_sample_percentage"
        ]
        for name in candidate_names
    ],
    dtype=np.float64,
)

bright_neutral_fraction = bright_neutral_percentage / 100.0
bright_chromatic_fraction = bright_chromatic_percentage / 100.0

# 中性亮端达到50%，视为直接中性证据充分。
bright_neutral_evidence = np.clip(
    bright_neutral_fraction / 0.50,
    0.0,
    1.0,
)

# 彩色亮端的熵从0.50到0.80映射为0到1的多样性证据。
bright_diversity_evidence = (
    bright_chromatic_fraction
    * np.clip(
        (bright_color_diversity - 0.50) / 0.30,
        0.0,
        1.0,
    )
)

# 两类场景证据只要有一类充分，就可以使用亮端指标。
bright_evidence_confidence = max(
    bright_neutral_evidence,
    bright_diversity_evidence,
)

# 聚合分离达到25%时得分为0。
bright_aggregate_scores = np.clip(
    1.0 - bright_aggregate_spread_values / 25.0,
    0.0,
    1.0,
)

# 亮端潜在裁剪达到40%时得分为0。
bright_clipping_scores = np.clip(
    1.0 - bright_clipping_values / 40.0,
    0.0,
    1.0,
)

bright_candidate_scores = (
    0.75 * bright_aggregate_scores
    + 0.25 * bright_clipping_scores
)

# 亮端证据最多占V3候选总分的40%。
maximum_bright_score_weight = 0.40
effective_bright_score_weight = (
    maximum_bright_score_weight
    * bright_evidence_confidence
)

v3_reliability_scores = (
    (1.0 - effective_bright_score_weight)
    * reliability_scores
    + effective_bright_score_weight
    * bright_candidate_scores
)

print("\nHybrid V3 bright-tail score diagnostic:")
print(
    "Neutral-evidence confidence:",
    f"{bright_neutral_evidence:.4f}"
)
print(
    "Diversity-evidence confidence:",
    f"{bright_diversity_evidence:.4f}"
)
print(
    "Combined bright evidence confidence:",
    f"{bright_evidence_confidence:.4f}"
)
print(
    "Effective bright-score weight:",
    f"{effective_bright_score_weight:.4f}"
)

for candidate_index, name in enumerate(candidate_names):
    print(f"\n{name}:")
    print(
        "  Bright aggregate component:",
        f"{bright_aggregate_scores[candidate_index]:.4f}"
    )
    print(
        "  Bright clipping component:",
        f"{bright_clipping_scores[candidate_index]:.4f}"
    )
    print(
        "  Bright candidate score:",
        f"{bright_candidate_scores[candidate_index]:.4f}"
    )
    print(
        "  V3 reliability score:",
        f"{v3_reliability_scores[candidate_index]:.4f}"
    )

# ============================================================
# Hybrid V3: family-aware fusion
# ============================================================

gray_family_indices = np.array([0, 1])
pca_candidate_index = candidate_names.index("PCA")

# Gray World与Norm2先在同一家族内部竞争。
gray_family_member_scores = v3_reliability_scores[
    gray_family_indices
]
gray_family_shifted_scores = (
    gray_family_member_scores
    - np.max(gray_family_member_scores)
)
gray_family_softmax = np.exp(
    gray_family_shifted_scores
    / softmax_temperature
)
gray_family_member_weights = (
    gray_family_softmax
    / np.sum(gray_family_softmax)
)

# 增益是乘法比例，因此在log域融合。
gray_family_gains = np.exp(
    np.sum(
        gray_family_member_weights[:, np.newaxis]
        * np.log(candidate_gains[gray_family_indices]),
        axis=0,
    )
)

# 家族可靠性是两个成员分数的内部加权结果，
# 而不是把它们当成两张独立选票。
gray_family_reliability = np.sum(
    gray_family_member_weights
    * gray_family_member_scores
)

v3_family_names = [
    "Gray family",
    "PCA",
]

v3_family_gains = np.vstack(
    [
        gray_family_gains,
        candidate_gains[pca_candidate_index],
    ]
)

v3_family_scores = np.array(
    [
        gray_family_reliability,
        v3_reliability_scores[pca_candidate_index],
    ],
    dtype=np.float64,
)

v3_best_family_index = int(
    np.argmax(v3_family_scores)
)

v3_family_score_margin = float(
    np.max(v3_family_scores)
    - np.min(v3_family_scores)
)

# 场景分歧明显且家族赢家明确时才硬选择；
# 否则在两个家族之间软融合。
v3_use_hard_selection = (
    overall_disagreement
    >= hard_disagreement_threshold
    and v3_family_score_margin
    >= hard_margin_threshold
)

if v3_use_hard_selection:
    v3_decision_mode = "Hard family selection"

    v3_family_weights = np.zeros(
        len(v3_family_names),
        dtype=np.float64,
    )

    v3_family_weights[
        v3_best_family_index
    ] = 1.0
else:
    v3_decision_mode = "Soft family fusion"

    v3_shifted_family_scores = (
        v3_family_scores
        - np.max(v3_family_scores)
    )

    v3_family_softmax = np.exp(
        v3_shifted_family_scores
        / softmax_temperature
    )

    v3_family_weights = (
        v3_family_softmax
        / np.sum(v3_family_softmax)
    )

v3_hybrid_gains = np.exp(
    np.sum(
        v3_family_weights[:, np.newaxis]
        * np.log(v3_family_gains),
        axis=0,
    )
)

print("\nHybrid V3 family-aware fusion:")
print("Gray-family member weights:")

for member_index, weight in zip(
    gray_family_indices,
    gray_family_member_weights,
):
    print(
        f"  {candidate_names[member_index]}:",
        f"{weight:.4f}"
    )

print(
    "Gray-family gains [R, B]:",
    np.round(gray_family_gains, 6)
)
print(
    "Family score margin:",
    f"{v3_family_score_margin:.4f}"
)
print(
    "Decision mode:",
    v3_decision_mode
)
print("Top-level family scores and weights:")

for family_name, score, weight in zip(
    v3_family_names,
    v3_family_scores,
    v3_family_weights,
):
    print(
        f"  {family_name}:",
        f"score={score:.4f},",
        f"weight={weight:.4f}"
    )

print(
    "Hybrid V3 gains [R, B]:",
    np.round(v3_hybrid_gains, 6)
)
