import numpy as np
import matplotlib.pyplot as plt

height = 16
width = 16

pattern = np.empty((height, width), dtype="U1")

pattern[0::2, 0::2] = "R"
pattern[0::2, 1::2] = "G"
pattern[1::2, 0::2] = "G"
pattern[1::2, 1::2] = "B"

print(pattern)
# 创建一张 RGB 图片，仅用于“可视化”滤光片颜色
display = np.zeros((height, width, 3), dtype=np.uint8)

display[pattern == "R"] = [255, 0, 0]
display[pattern == "G"] = [0, 180, 0]
display[pattern == "B"] = [0, 0, 255]

plt.figure(figsize=(6, 6))
plt.imshow(display)

# 给每个格子标上 R / G / B
for row in range(height):
    for col in range(width):
        plt.text(
            col,
            row,
            pattern[row, col],
            color="white",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

plt.title("RGGB Bayer Color Filter Array")
plt.xticks(range(width))
plt.yticks(range(height))
plt.tight_layout()

plt.savefig("results/02_bayer_pattern.png", dpi=200)
plt.close()

print("Saved to results/02_bayer_pattern.png")
# -------------------------
# Simulate RGB -> Bayer RAW
# -------------------------


height = 16
width = 16

scene_rgb = np.zeros(
    (height, width, 3),
    dtype=np.float32
)

# 左半边：偏红橙色
scene_rgb[:, :width // 2] = [200, 60, 40]

# 右半边：偏蓝色
scene_rgb[:, width // 2:] = [30, 90, 220]

print("RGB scene shape:", scene_rgb.shape)
print("One RGB pixel:", scene_rgb[0, 0])

raw = np.zeros((height, width), dtype=np.uint8)

raw[pattern == "R"] = scene_rgb[:, :, 0][pattern == "R"]
raw[pattern == "G"] = scene_rgb[:, :, 1][pattern == "G"]
raw[pattern == "B"] = scene_rgb[:, :, 2][pattern == "B"]

print("RAW shape:", raw.shape)
print("RAW data:")
print(raw)

raw_r = np.zeros((height, width), dtype=np.uint8)
raw_r[pattern == "R" ] = raw[pattern == "R"]
print(raw_r)
raw_g = np.zeros((height, width), dtype=np.uint8)
raw_g[pattern == "G" ] = raw[pattern == "G"]
print(raw_g)
raw_b = np.zeros((height, width), dtype=np.uint8)
raw_b[pattern == "B" ] = raw[pattern == "B"]
print(raw_b)

sparse_rgb = np.zeros((height, width, 3), dtype=np.uint8)
sparse_rgb[:, :, 0] = raw_r
sparse_rgb[:, :, 1] = raw_g
sparse_rgb[:, :, 2] = raw_b
print(sparse_rgb[0, 0])
print(sparse_rgb[0, 1])
print(sparse_rgb[1, 1])

g_estimate = np.mean([
    sparse_rgb[1, 2, 1],
    sparse_rgb[2, 1, 1],
    sparse_rgb[2, 3, 1],
    sparse_rgb[3, 2, 1]
]
)
print("Estimated G:", g_estimate)

b_estimate = np.mean([
    sparse_rgb[1, 1, 2],
    sparse_rgb[1, 3, 2],
    sparse_rgb[3, 1, 2],
    sparse_rgb[3, 3, 2]
]
)
print("Estimated B:", b_estimate)

demosaic_rgb = sparse_rgb.astype(np.float32)
demosaic_rgb[2, 2, 1] = g_estimate
demosaic_rgb[2, 2, 2] = b_estimate

print("Before:", sparse_rgb[2, 2])
print("After:", demosaic_rgb[2, 2])

g_at_b = np.mean([
    sparse_rgb[0, 1, 1],
    sparse_rgb[1, 0, 1],
    sparse_rgb[1, 2, 1],
    sparse_rgb[2, 1, 1]
])

r_at_b = np.mean([
    sparse_rgb[0, 0, 0],
    sparse_rgb[0, 2, 0],
    sparse_rgb[2, 0, 0],
    sparse_rgb[2, 2, 0]
])

demosaic_rgb[1, 1, 1] = g_at_b
demosaic_rgb[1, 1, 0] = r_at_b
print("Estimated G at B:", g_at_b)
print("Estimated R at B:", r_at_b)

r_at_g = np.mean([
    sparse_rgb[2, 0, 0],
    sparse_rgb[2, 2, 0]
])

b_at_g = np.mean([
    sparse_rgb[1, 1, 2],
    sparse_rgb[3, 1, 2]
])

demosaic_rgb[2, 1, 0] = r_at_g
demosaic_rgb[2, 1, 2] = b_at_g

print("Before at G:", sparse_rgb[2, 1])
print("After at G:", demosaic_rgb[2, 1])

source_rgb = sparse_rgb.astype(np.float32)

padded_rgb = np.pad(
    source_rgb,
    ((1, 1), (1, 1), (0, 0)),
    mode="reflect"
)

print("Before padding:", source_rgb.shape)
print("After padding:", padded_rgb.shape)

demosaic_full = source_rgb.copy()

for y in range(height):
    for x in range(width):
        py = y + 1
        px = x + 1

        if pattern[y, x] == "R":
            # R位置：G取上下左右，B取四个对角线
            demosaic_full[y, x, 1] = np.mean([
                padded_rgb[py - 1, px, 1],
                padded_rgb[py + 1, px, 1],
                padded_rgb[py, px - 1, 1],
                padded_rgb[py, px + 1, 1]
            ])

            demosaic_full[y, x, 2] = np.mean([
                padded_rgb[py - 1, px - 1, 2],
                padded_rgb[py - 1, px + 1, 2],
                padded_rgb[py + 1, px - 1, 2],
                padded_rgb[py + 1, px + 1, 2]
            ])

        elif pattern[y, x] == "B":
            # B位置：G取上下左右，R取四个对角线
            demosaic_full[y, x, 1] = np.mean([
                padded_rgb[py - 1, px, 1],
                padded_rgb[py + 1, px, 1],
                padded_rgb[py, px - 1, 1],
                padded_rgb[py, px + 1, 1]
            ])

            demosaic_full[y, x, 0] = np.mean([
                padded_rgb[py - 1, px - 1, 0],
                padded_rgb[py - 1, px + 1, 0],
                padded_rgb[py + 1, px - 1, 0],
                padded_rgb[py + 1, px + 1, 0]
            ])

        else:
            # G有两种排列方向
            if y % 2 == 0:
                # RG这一行：R在左右，B在上下
                demosaic_full[y, x, 0] = np.mean([
                    padded_rgb[py, px - 1, 0],
                    padded_rgb[py, px + 1, 0]
                ])

                demosaic_full[y, x, 2] = np.mean([
                    padded_rgb[py - 1, px, 2],
                    padded_rgb[py + 1, px, 2]
                ])

            else:
                # GB这一行：R在上下，B在左右
                demosaic_full[y, x, 0] = np.mean([
                    padded_rgb[py - 1, px, 0],
                    padded_rgb[py + 1, px, 0]
                ])

                demosaic_full[y, x, 2] = np.mean([
                    padded_rgb[py, px - 1, 2],
                    padded_rgb[py, px + 1, 2]
                ])

print("R position:", demosaic_full[0, 0])
print("G position:", demosaic_full[0, 1])
print("B position:", demosaic_full[1, 1])
print("All correct:", np.allclose(demosaic_full, scene_rgb))


# 1. 把计算结果转换为可保存的8-bit图像
demosaic_uint8 = np.clip(
    demosaic_full,
    0,
    255
).astype(np.uint8)

# 2. 单独保存Demosaicing结果
plt.imsave(
    "results/03_demosaic_bilinear.png",
    demosaic_uint8
)

# 3. 制作前后对比图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

original_vis = np.clip(scene_rgb, 0, 255).astype(np.uint8)
sparse_vis = np.clip(sparse_rgb, 0, 255).astype(np.uint8)

axes[0].imshow(original_vis)
axes[0].set_title("Original RGB")

axes[1].imshow(sparse_vis)
axes[1].set_title("Bayer Sparse RGB")

axes[2].imshow(demosaic_uint8)
axes[2].set_title("Demosaiced RGB")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "results/06_edge_demosaic_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

# 计算每个数值的绝对误差
absolute_error = np.abs(demosaic_full - scene_rgb)

# 对所有像素、所有通道的绝对误差取平均
mae = np.mean(absolute_error)

mae_rgb = np.mean(absolute_error, axis=(0, 1))

print("Overall MAE:", mae)
print("R MAE:", mae_rgb[0])
print("G MAE:", mae_rgb[1])
print("B MAE:", mae_rgb[2])

# 原图与恢复图的差值
error = demosaic_full - scene_rgb

# 将每个误差平方，然后求整体平均
mse = np.mean(error ** 2)

# 分别计算R、G、B通道的MSE
mse_rgb = np.mean(error ** 2, axis=(0, 1))

print("Overall MSE:", mse)
print("R MSE:", mse_rgb[0])
print("G MSE:", mse_rgb[1])
print("B MSE:", mse_rgb[2])

# MSE开平方，得到RMSE
rmse = np.sqrt(mse)

# 根据8-bit图像的最大值255计算PSNR
if mse == 0:
    psnr = float("inf")
else:
    psnr = 10 * np.log10((255.0 ** 2) / mse)

print("RMSE:", rmse)
print("PSNR:", psnr, "dB")

# 所有RGB通道中最大的绝对误差
max_error = np.max(absolute_error)

# 创建三个子图，分别显示R、G、B通道误差
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

channel_names = ["R Error", "G Error", "B Error"]

for c in range(3):
    image = axes[c].imshow(
        absolute_error[:, :, c],
        cmap="magma",
        vmin=0,
        vmax=max_error,
        interpolation="nearest"
    )

    axes[c].set_title(channel_names[c])
    axes[c].set_xlabel("x")
    axes[c].set_ylabel("y")

# 给三张图添加统一的颜色刻度
fig.colorbar(
    image,
    ax=axes,
    label="Absolute Error",
    shrink=0.8
)

plt.savefig(
    "results/07_edge_error_map.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("results/07_edge_error_map.png")
print("Saved: results/03_demosaic_bilinear.png")
print("Saved: results/04_demosaic_comparison.png")

print("第一列原始R：")
print(scene_rgb[:, 0, 0])

print("第一列恢复R：")
print(demosaic_full[:, 0, 0])

print("最后一列原始R：")
print(scene_rgb[:, -1, 0])

print("最后一列恢复R：")
print(demosaic_full[:, -1, 0])

# 去掉最外面一圈，只评价内部区域
original_inner = scene_rgb[1:-1, 1:-1]
demosaic_inner = demosaic_full[1:-1, 1:-1]

inner_error = demosaic_inner - original_inner
inner_mse = np.mean(inner_error ** 2)

if inner_mse == 0:
    inner_psnr = float("inf")
else:
    inner_psnr = 10 * np.log10((255.0 ** 2) / inner_mse)

print("Full-image PSNR:", psnr, "dB")
print("Inner-region MSE:", inner_mse)
print("Inner-region PSNR:", inner_psnr, "dB")

print("\n边缘附近像素对比：")

for y in [8, 9]:
    print(f"\ny = {y}")

    for x in [6, 7, 8, 9]:
        print(
            f"x={x}:",
            "original =", scene_rgb[y, x],
            "demosaiced =", demosaic_full[y, x],
            "error =", absolute_error[y, x]
        )