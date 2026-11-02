"""按亮度-色相把专辑封面排成壁纸。

四种构图：
  wall   亮度分带、带内按色相排序的满版拼贴（推荐）
  field  亮度分行的抖动网格，松弛后略带随机感
  random 无圆角、无间距、随机排序的满版方格
  plot   保留原始亮度-色相坐标，只放大封面并加背景氛围
"""

import argparse
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from album_rainbow import dedupe_images, get_images_from_folder, layout_covers
from continuous_rounded_corner import round_corner

CANVAS_SIZE = (4000, 2250)
DEFAULT_FOLDER = 'album_cover_music'
DEFAULT_OUTPUT = 'hue-brightness.jpg'
MARGIN = 60


def get_cover_stats(image_path):
    pixels = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float64)
    average_rgb = pixels.mean(axis=(0, 1)) / 255.0
    brightness = float(pixels[..., :3].mean() / 255.0)
    hue = colorsys.rgb_to_hsv(*average_rgb)[0]
    return brightness, hue, average_rgb


def get_stats_for_images(images):
    stats = [get_cover_stats(path) for path in images]
    return (
        np.array([s[0] for s in stats]),
        np.array([s[1] for s in stats]),
        np.array([s[2] for s in stats]),
    )


def rank_normalize(values):
    order = np.argsort(np.argsort(values))
    return (order + 0.5) / len(values)


def linear_normalize(values, span):
    values = np.asarray(values, dtype=np.float64)
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return np.full_like(values, span / 2)
    return (values - minimum) / (maximum - minimum) * span


def build_field_layout(brightness, hue, canvas_size, sizes, rng,
                       gap=18, iterations=1200):
    """亮度分行、色相排开，再用排斥力消除重叠；整体上暗下亮。"""
    width, height = canvas_size
    count = len(sizes)
    rows = max(int(round(np.sqrt(count * height / width))), 1)
    cols = int(np.ceil(count / rows))
    cell_w, cell_h = width / cols, height / rows

    row_of = np.minimum((rank_normalize(brightness) * rows).astype(int), rows - 1)
    anchors = np.zeros((count, 2))
    for row in range(rows):
        members = np.where(row_of == row)[0]
        members = members[np.argsort(hue[members])]
        slots = (np.arange(len(members)) + 0.5) * cell_w
        anchors[members, 0] = slots
        anchors[members, 1] = (row + 0.5) * cell_h
    anchors += rng.normal(0, 1, anchors.shape) * np.array(
        [cell_w * 0.26, cell_h * 0.24]
    )

    half = sizes / 2
    pos = anchors.copy()
    for _ in range(iterations):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        overlap_x = (half[:, None] + half[None, :] + gap) - np.abs(dx)
        overlap_y = (half[:, None] + half[None, :] + gap) - np.abs(dy)
        colliding = (overlap_x > 0) & (overlap_y > 0)
        np.fill_diagonal(colliding, False)

        push_x = np.sign(dx) * np.where(overlap_x < overlap_y, overlap_x, 0)
        push_y = np.sign(dy) * np.where(overlap_x >= overlap_y, overlap_y, 0)
        pos[:, 0] += np.where(colliding, push_x, 0).sum(axis=1) * 0.5
        pos[:, 1] += np.where(colliding, push_y, 0).sum(axis=1) * 0.5

        pos += 0.0015 * (anchors - pos)
        pos[:, 0] = np.clip(pos[:, 0], half + MARGIN, width - half - MARGIN)
        pos[:, 1] = np.clip(pos[:, 1], half + MARGIN, height - half - MARGIN)
    return pos


def build_wall_layout(brightness, hue, canvas_size, rng, gap=18):
    """亮度分行、色相排开的满版拼贴：上暗下亮，行内位置带轻微抖动。"""
    width, height = canvas_size
    count = len(brightness)
    rows = int(round(np.sqrt(count * height / width)))
    rows = max(rows, 1)

    row_of = np.minimum((rank_normalize(brightness) * rows).astype(int), rows - 1)
    cell_h = height / rows

    positions = np.zeros((count, 2))
    positions[:, 1] = (row_of + 0.5) * cell_h

    min_spacing = width
    for row in range(rows):
        members = np.where(row_of == row)[0]
        members = members[np.argsort(hue[members])]
        slots = (np.arange(len(members)) + 0.5) * width / len(members)
        positions[members, 0] = slots
        positions[members, 1] += rng.normal(0, cell_h * 0.04, len(members))
        min_spacing = min(min_spacing, width / len(members))

    size = min(min_spacing, cell_h) - gap
    positions += rng.normal(0, 3.0, positions.shape)
    positions[:, 0] = np.clip(positions[:, 0], size / 2, width - size / 2)
    positions[:, 1] = np.clip(positions[:, 1], size / 2, height - size / 2)
    return positions, size


def build_plot_layout(brightness, hue, canvas_size, sizes, rng):
    width, height = canvas_size
    anchors = np.column_stack([
        linear_normalize(hue, width),
        linear_normalize(brightness, height),
    ])
    return layout_covers(anchors, sizes, canvas_size, gap=16, iterations=900)


def make_backdrop(colors, positions, sizes, canvas_size, gap=18, scale=8):
    """用每张封面的主色在缩小画布上铺一层柔光，作为封面底下的氛围背景。"""
    width, height = canvas_size
    tiny_size = (max(width // scale, 1), max(height // scale, 1))
    tiny = Image.new('RGB', tiny_size, (7, 8, 12))
    draw = ImageDraw.Draw(tiny)

    for color, (x, y), size in zip(colors, positions, sizes):
        hue, saturation, value = colorsys.rgb_to_hsv(*color)
        saturation = min(saturation * 1.9, 1.0)
        value = min(value * 1.0 + 0.08, 1.0)
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value))
        radius = (size + gap * 3) / 2 / scale
        draw.ellipse([
            x / scale - radius, y / scale - radius,
            x / scale + radius, y / scale + radius,
        ], fill=rgb)

    tiny = tiny.filter(ImageFilter.GaussianBlur(scale * 0.9))
    backdrop = tiny.resize(canvas_size, Image.LANCZOS)
    backdrop = backdrop.filter(ImageFilter.GaussianBlur(22))

    # 上暗下亮：背景光在顶部压暗、到底部逐渐提亮
    ys = np.linspace(0, 1, canvas_size[1])[:, None, None]
    glow = np.asarray(backdrop, dtype=np.float64) / 255
    glow *= 0.22 + 0.80 * ys ** 1.2

    base = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float64)
    base[:] = np.array([6, 7, 11]) * (1 - ys) + np.array([34, 37, 46]) * ys ** 1.4
    base /= 255
    blended = base + glow * (1 - base)
    backdrop = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    return backdrop


def add_grain(image, amount=3.5, seed=0):
    rng = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float64)
    noise = rng.normal(0, amount, array.shape[:2])[..., None]
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def paste_cover(canvas, path, position, size, angle):
    cover = Image.open(path).convert('RGB')
    cover = ImageOps.fit(cover, (size, size), Image.LANCZOS).convert('RGBA')
    cover = round_corner(cover)

    shadow_pad = max(size // 4, 18)
    shadow_alpha = cover.getchannel('A').filter(
        ImageFilter.GaussianBlur(max(size / 28, 4))
    )
    shadow = Image.new('RGBA', (size + 2 * shadow_pad, size + 2 * shadow_pad),
                       (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 150), (shadow_pad, shadow_pad), shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(size / 60, 2)))

    if angle:
        cover = cover.rotate(angle, resample=Image.BICUBIC, expand=True)
        shadow = shadow.rotate(angle, resample=Image.BICUBIC, expand=True)

    x = int(position[0] - cover.size[0] / 2)
    y = int(position[1] - cover.size[1] / 2)

    canvas.alpha_composite(shadow, (int(position[0] - shadow.size[0] / 2),
                                    int(position[1] - shadow.size[1] / 2 + size * 0.02)))
    canvas.alpha_composite(cover, (x, y))


def render(images, colors, positions, sizes, angles, canvas_size,
           output_path, gap=18, seed=0):
    backdrop = make_backdrop(colors, positions, sizes, canvas_size, gap)
    canvas = backdrop.convert('RGBA')

    for index in np.argsort(-sizes):
        paste_cover(canvas, images[index], positions[index],
                    int(sizes[index]), angles[index])

    canvas = canvas.convert('RGB')
    canvas = add_grain(canvas, seed=seed)
    canvas.save(output_path, 'JPEG', quality=95, subsampling=0)
    return canvas


def cover_lightness(image_path):
    """封面的感知亮度，用来判断它算浅色还是深色。"""
    pixels = np.asarray(Image.open(image_path).convert('L'), dtype=np.float64)
    return float(pixels.mean() / 255)


def invert_brightness(image):
    """完全反转明度（V → 1−V），色相和饱和度保持不变。"""
    lookup = [255 - value for value in range(256)]
    hue, saturation, value = image.convert('HSV').split()
    return Image.merge(
        'HSV', (hue, saturation, value.point(lookup))
    ).convert('RGB')


def create_random_tiles(images, canvas_size, output_path, rng, seed=0,
                        dark=False, threshold=0.5):
    """无圆角、无间距、随机排序的满版方格；行高统一，除不尽的放最后一行。"""
    width, height = canvas_size
    count = len(images)
    rows = max(int(round(np.sqrt(count * height / width))), 1)
    base, extra = divmod(count, rows)
    counts = [base] * rows
    counts[-1] += extra

    light_covers = (
        np.array([cover_lightness(path) > threshold for path in images])
        if dark else np.zeros(count, dtype=bool)
    )
    order = rng.permutation(count)
    canvas = Image.new('RGB', canvas_size)
    cell_h = height / rows
    cursor = 0
    for row, n in enumerate(counts):
        y0 = int(round(row * cell_h))
        y1 = height if row == rows - 1 else int(round((row + 1) * cell_h))
        for col in range(n):
            x0 = int(round(col * width / n))
            x1 = int(round((col + 1) * width / n))
            index = order[cursor]
            cover = Image.open(images[index]).convert('RGB')
            cover = ImageOps.fit(cover, (x1 - x0, y1 - y0), Image.LANCZOS)
            if light_covers[index]:
                cover = invert_brightness(cover)
            canvas.paste(cover, (x0, y0))
            cursor += 1

    canvas = add_grain(canvas, amount=2.0, seed=seed)
    canvas.save(output_path, 'JPEG', quality=95, subsampling=0)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description='把专辑封面排成壁纸')
    parser.add_argument('--folder', default=DEFAULT_FOLDER)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--style', default='wall',
                        choices=('field', 'wall', 'random', 'plot'))
    parser.add_argument('--size', type=float, default=140,
                        help='field/plot 的平均封面边长（像素）')
    parser.add_argument('--gap', type=float, default=20,
                        help='封面间距（像素，可为负值让封面轻微重叠）')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='封面最大旋转角度（度）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dark', action='store_true',
                        help='random 布局的深色模式：只反转浅色封面的明度')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    images = get_images_from_folder(args.folder)
    images = dedupe_images(images)
    print(f'去重后 {len(images)} 张封面')

    canvas_size = CANVAS_SIZE
    if args.style == 'random':
        create_random_tiles(images, canvas_size, args.output, rng,
                            seed=args.seed, dark=args.dark)
        print(f'输出图像已保存至 {args.output}')
        return

    brightness, hue, colors = get_stats_for_images(images)
    count = len(images)

    if args.style == 'wall':
        positions, base_size = build_wall_layout(
            brightness, hue, canvas_size, rng, gap=args.gap
        )
        sizes = base_size * rng.uniform(0.86, 1.06, count)
    else:
        sizes = args.size * rng.uniform(0.78, 1.22, count)
        if args.style == 'field':
            positions = build_field_layout(
                brightness, hue, canvas_size, sizes, rng, gap=args.gap
            )
        else:
            positions = build_plot_layout(
                brightness, hue, canvas_size, sizes, rng
            )

    angles = rng.uniform(-args.rotation, args.rotation, count)
    render(images, colors, positions, sizes, angles, canvas_size,
           args.output, gap=args.gap, seed=args.seed)
    print(f'输出图像已保存至 {args.output}')


if __name__ == '__main__':
    main()
