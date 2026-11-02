import argparse
import colorsys
import os

import numpy as np
from PIL import Image, ImageOps

from continuous_rounded_corner import round_corner

CANVAS_SIZE = (4000, 2250)
DEFAULT_FOLDER = 'album_cover_music'
DEFAULT_OUTPUT = 'hue-brightness.jpg'
BINS = 10
BASE_SCALE = 0.078
MAX_SCALE = 0.22
COVER_GAP = 10
MARGIN = 20
LAYOUT_ITERATIONS = 600
DEDUPE_DISTANCE = 2


def get_images_from_folder(folder_path):
    filenames = sorted(os.listdir(folder_path))
    return [
        os.path.join(folder_path, filename)
        for filename in filenames
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def get_brightness_and_hue(image_path):
    pixels = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float64)

    brightness = pixels[..., :3].mean() / 255.0

    average_rgb = pixels[..., :3].mean(axis=(0, 1)) / 255.0
    hue = colorsys.rgb_to_hsv(*average_rgb)[0]

    return brightness, hue


def get_brightness_and_hue_for_images(images):
    return [get_brightness_and_hue(path) for path in images]


def difference_hash(image_path, size=16):
    gray = ImageOps.grayscale(Image.open(image_path))
    gray = gray.resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def dedupe_images(images, max_distance=DEDUPE_DISTANCE):
    """去掉封面完全相同的重复专辑（同一张图的不同版本/语言标签）。"""
    kept, hashes = [], []
    for path in images:
        digest = difference_hash(path)
        if any(np.count_nonzero(digest != other) <= max_distance
               for other in hashes):
            continue
        kept.append(path)
        hashes.append(digest)
    return kept


def normalize(values, canvas_dim):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    if span == 0:
        return np.full_like(values, (canvas_dim - 1) / 2)
    return (values - values.min()) / span * (canvas_dim - 1)


def get_cover_sizes(brightness, hue):
    """按邻域密度确定封面尺寸：越密集的封面越小。"""
    hist, brightness_edges, hue_edges = np.histogram2d(
        brightness, hue, bins=BINS
    )
    density = (hist.T + 1) ** 0.7
    density /= density.max()

    brightness_bin = np.clip(np.digitize(brightness, brightness_edges) - 1, 0, BINS - 1)
    hue_bin = np.clip(np.digitize(hue, hue_edges) - 1, 0, BINS - 1)
    local_density = density[hue_bin, brightness_bin]

    scale = np.clip(BASE_SCALE / local_density, BASE_SCALE, MAX_SCALE)
    return scale * 632


def layout_covers(anchors, sizes, canvas_size, gap=COVER_GAP,
                  iterations=LAYOUT_ITERATIONS):
    """将封面围绕各自的亮度-色相坐标铺开，消除封面之间的相互遮挡。"""
    width, height = canvas_size
    half = sizes / 2
    pos = anchors.astype(np.float64).copy()
    n = len(sizes)

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

        # 弱弹簧把封面拉回原本的坐标，保持亮度-色相分布
        pos[:, 0] += 0.002 * (anchors[:, 0] - pos[:, 0])
        pos[:, 1] += 0.002 * (anchors[:, 1] - pos[:, 1])

        pos[:, 0] = np.clip(pos[:, 0], half + MARGIN, width - half - MARGIN)
        pos[:, 1] = np.clip(pos[:, 1], half + MARGIN, height - half - MARGIN)

    return pos


def create_hue_brightness_mosaic(images, output_path,
                                 canvas_size=CANVAS_SIZE):
    before = len(images)
    images = dedupe_images(images)
    print(f'封面 {before} 张，去重后 {len(images)} 张')

    properties = np.asarray(get_brightness_and_hue_for_images(images))
    brightness, hue = properties[:, 0], properties[:, 1]
    sizes = get_cover_sizes(brightness, hue)

    anchors = np.column_stack([
        normalize(brightness, canvas_size[0]),
        normalize(hue, canvas_size[1]),
    ])
    positions = layout_covers(anchors, sizes, canvas_size)

    canvas = Image.new('RGB', canvas_size, 'black')

    # 大图先贴，小图后贴，少量残余重叠时小图在上层
    for index in np.argsort(-sizes):
        size = int(sizes[index])
        cover = Image.open(images[index]).convert('RGBA')
        cover = ImageOps.fit(cover, (size, size), Image.LANCZOS)
        cover = round_corner(cover)

        x = int(positions[index, 0] - size / 2)
        y = int(positions[index, 1] - size / 2)
        canvas.paste(cover, (x, y), cover)

    canvas.save(output_path, 'JPEG', quality=95, subsampling=0)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description='按亮度-色相排列专辑封面')
    parser.add_argument('--folder', default=DEFAULT_FOLDER,
                        help=f'封面目录（默认 {DEFAULT_FOLDER}）')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'输出文件（默认 {DEFAULT_OUTPUT}）')
    return parser.parse_args()


def main():
    args = parse_args()

    images = get_images_from_folder(args.folder)
    create_hue_brightness_mosaic(images, args.output)
    print(f'输出图像已保存至 {args.output}')


if __name__ == '__main__':
    main()
