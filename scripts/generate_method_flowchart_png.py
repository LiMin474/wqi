from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"D:\study\project\water-quality\Stacking-WQI-Hyperopt\results\method_framework_polished.png")
W, H = 2400, 1350


def get_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def rounded_box(draw, xy, radius=22, fill="#FFFFFF", outline="#7A8AA0", width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def fit_multiline_text(draw, lines, box_w, box_h, start_size, min_size=18, bold=False, line_spacing=10):
    if isinstance(lines, str):
        lines = [lines]
    size = start_size
    while size >= min_size:
        font = get_font(size, bold)
        bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        widths = [b[2] - b[0] for b in bboxes]
        heights = [b[3] - b[1] for b in bboxes]
        total_h = sum(heights) + line_spacing * (len(lines) - 1)
        if max(widths) <= box_w and total_h <= box_h:
            return font, heights, widths
        size -= 1
    font = get_font(min_size, bold)
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    return font, heights, widths


def center_multiline(draw, box, lines, start_size=30, min_size=18, bold=False, fill="#243B53", line_spacing=10):
    x1, y1, x2, y2 = box
    pad_x = 18
    pad_y = 16
    font, heights, widths = fit_multiline_text(
        draw,
        lines,
        box_w=(x2 - x1 - 2 * pad_x),
        box_h=(y2 - y1 - 2 * pad_y),
        start_size=start_size,
        min_size=min_size,
        bold=bold,
        line_spacing=line_spacing,
    )
    total_h = sum(heights) + line_spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + line_spacing


def left_title(draw, x, y, title):
    draw.text((x, y), title, font=get_font(34, True), fill="#243B53")


def arrow(draw, start, end, color="#4B5563", width=4, head=16):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        draw.polygon(
            [(x2, y2), (x2 - sign * head, y2 - head / 2), (x2 - sign * head, y2 + head / 2)],
            fill=color,
        )
    else:
        sign = 1 if y2 >= y1 else -1
        draw.polygon(
            [(x2, y2), (x2 - head / 2, y2 - sign * head), (x2 + head / 2, y2 - sign * head)],
            fill=color,
        )


def bezier_points(p0, p1, p2, p3, steps=50):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (
            (1 - t) ** 3 * p0[0]
            + 3 * (1 - t) ** 2 * t * p1[0]
            + 3 * (1 - t) * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            (1 - t) ** 3 * p0[1]
            + 3 * (1 - t) ** 2 * t * p1[1]
            + 3 * (1 - t) * t**2 * p2[1]
            + t**3 * p3[1]
        )
        pts.append((x, y))
    return pts


def curved_arrow(draw, p0, p1, p2, p3, color="#4B5563", width=4):
    pts = bezier_points(p0, p1, p2, p3)
    draw.line(pts, fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, width=width)


def main():
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    rounded_box(draw, (70, 85, 560, 560), radius=30, fill="#FCFCFD", outline="#C7D2DE", width=3)
    rounded_box(draw, (610, 85, 2330, 660), radius=30, fill="#FCFCFD", outline="#C7D2DE", width=3)
    rounded_box(draw, (390, 760, 1880, 1190), radius=30, fill="#FCFCFD", outline="#C7D2DE", width=3)

    left_title(draw, 105, 112, "Stage 1  Data Preparation")
    left_title(draw, 645, 112, "Stage 2  Hyperparameter Optimization via Six Evolutionary Algorithms")
    left_title(draw, 425, 788, "Stage 3  Ensemble Construction and Final Evaluation")

    rounded_box(draw, (125, 190, 505, 345), radius=24, fill="#EEF4FF", outline="#8AA0C8")
    center_multiline(
        draw,
        (125, 190, 505, 345),
        ["Three WQI datasets", "Jajpur-Groundwater", "Irish-River-CCME", "AKH-WQI"],
        start_size=28,
        min_size=20,
        bold=True,
        line_spacing=8,
    )

    rounded_box(draw, (125, 410, 505, 520), radius=24, fill="#F5F7FA", outline="#8AA0C8")
    center_multiline(
        draw,
        (125, 410, 505, 520),
        ["Preprocessing and split", "80/20 hold-out + 5-fold CV"],
        start_size=28,
        min_size=20,
        bold=False,
        line_spacing=8,
    )
    arrow(draw, (315, 345), (315, 410))
    arrow(draw, (505, 345), (610, 345))

    rounded_box(draw, (1160, 180, 1620, 315), radius=24, fill="#EAF6F2", outline="#4E8B7A")
    center_multiline(
        draw,
        (1160, 180, 1620, 315),
        ["ANN hyperparameter search", "via six evolutionary", "optimizers"],
        start_size=30,
        min_size=22,
        bold=True,
        line_spacing=8,
    )

    algo_names = ["DE", "SHADE", "CMA-ES", "NRBO", "BOA", "HHO-Lite"]
    algo_x = [820, 1050, 1280, 1510, 1740, 1970]
    for x, name in zip(algo_x, algo_names):
        rounded_box(draw, (x - 80, 430, x + 80, 510), radius=18, fill="#F3F0FF", outline="#9A89D0")
        center_multiline(draw, (x - 80, 430, x + 80, 510), [name], start_size=27, min_size=22, bold=True)

    rounded_box(draw, (1125, 560, 1655, 660), radius=24, fill="#F5F7FA", outline="#5B7C99")
    center_multiline(
        draw,
        (1125, 560, 1655, 660),
        ["Six optimized ANN", "base models"],
        start_size=30,
        min_size=22,
        bold=True,
        line_spacing=8,
    )

    for x in algo_x:
        curved_arrow(draw, (1390, 315), (1390, 370), (x, 360), (x, 430))
        curved_arrow(draw, (x, 510), (x, 555), (1390, 540), (1390, 560))

    rounded_box(draw, (520, 875, 995, 1025), radius=24, fill="#FFF8E8", outline="#C79B2C")
    center_multiline(
        draw,
        (520, 875, 995, 1025),
        ["WeightedAvg ensemble", "weights derived from", "normalized R2CV"],
        start_size=29,
        min_size=21,
        bold=True,
        line_spacing=8,
    )

    rounded_box(draw, (1210, 875, 1710, 1025), radius=24, fill="#EEF4FF", outline="#5B7C99")
    center_multiline(
        draw,
        (1210, 875, 1710, 1025),
        ["Final ensemble prediction", "Metrics: R2, R2CV,", "RMSE, MAE, Time"],
        start_size=28,
        min_size=20,
        bold=True,
        line_spacing=8,
    )

    arrow(draw, (1390, 660), (760, 875))
    arrow(draw, (995, 950), (1210, 950))

    draw.text(
        (W / 2 - 620, 1245),
        "Framework of the proposed multi-evolutionary-algorithm ANN optimization and R2CV-weighted ensemble strategy",
        font=get_font(24),
        fill="#52606D",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
