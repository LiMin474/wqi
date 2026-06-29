from __future__ import annotations

from pathlib import Path


OUT = Path(r"D:\study\project\water-quality\Stacking-WQI-Hyperopt\results\method_framework_polished.svg")


def rect(x, y, w, h, rx=16, fill="#FFFFFF", stroke="#7A8AA0", sw=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'


def text(x, y, lines, size=24, weight="400", color="#243B53", anchor="middle"):
    if isinstance(lines, str):
        lines = [lines]
    spans = []
    start_y = y - (len(lines) - 1) * size * 0.7 / 2
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * 1.35
        spans.append(
            f'<tspan x="{x}" dy="{dy if i else 0}">{line}</tspan>'
        )
    return (
        f'<text x="{x}" y="{start_y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">\n'
        + "\n".join(spans)
        + "\n</text>\n"
    )


def arrow(x1, y1, x2, y2, path=None):
    if path is None:
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4B5563" stroke-width="2.2" marker-end="url(#arrow)"/>\n'
    return f'<path d="{path}" fill="none" stroke="#4B5563" stroke-width="2.2" marker-end="url(#arrow)"/>\n'


def main():
    w, h = 1800, 1020
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]

    # Group panels
    svg.append(rect(50, 80, 360, 370, rx=22, fill="#FCFCFD", stroke="#C7D2DE"))
    svg.append(rect(470, 80, 1280, 450, rx=22, fill="#FCFCFD", stroke="#C7D2DE"))
    svg.append(rect(320, 620, 1040, 280, rx=22, fill="#FCFCFD", stroke="#C7D2DE"))

    svg.append(text(100, 118, "Stage 1  Data Preparation", size=28, weight="700", anchor="start"))
    svg.append(text(520, 118, "Stage 2  Hyperparameter Optimization via Six Evolutionary Algorithms", size=28, weight="700", anchor="start"))
    svg.append(text(370, 658, "Stage 3  Ensemble Construction and Final Evaluation", size=28, weight="700", anchor="start"))

    # Stage 1
    svg.append(rect(90, 170, 280, 120, fill="#EEF4FF", stroke="#8AA0C8"))
    svg.append(text(230, 220, ["Three WQI Datasets", "Jajpur-Groundwater | Irish-River-CCME | AKH-WQI"], size=22, weight="700"))
    svg.append(rect(90, 340, 280, 90, fill="#F5F7FA", stroke="#8AA0C8"))
    svg.append(text(230, 382, ["Preprocessing and Split", "8:2 hold-out + 5-fold CV"], size=21))
    svg.append(arrow(230, 290, 230, 340))
    svg.append(arrow(370, 300, 470, 300))

    # Stage 2
    svg.append(rect(860, 145, 420, 95, fill="#EAF6F2", stroke="#4E8B7A"))
    svg.append(text(1070, 190, ["ANN Hyperparameter Search", "via Six Evolutionary Optimizers"], size=24, weight="700"))

    algo_names = ["DE", "SHADE", "CMA-ES", "NRBO", "BOA", "HHO-Lite"]
    algo_x = [640, 810, 980, 1150, 1320, 1490]
    for x, name in zip(algo_x, algo_names):
        svg.append(rect(x - 50, 330, 100, 62, fill="#F3F0FF", stroke="#9A89D0"))
        svg.append(text(x, 366, name, size=22, weight="700"))

    svg.append(rect(860, 455, 420, 78, fill="#F5F7FA", stroke="#5B7C99"))
    svg.append(text(1070, 494, "Six optimized ANN base models", size=24, weight="700"))

    for x in algo_x:
        svg.append(arrow(1070, 240, x, 330, path=f"M 1070 240 Q {x} 255 {x} 330"))
        svg.append(arrow(x, 392, 1070, 455, path=f"M {x} 392 Q {x} 445 1070 455"))

    # Stage 3
    svg.append(rect(430, 720, 350, 110, fill="#FFF8E8", stroke="#C79B2C"))
    svg.append(text(605, 772, ["WeightedAvg Ensemble", "Weights derived from normalized R²CV"], size=24, weight="700"))
    svg.append(rect(915, 720, 350, 110, fill="#EEF4FF", stroke="#5B7C99"))
    svg.append(text(1090, 772, ["Final Ensemble Prediction", "Metrics: R², R²CV, RMSE, MAE, Time"], size=22, weight="700"))

    svg.append(arrow(1070, 533, 605, 720))
    svg.append(arrow(780, 775, 915, 775))

    svg.append(text(900, 965, "Framework of the proposed multi-evolutionary-algorithm ANN optimization and R²CV-weighted ensemble strategy", size=20, color="#52606D"))
    svg.append("</svg>")
    OUT.write_text("".join(svg), encoding="utf-8")
    print(str(OUT))


if __name__ == "__main__":
    main()
