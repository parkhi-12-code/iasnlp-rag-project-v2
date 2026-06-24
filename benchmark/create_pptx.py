"""
Generate IASNLP_Presentation.pptx — 16-slide deck for IIIT-Hyderabad IASNLP 2026.
Design: navy #1a2744 background, white text, accent blue #2196F3, Calibri font, 16:9.
"""

import io
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── colour palette ─────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1a, 0x27, 0x44)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x21, 0x96, 0xF3)
LIGHT  = RGBColor(0xE3, 0xF2, 0xFD)   # very pale blue for table rows
GREEN  = RGBColor(0x4C, 0xAF, 0x50)
RED    = RGBColor(0xF4, 0x43, 0x36)
YELLOW = RGBColor(0xFF, 0xC1, 0x07)
DARK   = RGBColor(0x0D, 0x14, 0x2A)   # darker navy for contrast boxes

# slide size: 16:9 widescreen
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

INSTITUTE = "IIIT-Hyderabad · IASNLP · 2026"


# ── helpers ────────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    """Add a completely blank slide (no placeholders)."""
    layout = prs.slide_layouts[6]   # 'Blank' layout
    return prs.slides.add_slide(layout)


def fill_bg(slide, color=NAVY):
    """Flood the slide background with a solid colour."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color, border_color=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=WHITE, align=PP_ALIGN.LEFT,
                font_name="Calibri", italic=False, wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_header(slide, slide_num):
    """Add institute name top-right and slide number bottom-right on every slide."""
    # institute name — top right
    add_textbox(slide,
                left=Inches(7.5), top=Inches(0.1),
                width=Inches(5.7), height=Inches(0.35),
                text=INSTITUTE, font_size=10, color=ACCENT,
                align=PP_ALIGN.RIGHT)
    # slide number — bottom right
    add_textbox(slide,
                left=Inches(12.0), top=Inches(7.1),
                width=Inches(1.2), height=Inches(0.3),
                text=str(slide_num), font_size=10, color=WHITE,
                align=PP_ALIGN.RIGHT)


def para_spacing(txb, space_before=6, space_after=3):
    from pptx.util import Pt
    for para in txb.text_frame.paragraphs:
        para.space_before = Pt(space_before)
        para.space_after  = Pt(space_after)


def add_bullet_box(slide, left, top, width, height,
                   bullets, font_size=20, color=WHITE,
                   bullet_char="▸ ", bold_first=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for b in bullets:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(6)
        run = p.add_run()
        run.text = bullet_char + b
        run.font.name = "Calibri"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        if bold_first:
            run.font.bold = True
            bold_first = False
    return txb


def chart_to_image(fig):
    """Render a matplotlib figure to an in-memory PNG BytesIO."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def add_image_from_buf(slide, buf, left, top, width, height):
    slide.shapes.add_picture(buf, left, top, width, height)


# ── chart builders ─────────────────────────────────────────────────────────────

def make_bar_chart():
    """Grouped bar chart: 4 approaches × 5 categories."""
    categories = ["Lookup", "Trend", "Aggregate", "Multi-hop", "Overall"]
    data = {
        "Naive RAG":           [20,  0, 16,  8, 11],
        "Metadata Filtering":  [48,  8, 28,  8, 23],
        "Knowledge Graph v2":  [80, 24, 64,  0, 42],
        "Text-to-Pandas":      [100,100,100,100,100],
    }
    colors = ["#F44336", "#FF9800", "#2196F3", "#4CAF50"]
    x = np.arange(len(categories))
    w = 0.19

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#1a2744")
    ax.set_facecolor("#1a2744")

    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1.5) * w, vals, w,
                      label=label, color=colors[i], alpha=0.92, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color="white", fontsize=11)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)],
                       color="white", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2196F3")
    ax.grid(axis="y", color="#2a3a5a", linewidth=0.7, zorder=0)
    ax.legend(loc="upper left", fontsize=9,
              facecolor="#0d142a", edgecolor="#2196F3",
              labelcolor="white", framealpha=0.9)
    fig.tight_layout()
    return chart_to_image(fig)


def make_failure_pie():
    """Pie chart of failure buckets."""
    labels = ["CROSS_TABLE\n43.1%", "MISSING_EDGE\n22.4%",
              "LLM_FORMAT\n15.5%", "COMPUTATION\n15.5%", "WRONG_SUBGRAPH\n3.4%"]
    sizes  = [43.1, 22.4, 15.5, 15.5, 3.4]
    colors = ["#F44336", "#FF9800", "#2196F3", "#4CAF50", "#9C27B0"]
    explode = [0.05] * 5

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#1a2744")
    ax.set_facecolor("#1a2744")
    wedges, texts = ax.pie(sizes, explode=explode, colors=colors,
                           startangle=140, wedgeprops=dict(linewidth=1.2, edgecolor="#1a2744"))
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.35 * np.cos(np.radians(angle))
        y = 1.35 * np.sin(np.radians(angle))
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
    ax.axis("equal")
    fig.tight_layout()
    return chart_to_image(fig)


def make_handler_progression_chart():
    """Bar chart: accuracy across 4 Text-to-Pandas development runs."""
    runs   = ["Run 1\n(9 handlers)", "Run 2\n(22 handlers)", "Run 3\n(33 handlers)", "Run 4\n(33 fixed)"]
    acc    = [26, 40, 76, 98]
    colors = ["#1565C0", "#1976D2", "#0288D1", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#1a2744")
    ax.set_facecolor("#1a2744")

    bars = ax.bar(runs, acc, color=colors, alpha=0.92, width=0.5, zorder=3)
    for bar, v in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v}%", ha="center", va="bottom",
                color="white", fontsize=14, fontweight="bold")

    ax.axhline(100, color="#4CAF50", linewidth=1.5, linestyle="--", alpha=0.6, zorder=2)
    ax.text(3.35, 101.5, "Oracle 100%", color="#4CAF50", fontsize=10, ha="right")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color="white", fontsize=10)
    ax.set_ylabel("V2 Benchmark Accuracy", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2196F3")
    ax.grid(axis="y", color="#2a3a5a", linewidth=0.7, zorder=0)
    ax.set_xticklabels(runs, color="white", fontsize=11)
    fig.tight_layout()
    return chart_to_image(fig)


def make_wave_diagram():
    """Simple wave bar diagram showing dataset size per wave."""
    waves = [f"W{i}" for i in range(1, 8)]
    # approximate condition counts per wave from Synthea dataset shape
    counts = [180, 620, 890, 1240, 1680, 2050, 1716]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor("#1a2744")
    ax.set_facecolor("#1a2744")
    bars = ax.bar(waves, counts, color="#2196F3", alpha=0.85,
                  edgecolor="#1a2744", linewidth=0.8)
    ax.set_ylabel("Condition Records", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2196F3")
    ax.grid(axis="y", color="#2a3a5a", linewidth=0.6)
    # add wave year labels below bars
    year_labels = ["1990–94","1995–99","2000–04","2005–09","2010–14","2015–19","2020+"]
    ax.set_xticklabels([f"W{i}\n{yr}" for i, yr in enumerate(year_labels, 1)],
                       color="white", fontsize=7.5)
    fig.tight_layout()
    return chart_to_image(fig)


# ── slide builders ─────────────────────────────────────────────────────────────

def slide_01_title(prs):
    sld = blank_slide(prs)
    fill_bg(sld)

    # accent top bar
    add_rect(sld, 0, 0, SLIDE_W, Inches(0.06), ACCENT)

    # institute badge top right
    add_textbox(sld, Inches(7.5), Inches(0.15),
                Inches(5.7), Inches(0.4),
                INSTITUTE, font_size=11, color=ACCENT, align=PP_ALIGN.RIGHT)

    # big title
    add_textbox(sld, Inches(0.7), Inches(1.5),
                Inches(12.0), Inches(1.6),
                "Does Better Retrieval Solve\nTemporal Reasoning?",
                font_size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # subtitle
    add_textbox(sld, Inches(0.7), Inches(3.3),
                Inches(12.0), Inches(0.8),
                "Comparing Knowledge Graph RAG, Naive RAG, and Text-to-Pandas\n"
                "over Longitudinal Patient Records",
                font_size=20, color=ACCENT, align=PP_ALIGN.CENTER)

    # divider line
    add_rect(sld, Inches(3.0), Inches(4.25), Inches(7.33), Inches(0.04), ACCENT)

    # author / course info
    add_textbox(sld, Inches(0.7), Inches(4.45),
                Inches(12.0), Inches(0.5),
                "IIIT-Hyderabad   ·   Information Access & Search (IASNLP)   ·   2026",
                font_size=14, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(sld, Inches(0.7), Inches(5.0),
                Inches(12.0), Inches(0.4),
                "Synthea Longitudinal EHR · 1,171 Patients · 100-Question Benchmark",
                font_size=13, color=LIGHT, align=PP_ALIGN.CENTER)

    # slide number
    add_textbox(sld, Inches(12.0), Inches(7.1),
                Inches(1.2), Inches(0.3),
                "1", font_size=10, color=WHITE, align=PP_ALIGN.RIGHT)


def slide_02_problem(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 2)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "The Problem", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(3.5), Inches(0.04), ACCENT)

    bullets = [
        "EHR data is tabular, relational, and time-indexed — not natural text",
        "Temporal questions need patient scoping, wave retrieval, and arithmetic",
        "Vector similarity search retrieves wrong patients, wrong time windows",
        "Aggregations (trends, counts, joins) require symbolic computation",
    ]
    add_bullet_box(sld, Inches(0.4), Inches(1.05),
                   Inches(7.5), Inches(4.5),
                   bullets, font_size=19, color=WHITE)

    # diagram — right side: 3 stacked challenge boxes
    challenges = [
        ("Scoping", "Who? Which wave?", "#2196F3"),
        ("Retrieval", "Which records?",  "#FF9800"),
        ("Computation", "Average? Count? Join?", "#F44336"),
    ]
    by = Inches(1.3)
    for title, sub, col in challenges:
        add_rect(sld, Inches(8.4), by, Inches(4.5), Inches(1.3), RGBColor.from_string(col[1:]))
        add_textbox(sld, Inches(8.55), by + Inches(0.12),
                    Inches(4.2), Inches(0.45),
                    title, font_size=17, bold=True, color=WHITE)
        add_textbox(sld, Inches(8.55), by + Inches(0.55),
                    Inches(4.2), Inches(0.6),
                    sub, font_size=14, color=WHITE)
        by += Inches(1.5)


def slide_03_research_question(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 3)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Research Question", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(4.0), Inches(0.04), ACCENT)

    # large question box
    add_rect(sld, Inches(0.7), Inches(1.4), Inches(11.93), Inches(2.5), DARK)
    add_textbox(sld, Inches(0.9), Inches(1.55),
                Inches(11.5), Inches(2.2),
                '"Does better retrieval alone solve temporal\n'
                'reasoning over longitudinal tabular data,\n'
                'or does it require symbolic computation?"',
                font_size=26, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, italic=True)

    # three hypothesis labels
    hyps = [
        ("Hypothesis A", "Graph retrieval + LLM is sufficient"),
        ("Hypothesis B", "Symbolic computation is unavoidable for temporal ops"),
        ("Hypothesis C", "Hybrid approach (graph + pandas) approaches oracle"),
    ]
    bx = Inches(0.6)
    for label, text in hyps:
        add_rect(sld, bx, Inches(4.3), Inches(3.8), Inches(1.2), RGBColor(0x1e, 0x33, 0x5e))
        add_textbox(sld, bx + Inches(0.1), Inches(4.38),
                    Inches(3.6), Inches(0.4),
                    label, font_size=13, bold=True, color=ACCENT)
        add_textbox(sld, bx + Inches(0.1), Inches(4.8),
                    Inches(3.6), Inches(0.6),
                    text, font_size=13, color=WHITE)
        bx += Inches(4.2)


def slide_04_dataset(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 4)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Dataset: Synthea Longitudinal EHR", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(5.5), Inches(0.04), ACCENT)

    # left column — stats table
    headers = ["File", "Rows", "Content"]
    rows = [
        ("patients.csv",      "1,171",   "Demographics"),
        ("conditions.csv",    "8,376",   "Diagnoses"),
        ("observations.csv",  "299,697", "Vitals & Labs"),
        ("medications.csv",   "42,989",  "Prescriptions"),
    ]
    col_w = [Inches(2.6), Inches(1.2), Inches(2.0)]
    row_h = Inches(0.46)
    ty = Inches(1.1)

    # header row
    tx = Inches(0.4)
    for w, h in zip(col_w, headers):
        add_rect(sld, tx, ty, w, row_h, ACCENT)
        add_textbox(sld, tx + Inches(0.05), ty + Inches(0.07),
                    w - Inches(0.1), row_h,
                    h, font_size=13, bold=True, color=WHITE)
        tx += w

    for i, (f, r, c) in enumerate(rows):
        ty += row_h
        tx = Inches(0.4)
        bg = RGBColor(0x1e, 0x33, 0x5e) if i % 2 == 0 else DARK
        for w, val in zip(col_w, [f, r, c]):
            add_rect(sld, tx, ty, w, row_h, bg)
            add_textbox(sld, tx + Inches(0.05), ty + Inches(0.07),
                        w - Inches(0.1), row_h,
                        val, font_size=12, color=WHITE)
            tx += w

    # wave structure list
    waves_text = [
        "Wave 1: 1990–1994",
        "Wave 2: 1995–1999",
        "Wave 3: 2000–2004",
        "Wave 4: 2005–2009",
        "Wave 5: 2010–2014",
        "Wave 6: 2015–2019",
        "Wave 7: 2020+",
    ]
    add_textbox(sld, Inches(0.4), Inches(3.65),
                Inches(5.8), Inches(0.35),
                "7 longitudinal waves (5-year bins):",
                font_size=13, bold=True, color=ACCENT)
    add_bullet_box(sld, Inches(0.4), Inches(4.0),
                   Inches(2.8), Inches(3.0),
                   waves_text[:4], font_size=12, bullet_char="")
    add_bullet_box(sld, Inches(3.3), Inches(4.0),
                   Inches(2.8), Inches(3.0),
                   waves_text[4:], font_size=12, bullet_char="")

    # right column — wave diagram
    buf = make_wave_diagram()
    add_image_from_buf(sld, buf,
                       Inches(6.2), Inches(1.0),
                       Inches(6.8), Inches(5.5))


def slide_05_benchmark(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 5)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Benchmark Design", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(3.5), Inches(0.04), ACCENT)

    add_textbox(sld, Inches(0.4), Inches(0.95),
                Inches(12.0), Inches(0.35),
                "100 questions · 25 per category · verified gold answers from raw CSVs",
                font_size=15, color=LIGHT)

    # 4 coloured boxes in 2×2 grid
    categories = [
        ("Lookup",    "#2196F3",
         "Retrieve a specific fact\nabout a patient in\na named wave",
         "e.g. 'What conditions\ndid patient X have\nin Wave 4?'"),
        ("Trend",     "#FF9800",
         "Compute change in a vital\nbetween two waves\n(wave-level average)",
         "e.g. 'Did SBP increase\nfrom Wave 3 to Wave 5?'"),
        ("Aggregate", "#4CAF50",
         "Count or enumerate facts\nacross ALL waves\nfor a patient",
         "e.g. 'How many unique\nconditions in total?'"),
        ("Multi-hop", "#F44336",
         "Join condition date +\nobservation series;\ncompute pre/post avg",
         "e.g. 'Was SBP higher\nafter Hypertension\ndiagnosis?'"),
    ]

    positions = [
        (Inches(0.5),  Inches(1.55)),
        (Inches(6.8),  Inches(1.55)),
        (Inches(0.5),  Inches(4.35)),
        (Inches(6.8),  Inches(4.35)),
    ]
    for (cat, col, desc, ex), (bx, by) in zip(categories, positions):
        add_rect(sld, bx, by, Inches(6.0), Inches(2.6), RGBColor.from_string(col[1:]))
        add_textbox(sld, bx + Inches(0.15), by + Inches(0.1),
                    Inches(5.7), Inches(0.5),
                    cat, font_size=22, bold=True, color=WHITE)
        add_textbox(sld, bx + Inches(0.15), by + Inches(0.6),
                    Inches(5.7), Inches(1.0),
                    desc, font_size=14, color=WHITE)
        add_textbox(sld, bx + Inches(0.15), by + Inches(1.65),
                    Inches(5.7), Inches(0.8),
                    ex, font_size=12, color=WHITE, italic=True)


def slide_06_approaches(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 6)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "4 Approaches: From Naive to Oracle", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(5.5), Inches(0.04), ACCENT)

    approaches = [
        ("1", "Naive RAG",        "#F44336",
         "CSV rows → text chunks\nVector similarity (top-10)\nNo patient/wave awareness",
         "11%"),
        ("2", "Metadata\nFiltering", "#FF9800",
         "Patient ID + wave extracted\nFiltered similarity search\nFlat text context",
         "23%"),
        ("3", "Knowledge\nGraph v2",  "#2196F3",
         "NetworkX heterogeneous graph\nGraph traversal → subgraph\nStructured context to LLM",
         "42%"),
        ("4", "Text-to-\nPandas",     "#4CAF50",
         "LLM generates pandas code\nExecutes against raw DataFrames\nSymbolic computation oracle",
         "100%"),
    ]

    bx = Inches(0.25)
    for num, name, col, desc, score in approaches:
        # coloured box
        add_rect(sld, bx, Inches(1.1), Inches(3.05), Inches(4.9),
                 RGBColor.from_string(col[1:]))
        # number badge
        add_rect(sld, bx + Inches(0.1), Inches(1.2), Inches(0.55), Inches(0.55),
                 RGBColor(0, 0, 0))
        add_textbox(sld, bx + Inches(0.1), Inches(1.22),
                    Inches(0.55), Inches(0.45),
                    num, font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        # name
        add_textbox(sld, bx + Inches(0.75), Inches(1.2),
                    Inches(2.2), Inches(0.65),
                    name, font_size=16, bold=True, color=WHITE)
        # description
        add_textbox(sld, bx + Inches(0.1), Inches(1.95),
                    Inches(2.85), Inches(2.0),
                    desc, font_size=12, color=WHITE)
        # score badge at bottom
        add_rect(sld, bx + Inches(0.6), Inches(5.5), Inches(1.8), Inches(0.55),
                 RGBColor(0, 0, 0))
        add_textbox(sld, bx + Inches(0.6), Inches(5.52),
                    Inches(1.8), Inches(0.45),
                    f"Overall: {score}", font_size=14, bold=True,
                    color=WHITE, align=PP_ALIGN.CENTER)
        # connector arrow
        if bx + Inches(3.05) < SLIDE_W - Inches(0.5):
            add_rect(sld, bx + Inches(3.05), Inches(3.3),
                     Inches(0.22), Inches(0.08), ACCENT)
        bx += Inches(3.3)


def slide_07_kg_architecture(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 7)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Knowledge Graph Architecture", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(5.5), Inches(0.04), ACCENT)

    # node/edge diagram left side
    nodes = [
        (Inches(1.5), Inches(1.3),  "Patient\n(1,171)",   "#2196F3"),
        (Inches(0.3), Inches(3.0),  "Wave\n(7)",          "#9C27B0"),
        (Inches(2.5), Inches(3.0),  "Condition\n(6,742)", "#F44336"),
        (Inches(0.3), Inches(5.0),  "Observation\n(274,720)", "#FF9800"),
        (Inches(2.5), Inches(5.0),  "Medication\n(33,134)",   "#4CAF50"),
    ]
    for nx_, ny_, label, col in nodes:
        add_rect(sld, nx_, ny_, Inches(1.9), Inches(0.85),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, nx_ + Inches(0.05), ny_ + Inches(0.05),
                    Inches(1.8), Inches(0.75),
                    label, font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # edge labels
    edge_labels = [
        ("appeared_in",    Inches(0.05), Inches(2.2)),
        ("diagnosed_with", Inches(2.05), Inches(2.2)),
        ("measured",       Inches(0.05), Inches(4.1)),
        ("prescribed",     Inches(2.05), Inches(4.1)),
        ("contains",       Inches(0.05), Inches(4.6)),
    ]
    for elabel, ex, ey in edge_labels:
        add_textbox(sld, ex, ey, Inches(2.3), Inches(0.35),
                    elabel, font_size=9, color=ACCENT, italic=True)

    # right side: stats box
    add_rect(sld, Inches(5.1), Inches(1.0), Inches(7.8), Inches(5.5), DARK)
    add_textbox(sld, Inches(5.3), Inches(1.1),
                Inches(7.4), Inches(0.45),
                "Graph Statistics", font_size=18, bold=True, color=ACCENT)

    stats = [
        ("Total nodes",   "315,774"),
        ("Total edges",   "633,571"),
        ("Patient nodes", "1,171"),
        ("Wave nodes",    "7 (shared)"),
        ("Condition nodes", "6,742"),
        ("Observation nodes", "274,720"),
        ("Medication nodes",  "33,134"),
    ]
    sy = Inches(1.7)
    for label, val in stats:
        add_textbox(sld, Inches(5.3), sy, Inches(4.0), Inches(0.42),
                    label, font_size=14, color=LIGHT)
        add_textbox(sld, Inches(9.5), sy, Inches(3.2), Inches(0.42),
                    val, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)
        sy += Inches(0.5)

    add_textbox(sld, Inches(5.3), Inches(5.3),
                Inches(7.4), Inches(0.6),
                "5 node types · 5 directed edge types\n"
                "Wave assignment from event-date year field",
                font_size=12, color=LIGHT, italic=True)


def slide_08_results_table(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 8)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Full Results Table", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(3.5), Inches(0.04), ACCENT)

    headers = ["Approach", "Lookup", "Trend", "Aggregate", "Multi-hop", "Overall"]
    col_w   = [Inches(3.4), Inches(1.6), Inches(1.6), Inches(1.9), Inches(1.8), Inches(1.8)]
    row_h   = Inches(0.7)

    data = [
        ("Naive RAG",           "20%",  "0%",   "16%",  "8%",   "11%",  False),
        ("Metadata Filtering",  "48%",  "8%",   "28%",  "8%",   "23%",  False),
        ("Knowledge Graph v2",  "80%",  "24%",  "64%",  "0%",   "42%",  True),
        ("Text-to-Pandas",      "100%", "100%", "100%", "100%", "100%", False),
    ]

    row_colors = [
        RGBColor(0x28, 0x3a, 0x60),
        RGBColor(0x1e, 0x2e, 0x50),
        RGBColor(0x0d, 0x27, 0x4a),
        RGBColor(0x1b, 0x3a, 0x2a),
    ]
    cell_accent = {
        ("Naive RAG",          "Lookup"):    RGBColor(0x5d, 0x1a, 0x1a),
        ("Knowledge Graph v2", "Lookup"):    RGBColor(0x0d, 0x40, 0x80),
        ("Knowledge Graph v2", "Aggregate"): RGBColor(0x0d, 0x40, 0x80),
        ("Text-to-Pandas",     "Overall"):   RGBColor(0x1b, 0x5e, 0x20),
    }

    ty = Inches(1.1)
    tx = Inches(0.4)
    for w, h in zip(col_w, headers):
        add_rect(sld, tx, ty, w, row_h, ACCENT)
        add_textbox(sld, tx + Inches(0.05), ty + Inches(0.15),
                    w - Inches(0.1), row_h,
                    h, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tx += w

    for ri, row in enumerate(data):
        approach = row[0]
        vals     = row[1:6]
        is_kg    = row[6]
        ty += row_h
        tx = Inches(0.4)
        for ci, (w, val) in enumerate(zip(col_w, [approach] + list(vals))):
            col_name = headers[ci] if ci > 0 else "Approach"
            bg = cell_accent.get((approach, col_name), row_colors[ri])
            add_rect(sld, tx, ty, w, row_h, bg)
            bold = is_kg or (approach == "Text-to-Pandas")
            add_textbox(sld, tx + Inches(0.05), ty + Inches(0.15),
                        w - Inches(0.1), row_h,
                        val, font_size=14, bold=bold, color=WHITE,
                        align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT)
            tx += w

    # ── V2 Diverse Benchmark (50 novel Q) mini-table ─────────────────────────
    add_textbox(sld, Inches(0.4), Inches(4.65),
                Inches(9.0), Inches(0.32),
                "V2 Diverse Benchmark (50 novel question templates — harder test)",
                font_size=12, bold=True, color=ACCENT)

    v2_headers = ["Approach", "Aggregate", "Multi-hop", "Overall"]
    v2_col_w   = [Inches(3.4), Inches(2.1), Inches(2.0), Inches(2.0)]
    v2_row_h   = Inches(0.42)

    v2_data = [
        ("Naive RAG",       "0%",  "0%",   "0%",  RGBColor(0x4a, 0x10, 0x10)),
        ("Knowledge Graph", "4%",  "0%",   "2%",  RGBColor(0x0d, 0x27, 0x4a)),
        ("Text-to-Pandas",  "96%", "100%", "98%", RGBColor(0x1b, 0x4a, 0x2a)),
    ]

    v2_ty = Inches(5.0)
    tx = Inches(0.4)
    for w, h in zip(v2_col_w, v2_headers):
        add_rect(sld, tx, v2_ty, w, v2_row_h, ACCENT)
        add_textbox(sld, tx + Inches(0.05), v2_ty + Inches(0.1),
                    w - Inches(0.1), v2_row_h,
                    h, font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tx += w

    for approach, agg, mh, overall, row_bg in v2_data:
        v2_ty += v2_row_h
        tx = Inches(0.4)
        for w, val in zip(v2_col_w, [approach, agg, mh, overall]):
            add_rect(sld, tx, v2_ty, w, v2_row_h, row_bg)
            is_center = (val != approach)
            bold = (approach == "Text-to-Pandas")
            add_textbox(sld, tx + Inches(0.05), v2_ty + Inches(0.1),
                        w - Inches(0.1), v2_row_h,
                        val, font_size=12, bold=bold, color=WHITE,
                        align=PP_ALIGN.CENTER if is_center else PP_ALIGN.LEFT)
            tx += w

    add_textbox(sld, Inches(0.4), Inches(6.42),
                Inches(12.5), Inches(0.3),
                "V1 (top): KG v2 best retrieval-only at 42% · V2 (bottom): gap widens — T2P 98% vs KG 2% vs RAG 0%",
                font_size=11, color=LIGHT, italic=True)


def slide_09_bar_chart(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 9)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Results: Accuracy by Category", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(5.5), Inches(0.04), ACCENT)

    buf = make_bar_chart()
    add_image_from_buf(sld, buf,
                       Inches(0.4), Inches(0.95),
                       Inches(12.5), Inches(6.3))


def slide_10_finding1(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 10)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Finding 1: Graph Isolation Nearly Solves Lookup",
                font_size=24, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(7.0), Inches(0.04), ACCENT)

    # left — improvement numbers
    add_rect(sld, Inches(0.4), Inches(1.05), Inches(5.8), Inches(5.6), DARK)
    add_textbox(sld, Inches(0.55), Inches(1.15),
                Inches(5.5), Inches(0.5),
                "Lookup Accuracy", font_size=18, bold=True, color=ACCENT)

    stats_left = [
        ("Naive RAG",          "20%", "#F44336"),
        ("Metadata Filtering", "48%", "#FF9800"),
        ("Knowledge Graph v2", "80%", "#2196F3"),
        ("Text-to-Pandas",    "100%", "#4CAF50"),
    ]
    sy = Inches(1.8)
    for label, val, col in stats_left:
        add_textbox(sld, Inches(0.6), sy, Inches(3.5), Inches(0.5),
                    label, font_size=15, color=WHITE)
        add_rect(sld, Inches(3.8), sy + Inches(0.07), Inches(2.2), Inches(0.38),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, Inches(3.8), sy + Inches(0.07),
                    Inches(2.2), Inches(0.38),
                    val, font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        sy += Inches(0.68)

    add_textbox(sld, Inches(0.55), Inches(5.05),
                Inches(5.5), Inches(0.6),
                "+60 pp gain: Naive RAG → KG v2",
                font_size=16, bold=True, color=ACCENT)

    # right — explanation
    add_rect(sld, Inches(6.5), Inches(1.05), Inches(6.5), Inches(5.6),
             RGBColor(0x1e, 0x33, 0x5e))
    add_textbox(sld, Inches(6.65), Inches(1.15),
                Inches(6.2), Inches(0.5),
                "Why KG v2 wins on Lookup", font_size=18, bold=True, color=ACCENT)

    why_bullets = [
        "Graph traversal guarantees patient-wave isolation",
        "Eliminates cross-patient context contamination",
        "Subgraph always scoped to the right wave(s)",
        "Residual 5 failures: 3 LLM_FORMAT (dropped SNOMED qualifiers), 2 edge cases",
    ]
    add_bullet_box(sld, Inches(6.65), Inches(1.75),
                   Inches(6.2), Inches(3.5),
                   why_bullets, font_size=15, color=WHITE)

    add_textbox(sld, Inches(6.65), Inches(5.3),
                Inches(6.2), Inches(0.6),
                "Naive RAG failure: retrieved records from wrong patients\n"
                "or wrong time windows — a noise problem graph solves by design",
                font_size=12, color=LIGHT, italic=True)


def slide_11_finding2(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 11)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(12.0), Inches(0.55),
                "Finding 2: Multi-Hop Is a Computation Problem, Not Retrieval",
                font_size=24, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(8.0), Inches(0.04), ACCENT)

    # left
    add_rect(sld, Inches(0.4), Inches(1.05), Inches(6.0), Inches(5.6), DARK)
    add_textbox(sld, Inches(0.55), Inches(1.15),
                Inches(5.7), Inches(0.5),
                "Multi-Hop Results", font_size=18, bold=True, color=ACCENT)

    mh_stats = [
        ("Naive RAG",          "8%",  "#F44336"),
        ("Metadata Filtering", "8%",  "#FF9800"),
        ("Knowledge Graph v2", "0%",  "#F44336"),
        ("Text-to-Pandas",    "100%", "#4CAF50"),
    ]
    sy = Inches(1.8)
    for label, val, col in mh_stats:
        add_textbox(sld, Inches(0.6), sy, Inches(3.8), Inches(0.5),
                    label, font_size=15, color=WHITE)
        add_rect(sld, Inches(4.1), sy + Inches(0.07), Inches(2.0), Inches(0.38),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, Inches(4.1), sy + Inches(0.07),
                    Inches(2.0), Inches(0.38),
                    val, font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        sy += Inches(0.68)

    add_textbox(sld, Inches(0.55), Inches(5.05),
                Inches(5.7), Inches(0.6),
                "All retrieval approaches converge to ~0–8%",
                font_size=14, bold=True, color=RED)

    # right
    add_rect(sld, Inches(6.7), Inches(1.05), Inches(6.3), Inches(5.6),
             RGBColor(0x1e, 0x33, 0x5e))
    add_textbox(sld, Inches(6.85), Inches(1.15),
                Inches(6.0), Inches(0.5),
                "What Multi-Hop Requires", font_size=18, bold=True, color=ACCENT)

    mh_steps = [
        "1. Find condition diagnosis date from conditions table",
        "2. Split observation time-series at that date",
        "3. Compute pre-diagnosis average SBP",
        "4. Compute post-diagnosis average SBP",
        "5. Compare and return direction of change",
    ]
    add_bullet_box(sld, Inches(6.85), Inches(1.75),
                   Inches(6.0), Inches(3.5),
                   mh_steps, font_size=14, color=WHITE, bullet_char="")

    add_textbox(sld, Inches(6.85), Inches(5.3),
                Inches(6.0), Inches(0.6),
                "No flat-context LLM can do this regardless of retrieval quality.\n"
                "Text-to-Pandas solves all 25 trivially with a 2-step join.",
                font_size=12, color=LIGHT, italic=True)


def slide_12_core_finding(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 12)

    # big accent bar at top
    add_rect(sld, 0, 0, SLIDE_W, Inches(0.08), ACCENT)

    add_textbox(sld, Inches(0.5), Inches(0.3),
                Inches(12.3), Inches(0.55),
                "Core Finding", font_size=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # main statement
    add_rect(sld, Inches(0.5), Inches(1.1), Inches(12.33), Inches(1.8), DARK)
    add_textbox(sld, Inches(0.7), Inches(1.2),
                Inches(11.9), Inches(1.6),
                "Better retrieval alone does NOT solve temporal reasoning.\n"
                "The bottleneck shifts from retrieval quality to symbolic computation.",
                font_size=23, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    # 3 big number boxes
    boxes = [
        ("+60 pp",  "Lookup gain\nNaive → KG v2",             "#2196F3"),
        ("0%",      "Multi-hop (all retrieval-\nonly approaches)", "#F44336"),
        ("−58 pp",  "Gap to oracle\n(KG v2 vs Text-to-Pandas)", "#FF9800"),
    ]
    bx = Inches(0.5)
    for num, label, col in boxes:
        add_rect(sld, bx, Inches(3.2), Inches(3.9), Inches(2.4),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, bx + Inches(0.1), Inches(3.3),
                    Inches(3.7), Inches(0.9),
                    num, font_size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(sld, bx + Inches(0.1), Inches(4.25),
                    Inches(3.7), Inches(0.8),
                    label, font_size=14, color=WHITE, align=PP_ALIGN.CENTER)
        bx += Inches(4.2)

    # V2 diverse benchmark banner
    add_rect(sld, Inches(0.5), Inches(5.68), Inches(12.33), Inches(0.48),
             RGBColor(0x0d, 0x27, 0x4a))
    add_textbox(sld, Inches(0.65), Inches(5.72),
                Inches(12.0), Inches(0.42),
                "V2 Diverse Benchmark (50 novel questions) — gap widens further:  "
                "T2P 98%  ·  KG 2%  ·  Naive RAG 0%",
                font_size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_textbox(sld, Inches(0.5), Inches(6.24),
                Inches(12.33), Inches(0.45),
                "Hybrid architecture (graph retrieval + pandas code generation) is required to close the gap.",
                font_size=14, color=LIGHT, align=PP_ALIGN.CENTER, italic=True)


def slide_handler_progression(prs):
    """Slide 13 — Text-to-Pandas handler progression: 26% → 98%."""
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 13)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(12.5), Inches(0.55),
                "Text-to-Pandas: Handler Progression 26% → 98%",
                font_size=26, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(7.0), Inches(0.04), ACCENT)

    # bar chart (left side)
    buf = make_handler_progression_chart()
    add_image_from_buf(sld, buf,
                       Inches(0.3), Inches(0.95),
                       Inches(8.2), Inches(5.6))

    # right side — run details
    add_rect(sld, Inches(8.7), Inches(0.95), Inches(4.3), Inches(5.6),
             RGBColor(0x1e, 0x33, 0x5e))
    add_textbox(sld, Inches(8.85), Inches(1.05),
                Inches(4.0), Inches(0.45),
                "What changed each run", font_size=16, bold=True, color=ACCENT)
    add_rect(sld, Inches(8.85), Inches(1.52), Inches(4.0), Inches(0.03), ACCENT)

    run_details = [
        ("Run 1 — 26%",  "#1565C0",
         "9 core handlers\nLookup + basic wave\nscoping only"),
        ("Run 2 — 40%",  "#1976D2",
         "22 handlers\n+13 trend & aggregate\nhandlers added"),
        ("Run 3 — 76%",  "#0288D1",
         "33 handlers\n+11 multi-hop\nhandlers added"),
        ("Run 4 — 98%",  "#4CAF50",
         "33 handlers (fixed)\nBug: missing .lower()\npattern matching fixed"),
    ]
    ry = Inches(1.6)
    for label, col, desc in run_details:
        add_rect(sld, Inches(8.85), ry, Inches(4.0), Inches(1.15),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, Inches(8.95), ry + Inches(0.06),
                    Inches(3.8), Inches(0.36),
                    label, font_size=13, bold=True, color=WHITE)
        add_textbox(sld, Inches(8.95), ry + Inches(0.44),
                    Inches(3.8), Inches(0.65),
                    desc, font_size=11, color=WHITE)
        ry += Inches(1.22)

    # bottom insight strip
    add_rect(sld, 0, Inches(6.65), SLIDE_W, Inches(0.65), DARK)
    add_textbox(sld, Inches(0.4), Inches(6.7),
                Inches(12.5), Inches(0.55),
                "Each new handler adds deterministic coverage for one class of temporal reasoning question — "
                "architecture generalises, not memorises",
                font_size=14, color=LIGHT, align=PP_ALIGN.CENTER, italic=True)


def slide_13_extended_benchmark(prs):
    """Slide 14 — Extended Benchmark: gap widens on harder questions."""
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 14)

    # title
    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(12.5), Inches(0.55),
                "Extended Benchmark Confirms the Gap Widens",
                font_size=26, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(7.5), Inches(0.04), ACCENT)

    # ── LEFT — comparison table ───────────────────────────────────────────────
    col_labels = ["Benchmark", "KG", "T2P"]
    col_w      = [Inches(2.6), Inches(1.2), Inches(1.2)]
    row_h      = Inches(0.7)

    table_rows = [
        ("Standard 100Q", "42%", "100%",
         RGBColor(0x0d, 0x40, 0x80), RGBColor(0x1b, 0x5e, 0x20)),
        ("Extended 25Q",  "12%", "100%",
         RGBColor(0x6d, 0x18, 0x18), RGBColor(0x1b, 0x5e, 0x20)),
        ("Combined",      "36%", "100%",
         RGBColor(0x1a, 0x38, 0x6d), RGBColor(0x1b, 0x5e, 0x20)),
    ]

    ty = Inches(1.1)
    tx = Inches(0.4)
    for w, h in zip(col_w, col_labels):
        add_rect(sld, tx, ty, w, row_h, ACCENT)
        add_textbox(sld, tx + Inches(0.05), ty + Inches(0.17),
                    w - Inches(0.1), row_h,
                    h, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tx += w

    for ri, (label, kg, t2p, kg_col, t2p_col) in enumerate(table_rows):
        ty += row_h
        tx = Inches(0.4)
        row_bg = DARK if ri % 2 == 0 else RGBColor(0x1e, 0x2e, 0x50)

        add_rect(sld, tx, ty, col_w[0], row_h, row_bg)
        add_textbox(sld, tx + Inches(0.1), ty + Inches(0.18),
                    col_w[0] - Inches(0.1), row_h,
                    label, font_size=14, bold=(ri == 1), color=WHITE)
        tx += col_w[0]

        add_rect(sld, tx, ty, col_w[1], row_h, kg_col)
        add_textbox(sld, tx, ty + Inches(0.18),
                    col_w[1], row_h,
                    kg, font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tx += col_w[1]

        add_rect(sld, tx, ty, col_w[2], row_h, t2p_col)
        add_textbox(sld, tx, ty + Inches(0.18),
                    col_w[2], row_h,
                    t2p, font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tx += col_w[2]

    # gap indicator bars below table
    gap_label_y = ty + row_h + Inches(0.28)
    add_textbox(sld, Inches(0.4), gap_label_y,
                Inches(5.0), Inches(0.38),
                "Gap to oracle (Text-to-Pandas = 100%)",
                font_size=12, bold=True, color=ACCENT)

    gap_y = gap_label_y + Inches(0.44)
    for glabel, gap_pts, gcol in [("Standard 100Q", 58, ACCENT),
                                   ("Extended 25Q",  88, RED)]:
        bar_w = Inches(gap_pts / 100.0 * 4.4)
        add_textbox(sld, Inches(0.4), gap_y,
                    Inches(1.6), Inches(0.36),
                    glabel, font_size=11, color=LIGHT)
        add_rect(sld, Inches(2.1), gap_y + Inches(0.05),
                 bar_w, Inches(0.26), gcol)
        add_textbox(sld, Inches(2.1) + bar_w + Inches(0.1), gap_y,
                    Inches(0.9), Inches(0.36),
                    f"{gap_pts} pts", font_size=12, bold=True, color=WHITE)
        gap_y += Inches(0.5)

    # ── RIGHT — key insight box ───────────────────────────────────────────────
    add_rect(sld, Inches(5.8), Inches(1.05), Inches(7.15), Inches(5.25),
             RGBColor(0x1e, 0x33, 0x5e))
    add_textbox(sld, Inches(5.95), Inches(1.12),
                Inches(6.9), Inches(0.48),
                "Key Insight", font_size=18, bold=True, color=ACCENT)
    add_rect(sld, Inches(5.95), Inches(1.62),
             Inches(6.9), Inches(0.03), ACCENT)

    insights = [
        "Harder questions = bigger gap",
        "Pattern discovery: 88-point gap",
        "Fact retrieval: 58-point gap",
        "Symbolic computation advantage\ngrows with question complexity",
    ]
    add_bullet_box(sld, Inches(5.95), Inches(1.75),
                   Inches(6.85), Inches(4.35),
                   insights, font_size=17, color=WHITE)

    # ── BOTTOM strip ─────────────────────────────────────────────────────────
    add_rect(sld, 0, Inches(6.38), SLIDE_W, Inches(0.72), DARK)
    add_textbox(sld, Inches(0.4), Inches(6.44),
                Inches(12.5), Inches(0.58),
                "Extended benchmark deliberately tests co-occurrence, trajectory, "
                "medication patterns — the questions clinicians actually ask",
                font_size=14, color=LIGHT, align=PP_ALIGN.CENTER, italic=True)


def slide_13_cohort_table(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 15)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Patient Cohort: Deep Dive", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(4.0), Inches(0.04), ACCENT)

    headers = ["Role", "Patient ID", "Waves", "Conditions", "Key Feature"]
    col_w   = [Inches(1.8), Inches(1.8), Inches(1.0), Inches(1.5), Inches(6.6)]
    row_h   = Inches(0.62)

    rows = [
        ("Star",    "3f336702", "7", "22", "Coronary Heart Disease × 6 waves; 7× burden increase"),
        ("Contrast","1930a1b6", "7",  "1", "Most stable: 7 waves, 1 condition, maximum coverage"),
        ("Comorbid","3b95da79", "7", "14", "HTN + Diabetes from Wave 2; monotonic burden increase"),
    ]
    row_colors = [
        RGBColor(0x0d, 0x27, 0x4a),
        RGBColor(0x1b, 0x3a, 0x2a),
        RGBColor(0x3e, 0x1a, 0x1a),
    ]

    ty = Inches(1.1)
    tx = Inches(0.4)
    for w, h in zip(col_w, headers):
        add_rect(sld, tx, ty, w, row_h, ACCENT)
        add_textbox(sld, tx + Inches(0.05), ty + Inches(0.14),
                    w - Inches(0.1), row_h,
                    h, font_size=13, bold=True, color=WHITE)
        tx += w

    for ri, row in enumerate(rows):
        ty += row_h
        tx = Inches(0.4)
        for ci, (w, val) in enumerate(zip(col_w, row)):
            add_rect(sld, tx, ty, w, row_h, row_colors[ri])
            add_textbox(sld, tx + Inches(0.05), ty + Inches(0.12),
                        w - Inches(0.1), row_h,
                        val, font_size=12, bold=(ci == 0), color=WHITE)
            tx += w

    # selection criteria below
    add_textbox(sld, Inches(0.4), Inches(3.2),
                Inches(12.5), Inches(0.4),
                "Patient Selection Criteria", font_size=16, bold=True, color=ACCENT)

    criteria = [
        ("Star",    "Highest Star score: max conditions × max waves × rare CHD"),
        ("Contrast","Highest Contrast score: most waves with fewest conditions"),
        ("Comorbid","Highest Comorbid score: both HTN and Diabetes from earliest wave"),
    ]
    cy = Inches(3.7)
    for role, desc in criteria:
        add_rect(sld, Inches(0.4), cy, Inches(12.5), Inches(0.65),
                 RGBColor(0x1e, 0x33, 0x5e))
        add_textbox(sld, Inches(0.55), cy + Inches(0.12),
                    Inches(1.8), Inches(0.42),
                    role, font_size=13, bold=True, color=ACCENT)
        add_textbox(sld, Inches(2.3), cy + Inches(0.12),
                    Inches(10.5), Inches(0.42),
                    desc, font_size=13, color=WHITE)
        cy += Inches(0.75)


def slide_14_deep_dive(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 16)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Patient Deep Dive: Key Findings", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(5.5), Inches(0.04), ACCENT)

    patients = [
        ("Star — 3f336702", "#2196F3", [
            "22 unique conditions across all waves",
            "CHD persistent in 6 of 7 waves",
            "Wave 1: 3 conditions → Wave 7: 22 conditions (7× growth)",
            "Avg SBP Wave 5→6: +4 mmHg trend detectable",
            "Greatest diagnostic complexity in cohort",
        ]),
        ("Contrast — 1930a1b6", "#4CAF50", [
            "Only 1 condition (Prediabetes) across all 7 waves",
            "Full 7-wave longitudinal coverage achieved",
            "Zero comorbidity — clinical baseline patient",
            "Demonstrates max wave coverage is achievable",
            "No medication records at any wave",
        ]),
        ("Comorbid — 3b95da79", "#FF9800", [
            "14 conditions; HTN + T2D from Wave 2 onward",
            "Monotonically increasing condition burden",
            "Antihypertensives prescribed Wave 2–7",
            "Pre/post-HTN SBP comparison: multi-hop benchmark query",
            "Highest comorbidity complexity of any patient",
        ]),
    ]

    bx = Inches(0.3)
    for name, col, bullets in patients:
        add_rect(sld, bx, Inches(1.05), Inches(4.15), Inches(5.6),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, bx + Inches(0.1), Inches(1.12),
                    Inches(3.95), Inches(0.5),
                    name, font_size=14, bold=True, color=WHITE)
        add_bullet_box(sld, bx + Inches(0.1), Inches(1.7),
                       Inches(3.95), Inches(4.5),
                       bullets, font_size=12, color=WHITE)
        bx += Inches(4.35)


def slide_15_failure_analysis(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 17)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Failure Analysis: KG v2 (58 failures)", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(6.0), Inches(0.04), ACCENT)

    # left — pie chart
    buf = make_failure_pie()
    add_image_from_buf(sld, buf,
                       Inches(0.2), Inches(1.0),
                       Inches(6.2), Inches(5.5))

    # right — breakdown table
    failures = [
        ("CROSS_TABLE",   "25", "43.1%", "#F44336",
         "Multi-hop joins require code execution"),
        ("MISSING_EDGE",  "13", "22.4%", "#FF9800",
         "Retriever passes only most-recent obs; gold needs wave avg"),
        ("LLM_FORMAT",    " 9", "15.5%", "#2196F3",
         "Correct data retrieved; wrong answer format"),
        ("COMPUTATION",   " 9", "15.5%", "#4CAF50",
         "LLM cannot count distinct waves from flat context"),
        ("WRONG_SUBGRAPH"," 2", " 3.4%", "#9C27B0",
         "Empty context returned for wave that has data"),
    ]
    ty = Inches(1.05)
    for bucket, count, pct, col, desc in failures:
        add_rect(sld, Inches(6.6), ty, Inches(6.5), Inches(1.05),
                 RGBColor.from_string(col[1:]))
        add_textbox(sld, Inches(6.7), ty + Inches(0.04),
                    Inches(3.5), Inches(0.4),
                    bucket, font_size=13, bold=True, color=WHITE)
        add_textbox(sld, Inches(10.7), ty + Inches(0.04),
                    Inches(1.2), Inches(0.4),
                    pct, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)
        add_textbox(sld, Inches(6.7), ty + Inches(0.48),
                    Inches(6.2), Inches(0.5),
                    desc, font_size=11, color=WHITE)
        ty += Inches(1.15)


def slide_16_conclusion(prs):
    sld = blank_slide(prs)
    fill_bg(sld)
    add_header(sld, 18)

    add_textbox(sld, Inches(0.4), Inches(0.2),
                Inches(9.0), Inches(0.55),
                "Conclusion & Future Work", font_size=28, bold=True, color=WHITE)
    add_rect(sld, Inches(0.4), Inches(0.82), Inches(4.5), Inches(0.04), ACCENT)

    # left — conclusion
    add_rect(sld, Inches(0.4), Inches(1.05), Inches(6.1), Inches(5.6), DARK)
    add_textbox(sld, Inches(0.55), Inches(1.12),
                Inches(5.8), Inches(0.5),
                "Conclusions", font_size=18, bold=True, color=ACCENT)

    conclusions = [
        "Graph retrieval solves lookup — 80% accuracy vs 20% Naive RAG",
        "Retrieval alone cannot handle trend averaging or multi-hop joins",
        "Text-to-Pandas achieves 100% — symbolic computation is the bottleneck",
        "58-pp gap to oracle is a computation gap, not a retrieval gap",
        "Hybrid architecture (graph + pandas) is the recommended path forward",
    ]
    add_bullet_box(sld, Inches(0.55), Inches(1.7),
                   Inches(5.8), Inches(4.5),
                   conclusions, font_size=14, color=WHITE)

    # right — future work
    add_rect(sld, Inches(6.8), Inches(1.05), Inches(6.2), Inches(5.6),
             RGBColor(0x1e, 0x33, 0x5e))
    add_textbox(sld, Inches(6.95), Inches(1.12),
                Inches(5.9), Inches(0.5),
                "Future Work", font_size=18, bold=True, color=ACCENT)

    future = [
        "Fix MISSING_EDGE: pass all obs with wave mean → trend ~80%",
        "Two-stage multi-hop: graph resolves diagnosis date, pandas computes pre/post avg",
        "LLM-as-judge evaluator for format-sensitive answers",
        "Finer-grained temporal index (monthly vs 5-year waves)",
        "Validate on real EHR data (MIMIC-IV, eICU)",
    ]
    add_bullet_box(sld, Inches(6.95), Inches(1.7),
                   Inches(5.9), Inches(4.5),
                   future, font_size=14, color=WHITE)

    # bottom bar
    add_rect(sld, 0, Inches(6.9), SLIDE_W, Inches(0.6), ACCENT)
    add_textbox(sld, Inches(0.4), Inches(6.93),
                Inches(12.5), Inches(0.45),
                "IIIT-Hyderabad · IASNLP · 2026   |   "
                "github.com/iasnlp-rag-project-v2   |   "
                "Synthea EHR · 1,171 patients · 100-Q benchmark",
                font_size=11, color=WHITE, align=PP_ALIGN.CENTER)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    out_path = os.path.join(os.path.dirname(__file__), "IASNLP_Presentation.pptx")

    prs = new_prs()

    print("Building slides...")
    slide_01_title(prs)               ; print("  [1/18] Title")
    slide_02_problem(prs)             ; print("  [2/18] Problem")
    slide_03_research_question(prs)   ; print("  [3/18] Research Question")
    slide_04_dataset(prs)             ; print("  [4/18] Dataset")
    slide_05_benchmark(prs)           ; print("  [5/18] Benchmark Design")
    slide_06_approaches(prs)          ; print("  [6/18] Approaches")
    slide_07_kg_architecture(prs)     ; print("  [7/18] KG Architecture")
    slide_08_results_table(prs)       ; print("  [8/18] Results Table (V1+V2)")
    slide_09_bar_chart(prs)           ; print("  [9/18] Bar Chart")
    slide_10_finding1(prs)            ; print("  [10/18] Finding 1")
    slide_11_finding2(prs)            ; print("  [11/18] Finding 2")
    slide_12_core_finding(prs)        ; print("  [12/18] Core Finding (+V2 banner)")
    slide_handler_progression(prs)    ; print("  [13/18] Handler Progression 26%→98%")
    slide_13_extended_benchmark(prs)  ; print("  [14/18] Extended Benchmark")
    slide_13_cohort_table(prs)        ; print("  [15/18] Cohort Table")
    slide_14_deep_dive(prs)           ; print("  [16/18] Patient Deep Dive")
    slide_15_failure_analysis(prs)    ; print("  [17/18] Failure Analysis")
    slide_16_conclusion(prs)          ; print("  [18/18] Conclusion")

    prs.save(out_path)
    print(f"\nSaved: {out_path}")
    size_kb = os.path.getsize(out_path) // 1024
    print(f"File size: {size_kb} KB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
