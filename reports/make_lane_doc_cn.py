"""Generate a Chinese-language PDF that explains the lane-prediction
mechanism (the 4 steps inside dynamics_lane_v6) in detail, including a
glossary of every symbol.

Output: reports/figures/lane_prediction_cn.pdf
"""

# --- auto-added: make repo root importable when run as a script ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end shim ---

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, grey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# -------- Chinese fonts (Windows) --------
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
# simsun.ttc is a TrueType collection — without subfontIndex some reportlab
# versions fall back to the wrong sub-font and substitute glyphs (e.g. 纲→网).
pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc",
                                subfontIndex=0))

# -------- output dir for the formula PNGs we render via mathtext --------
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
FORMULA_DIR = os.path.join(FIG_DIR, "_formulas_lane_cn")
os.makedirs(FORMULA_DIR, exist_ok=True)


# ===================================================================
#  helpers — render a single LaTeX formula to PNG via matplotlib
# ===================================================================
def render_formula(tex, out_path, fontsize=15, padding=0.12):
    fig = plt.figure(figsize=(0.01, 0.01))
    text = fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="black")
    fig.canvas.draw()
    bbox = text.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    w = bbox.width + 2 * padding
    h = bbox.height + 2 * padding
    fig.set_size_inches(w, h)
    text.set_position((padding / w, padding / h))
    fig.savefig(out_path, dpi=220, transparent=True, bbox_inches="tight",
                pad_inches=padding)
    plt.close(fig)


def F(tex, name, width_cm=14):
    """Render LaTeX → reportlab Image, sized to width_cm wide."""
    path = os.path.join(FORMULA_DIR, name + ".png")
    if not os.path.exists(path):
        render_formula(tex, path)
    from PIL import Image as PILImage
    img = PILImage.open(path)
    w_px, h_px = img.size
    target_w = width_cm * cm
    target_h = target_w * h_px / w_px
    return Image(path, width=target_w, height=target_h, hAlign="LEFT")


# ===================================================================
#  document styles
# ===================================================================
PRIMARY = HexColor("#1F3A5F")
ACCENT  = HexColor("#9C2A2A")
MUTED   = HexColor("#666666")

s_title = ParagraphStyle(
    "title", fontName="SimHei", fontSize=20, leading=26,
    textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=8,
)
s_subtitle = ParagraphStyle(
    "subtitle", fontName="SimSun", fontSize=11, leading=15,
    textColor=MUTED, alignment=TA_CENTER, spaceAfter=18,
)
s_h1 = ParagraphStyle(
    "h1", fontName="SimHei", fontSize=15, leading=22,
    textColor=PRIMARY, spaceBefore=14, spaceAfter=6,
)
s_h2 = ParagraphStyle(
    "h2", fontName="SimHei", fontSize=12.5, leading=18,
    textColor=PRIMARY, spaceBefore=10, spaceAfter=4,
)
s_body = ParagraphStyle(
    "body", fontName="SimSun", fontSize=10.5, leading=17,
    textColor=black, alignment=TA_JUSTIFY, spaceAfter=4,
    firstLineIndent=22,
)
s_body_no_indent = ParagraphStyle(
    "body0", fontName="SimSun", fontSize=10.5, leading=17,
    textColor=black, alignment=TA_JUSTIFY, spaceAfter=4,
)
s_quote = ParagraphStyle(
    "quote", fontName="SimSun", fontSize=10, leading=16,
    textColor=HexColor("#444"), leftIndent=18, rightIndent=10,
    spaceBefore=2, spaceAfter=6, borderColor=ACCENT,
    borderPadding=6, borderWidth=0, backColor=HexColor("#FAF1F1"),
)
s_caption = ParagraphStyle(
    "cap", fontName="SimSun", fontSize=9.5, leading=13,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10,
)
s_table_header = ParagraphStyle(
    "th", fontName="SimHei", fontSize=10, leading=14,
    textColor=PRIMARY, alignment=TA_CENTER,
)
s_table_cell = ParagraphStyle(
    "td", fontName="SimSun", fontSize=10, leading=14,
    textColor=black, alignment=TA_LEFT,
)
s_table_sym = ParagraphStyle(
    "sym", fontName="SimSun", fontSize=10.5, leading=14,
    textColor=PRIMARY, alignment=TA_CENTER,
)


# ===================================================================
#  build the document
# ===================================================================
out_path = os.path.join(FIG_DIR, "lane_prediction_cn.pdf")
doc = SimpleDocTemplate(
    out_path, pagesize=A4,
    leftMargin=2.0 * cm, rightMargin=2.0 * cm,
    topMargin=2.0 * cm, bottomMargin=2.0 * cm,
    title="PIWM 车道预测机制详解",
)
story = []

# ----- title -----
story.append(Paragraph("PIWM 车道(Lane)预测机制详解", s_title))
story.append(Paragraph(
    "Resample-based lane propagation —— "
    "对应代码 <b>models/dynamics_lane_v6.py</b>",
    s_subtitle))


# ===================================================================
#  Section 1 — overview
# ===================================================================
story.append(Paragraph("1. 整体目标", s_h1))
story.append(Paragraph(
    "Encoder 从图像里预测一个 29 维的 latent："
    "<font name='SimHei'>9 维车体状态 + 20 维车道</font>。"
    "20 维车道由 10 个 waypoint 组成,每个 waypoint 是 body 系下的二维点 "
    "(x_b, y_b)。"
    "第 i 个 waypoint 永远表示"
    "<font color='#9C2A2A'>“沿车道中心线、距车 s_i 处的点”</font>,"
    "其中 s_i ∈ {-5, 0, 5, ..., 40}(共 10 个,间距 Δs=5)。",
    s_body))
story.append(Paragraph(
    "Dynamics 块的 lane 分支负责回答一个问题：<b>给定 t 时刻的 20 维车道、"
    "车的物理状态、动作 a_t,如何输出 t+1 时刻的 20 维车道,并且保证 "
    "“第 i 个槽位仍然代表沿路 s_i 处”?</b>"
    "v6 的解法分四步,每步只做一件可解释的事情。",
    s_body))


# ===================================================================
#  Section 2 — symbol table
# ===================================================================
story.append(Paragraph("2. 符号表", s_h1))

sym_rows = [
    ("符号", "维度 / 单位", "含义"),
    ("z<sub>t</sub><sup>car</sup>", "11 (归一化)",
     "t 时刻车体物理状态:"
     "[x, y, ψ, v_x, v_y, ω, w_0, w_1, w_2, w_3, δ]"
     ",分别为世界系位置、yaw、世界系速度、yaw 角速度、四个轮速、转向角"),
    ("z<sub>t</sub><sup>lane</sup>", "20 (归一化)",
     "t 时刻车道 waypoint 拼接,即 "
     "(x_b<sup>0</sup>, y_b<sup>0</sup>, ..., x_b<sup>9</sup>, y_b<sup>9</sup>)"),
    ("a<sub>t</sub>", "3",
     "t 时刻动作 (steer, gas, brake)"),
    ("w<sub>i</sub>", "(x_b, y_b) ∈ R<super>2</super>",
     "第 i 个 waypoint(t 时刻 body 系下),i = 0..9"),
    ("s<sub>i</sub>", "弧长(单位)",
     "第 i 个槽位的目标弧长偏移,固定为 {-5,0,5,...,40}"),
    ("Δs", "5 (单位)",
     "相邻槽位的弧长间距 (LANE_S_SAMPLES 步长)"),
    ("Δt", "1/50 s",
     "仿真步长(50 FPS,DT in config.py)"),
    ("v_x, v_y", "世界系",
     "车在世界系下的纵/横向速度分量(由 z_t<sup>car</sup> 反归一化得到)"),
    ("ψ (psi)", "rad",
     "车的 yaw 角(z_t<sup>car</sup>[2])"),
    ("ω (omega)", "rad/s",
     "yaw 角速度(z_t<sup>car</sup>[5])"),
    ("c, s", "标量",
     "cos(ψ), sin(ψ),反复出现的简写"),
    ("Δx_b", "标量",
     "一步内车在 body 系下的<b>横向</b>位移分量"),
    ("Δy_b", "标量",
     "一步内车在 body 系下的<b>前进</b>位移分量"),
    ("Δψ", "标量",
     "一步内车的 yaw 增量 = ω·Δt"),
    ("R(θ)", "2x2 旋转矩阵",
     "标准平面旋转,R(θ) = [[cos θ, -sin θ],[sin θ, cos θ]]"),
    ("w'<sub>i</sub>", "(x_b, y_b) ∈ R<super>2</super>",
     "<b>Step 1 后</b>,在 t+1 时刻 body 系下表达的同一个世界点"),
    ("w̃<sub>i</sub>", "(x_b, y_b) ∈ R<super>2</super>",
     "<b>Step 2 后</b>,重采样得到的、对应槽位 i 的新 waypoint"),
    ("α (alpha)", "无量纲",
     "重采样插值系数 α = Δy_b / Δs"),
    ("MLP_res", "23→20",
     "lane 残差网络:输入 [z_{t+1}<sup>car</sup>, w̃ (归一化), a_t]"),
]

# wrap rows in Paragraphs
table_data = []
for i, row in enumerate(sym_rows):
    if i == 0:
        table_data.append([Paragraph(c, s_table_header) for c in row])
    else:
        table_data.append([
            Paragraph(row[0], s_table_sym),
            Paragraph(row[1], s_table_cell),
            Paragraph(row[2], s_table_cell),
        ])

t = Table(table_data, colWidths=[3.2 * cm, 3.4 * cm, 10.4 * cm], hAlign="LEFT")
t.setStyle(TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), HexColor("#EAF0FA")),
    ("BOX",         (0, 0), (-1, -1), 0.5, grey),
    ("INNERGRID",   (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
]))
story.append(t)


# ===================================================================
#  Section 3 — body-frame motion
# ===================================================================
story.append(PageBreak())
story.append(Paragraph("3. 预备:计算车的 body 系单步位移", s_h1))
story.append(Paragraph(
    "在做任何车道几何之前,先把车自己的运动从世界系翻译到 body 系。"
    "这部分代码在 dynamics_lane_v6.py 第 64–74 行:",
    s_body))
story.append(Spacer(1, 4))
story.append(F(r"\Delta x_b \;=\; (v_x \cos\psi + v_y \sin\psi)\,\Delta t",
               "dx_b", width_cm=12))
story.append(F(r"\Delta y_b \;=\; (-v_x \sin\psi + v_y \cos\psi)\,\Delta t",
               "dy_b", width_cm=12))
story.append(F(r"\Delta\psi \;=\; \omega\,\Delta t", "dpsi", width_cm=6))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>注意 PIWM 约定</b>:body+y 是前进方向,所以 Δy_b 是车<b>沿车道</b>"
    "走的距离;Δx_b 是横向漂移(漂移过弯时不为零);"
    "Δψ 是车自身转过的角度。三者一起完整描述"
    "“一个 Δt 后,新的 body 系相对于老 body 系的位姿”。",
    s_body))


# ===================================================================
#  Section 4 — Step 1
# ===================================================================
story.append(Paragraph("4. Step 1 · 刚体逆变换 (rigid-body inverse transform)",
                       s_h1))
story.append(Paragraph(
    "<b>这一步只换坐标系,不动世界点。</b>"
    "想象一下:车在 Δt 内向前挪了 Δy_b、向侧滑了 Δx_b、转了 Δψ;"
    "可是赛道还在那儿,中心线一点没动。"
    "于是同一个世界点在<b>新的</b> body 系下的坐标只是个老坐标系到新坐标系的变换:",
    s_body))
story.append(F(
    r"w'_i \;=\; R(-\Delta\psi)\,(\,w_i - (\Delta x_b,\,\Delta y_b)\,)",
    "rigid", width_cm=12))
story.append(Paragraph(
    "代码里把这一步拆开来写了(去归一化 → 平移 → 旋转,见第 76–86 行),"
    "效果等价于上式。",
    s_body))
story.append(Paragraph(
    "<b>问题:槽位语义已经漂移了。</b>"
    "Step 1 之后,第 i 个槽位的 y_b 坐标大约等于 s_i − Δy_b,"
    "也就是说原本代表“前方 20 米”的那个点,现在看起来在前方 18.5 米的位置。"
    "如果就地停下,这就是 v5 的做法 —— 100 步以后这些点全跑到车后面去了。",
    s_quote))


# ===================================================================
#  Section 5 — Step 2
# ===================================================================
story.append(Paragraph("5. Step 2 · 重采样到固定 s (RESAMPLE,核心贡献)", s_h1))
story.append(Paragraph(
    "<b>目标:</b>让槽位 i 的 y_b 重新等于 s_i。"
    "我们手上有 10 个 w'_i 已经在新 body 系里,它们的 y_b 大致在 "
    "s_i − Δy_b 这条阶梯上,均匀间隔 Δs。"
    "对每个槽位 i,目标点 (y_b = s_i) 恰好落在 w'_i 与 w'_{i+1} 之间。"
    "用线性插值把它捞出来:",
    s_body))
story.append(F(
    r"\widetilde w_i \;=\; (1-\alpha)\,w'_i \;+\; \alpha\,w'_{i+1}"
    r",\qquad \alpha = \frac{\Delta y_b}{\Delta s}",
    "resample", width_cm=12))
story.append(Paragraph("<b>为什么 α = Δy_b / Δs?</b>", s_h2))
story.append(Paragraph(
    "Step 1 之后,槽位 i 的 y_b 是 s_i − Δy_b,槽位 i+1 的 y_b 是 "
    "s_{i+1} − Δy_b = s_i + Δs − Δy_b。"
    "两点之间走的纵向距离是 Δs(还是这个间距,刚体变换不改变距离)。"
    "我们要从 (s_i − Δy_b) 走到目标 s_i,需要前进 Δy_b 这么远;"
    "α 就是“前进 Δy_b ÷ 总距 Δs”,所以 α = Δy_b / Δs。"
    "因为 Δy_b ≪ Δs(50 FPS,典型 Δy_b ≈ 0.5–1.5,而 Δs = 5),α 始终很小,"
    "插值落在两端之间,数值很稳。",
    s_body))


# ===================================================================
#  Section 6 — Step 3
# ===================================================================
story.append(Paragraph("6. Step 3 · 前端槽位的切线外推", s_h1))
story.append(Paragraph(
    "上面的插值需要槽位 i+1。最远那个槽位 i = N-1 = 9 没有 i+1,"
    "因为它已经是“前方 40”这个最远的采样点了。"
    "于是用最后两个 waypoint (w'_8, w'_9) 估出车道在该处的"
    "<b>切线</b>方向,然后沿切线再走 α 这么多:",
    s_body))
story.append(F(
    r"\widetilde w_9 \;=\; w'_9 \;+\; \alpha\,(w'_9 - w'_8)",
    "front_ext", width_cm=10))
story.append(Paragraph(
    "几何上,这等价于假设车道在末端一小段近似为直线,先沿着这条直线再前进 α·Δs。"
    "由于 α 通常 ≪ 1,外推距离很短(几厘米到几十厘米),"
    "误差完全可以忽略。",
    s_body))


# ===================================================================
#  Section 7 — Step 4
# ===================================================================
story.append(Paragraph("7. Step 4 · 0.1 倍 MLP 学习残差", s_h1))
story.append(Paragraph(
    "前三步是<b>纯几何</b>,基于两条假设:"
    "(1) 车道局部接近直线;"
    "(2) 车一步 Δy_b ≪ Δs。"
    "对真实赛道,这两条只是近似:中心线有曲率;过弯时车身漂移、视觉透视等"
    "都会让“纯几何”预测略微偏离真值。"
    "于是再加一个小的可学习修正:",
    s_body))
story.append(F(
    r"w_i^{\,t+1} \;=\; \widetilde w_i \;+\; 0.1 \cdot \mathrm{MLP}_{\mathrm{res}}"
    r"(\,z^{\mathrm{car}}_{t+1},\ \widetilde w_{0..9}^{\,\mathrm{norm}},"
    r"\ a_t\,)",
    "lane_res", width_cm=14))
story.append(Paragraph(
    "<b>0.1 这个系数很关键。</b>它强制 MLP 只能做小范围微调,"
    "不可能推翻几何步的结果。整套 dynamics 是“几何主导,网络补差”——"
    "这正是 PIWM “Physics-Informed” 名字的由来。",
    s_quote))


# ===================================================================
#  Section 8 — full numerical example
# ===================================================================
story.append(PageBreak())
story.append(Paragraph("8. 完整数值示例", s_h1))
story.append(Paragraph(
    "取一个具体一步 (Δt = 0.02 s):"
    "Δs = 5,Δy_b = 1.5,Δx_b = Δψ = 0(为了把视线集中到 y_b 上,"
    "省去横向位移与转动)。"
    "于是 α = 1.5 / 5 = 0.3。",
    s_body))

# Three-state table for slot positions
demo_rows = [["slot i", "s_i (= 目标 y_b)", "Step 1 后 y_b", "Step 2 后 y_b"]]
S = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
for i, s in enumerate(S):
    after_rigid = s - 1.5
    if i < 9:
        after_resamp = 0.7 * after_rigid + 0.3 * (s + 5 - 1.5)
    else:
        after_resamp = after_rigid + 0.3 * 5.0
    demo_rows.append([
        str(i),
        f"{s:+d}",
        f"{after_rigid:+.1f}",
        f"{after_resamp:+.1f}",
    ])
demo_data = [[Paragraph(c, s_table_header if r == 0 else s_table_cell)
              for c in row] for r, row in enumerate(demo_rows)]
demo_t = Table(demo_data, colWidths=[2.0*cm, 3.2*cm, 3.6*cm, 3.6*cm],
               hAlign="LEFT")
demo_t.setStyle(TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), HexColor("#EAF0FA")),
    ("BOX",         (0, 0), (-1, -1), 0.5, grey),
    ("INNERGRID",   (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
    ("ALIGN",       (1, 1), (-1, -1), "CENTER"),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("FONTNAME",    (0, 1), (-1, -1), "SimSun"),
    ("FONTSIZE",    (0, 1), (-1, -1), 10),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
]))
story.append(demo_t)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "可以看到,经过 Step 2 之后,“Step 2 后 y_b” 一栏与“目标 y_b”一栏完全一致。"
    "这就是 RESAMPLE 的魔法:无论你是 1 步还是 100 步之后,"
    "槽位 i 永远代表“沿路前方 s_i 处”,语义不漂移。",
    s_body))


# ===================================================================
#  Section 9 — visualizations
# ===================================================================
story.append(Paragraph("9. 可视化", s_h1))

# Lane resample figure
img_resample = os.path.join(FIG_DIR, "lane_resample.png")
if os.path.exists(img_resample):
    from PIL import Image as PILImage
    pim = PILImage.open(img_resample)
    w_px, h_px = pim.size
    target_w = 17.0 * cm
    target_h = target_w * h_px / w_px
    story.append(Image(img_resample, width=target_w, height=target_h,
                        hAlign="CENTER"))
    story.append(Paragraph(
        "图 1 · 重采样三步走。(a) t 时刻,每个槽位都已经在 y_b = s_i;"
        "(b) Step 1 后,所有点 y_b 整体下移 Δy_b,槽位语义漂移;"
        "(c) Step 2 后,经线性插值,槽位 i 重新回到 y_b = s_i。"
        "图中以具体数字 Δy_b = 1.5,α = 0.3 为例。",
        s_caption))

# Lane waypoints figure
img_waypoints = os.path.join(FIG_DIR, "lane_waypoints.png")
if os.path.exists(img_waypoints):
    from PIL import Image as PILImage
    pim = PILImage.open(img_waypoints)
    w_px, h_px = pim.size
    target_w = 17.0 * cm
    target_h = target_w * h_px / w_px
    story.append(Image(img_waypoints, width=target_w, height=target_h,
                        hAlign="CENTER"))
    story.append(Paragraph(
        "图 2 · waypoint 是怎么取的。每帧把车投影到中心线得到 s_car,"
        "在 s_car + s_i 处采样中心线 (d=0),再旋转-平移到当前 body 系。",
        s_caption))


# ===================================================================
#  Section 10 — assumptions
# ===================================================================
story.append(Paragraph("10. 假设、限制与公平性", s_h1))
story.append(Paragraph(
    "<b>(a) 几何步的假设。</b>"
    "Step 2 / Step 3 都是“相邻 waypoint 之间是直线”的近似。"
    "采样间距 Δs = 5 远小于赛道曲率半径,这个近似在 CarRacing 上误差很小;"
    "Step 4 的 MLP 残差专门负责修这个近似误差。",
    s_body_no_indent))
story.append(Paragraph(
    "<b>(b) Δy_b 必须远小于 Δs。</b>"
    "α = Δy_b / Δs 应当落在 (0, 1) 之间。"
    "50 FPS 下 Δt = 0.02 s,即使车以 100 单位/秒(已经飙得很快)前进,"
    "Δy_b ≈ 2.0,仍然有 α ≈ 0.4,远没到失稳。"
    "如果未来要把帧率降到 5 FPS,就需要在两个相邻采样点之间插更多 waypoint,"
    "或者一次跨多个槽位。",
    s_body_no_indent))
story.append(Paragraph(
    "<b>(c) 训练时 vs 推理时。</b>"
    "训练阶段会用 GT 轨道(road_poly)+ GT 车位姿离线计算 lane_wp_body,"
    "作为 encoder 的<b>监督标签</b>(类似分类问题中的 label)。"
    "推理时,网络<b>只看图像和动作</b>,不接触任何 GT 轨道:"
    "encoder 从 15 帧叠加里读出 20 维 lane,然后 dynamics 块按上述四步滚动。"
    "这与“GT 轨道作为输入特征”是两件完全不同的事。",
    s_body_no_indent))
story.append(Paragraph(
    "<b>(d) baseline 的公平性。</b>"
    "为了和 PIWM 比,DVBF / GOKU / V2P / SINDy-C 都被改造成同样吃 29 维 latent,"
    "用同样的 encoder、同样的 lane 监督;只在 dynamics 替换。"
    "见 baselines/shared_dynamics_lane.py 与 baselines/sindyc_lane.py。"
    "v6 的优势仅来自“如何把这 20 维往前滚”这个 dynamics 设计本身。",
    s_body_no_indent))


# ===================================================================
#  Section 11 — summary
# ===================================================================
story.append(Paragraph("11. 一句话总结", s_h1))
story.append(Paragraph(
    "<b>每一步 lane 预测 = 换坐标系(Step 1) + "
    "沿折线前进 α=Δy_b/Δs 的步长重采样(Step 2) + "
    "末端切线外推(Step 3) + 0.1× 神经网络微调(Step 4)。</b>"
    "前三步保证“槽位 i 永远代表沿路 s_i 处”这个语义不漂移,"
    "第四步把几何近似的误差补回来。",
    s_body_no_indent))


# ===================================================================
#  build
# ===================================================================
doc.build(story)
print(f"wrote {out_path}")
