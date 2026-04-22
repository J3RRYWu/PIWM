"""Generate slim PDF reports (EN + CN) with LaTeX-rendered formulas.

Only two sections:
  1. Latent definitions (31-dim state breakdown)
  2. Dynamics implementation (car + lane with proper math)

Formulas are rendered by matplotlib's mathtext (no external LaTeX needed).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, grey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

os.makedirs("vis/formulas", exist_ok=True)


# ===================================================================
# LaTeX rendering via matplotlib mathtext
# ===================================================================
def render_formula(tex, out_path, fontsize=14, padding=0.12):
    """Render a single LaTeX math string to PNG via matplotlib mathtext."""
    fig = plt.figure(figsize=(0.01, 0.01))  # will auto-resize
    text = fig.text(0, 0, f"${tex}$", fontsize=fontsize,
                    color='black')
    fig.canvas.draw()
    bbox = text.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    w = bbox.width + 2*padding
    h = bbox.height + 2*padding
    fig.set_size_inches(w, h)
    # Re-place with proper margin so text is fully inside
    text.set_position((padding / w, padding / h))
    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches='tight',
                pad_inches=padding)
    plt.close(fig)


def tex_img(tex, name, width_cm=12):
    """Render LaTeX and return a reportlab Image sized to width_cm (cm)."""
    path = f"vis/formulas/{name}.png"
    if not os.path.exists(path):
        render_formula(tex, path)
    from PIL import Image as PILImage
    img = PILImage.open(path)
    w_px, h_px = img.size
    target_w = width_cm * cm
    target_h = target_w * h_px / w_px
    return Image(path, width=target_w, height=target_h, hAlign='LEFT')


# ===================================================================
# Pre-render all formulas once
# ===================================================================
FORMULAS = {
    # Car dynamics (v4)
    "slip_f":   r"\alpha_f = \delta - \arctan\!\left(\frac{v_y^{\,b} + L_f\,\omega}{v_x^{\,b}}\right)",
    "slip_r":   r"\alpha_r = -\arctan\!\left(\frac{v_y^{\,b} - L_r\,\omega}{v_x^{\,b}}\right)",
    "tire":     r"F_{yf} = -C_f\,\alpha_f,\qquad F_{yr} = -C_r\,\alpha_r",
    "newton_x": r"\dot{v}_x^{\,b} \;=\; a_{\mathrm{fwd}} \;-\; \frac{F_{yf}\sin\delta}{m} \;+\; v_y^{\,b}\,\omega",
    "newton_y": r"\dot{v}_y^{\,b} \;=\; \frac{F_{yf}\cos\delta + F_{yr}}{m} \;-\; v_x^{\,b}\,\omega",
    "newton_w": r"\dot{\omega} \;=\; \frac{L_f F_{yf}\cos\delta \;-\; L_r F_{yr}}{I_z}",
    "euler":    r"v_{x,\mathrm{next}}^{\,b} = v_x^{\,b} + \dot{v}_x^{\,b}\,\Delta t,\qquad \omega_{\mathrm{next}} = \omega + \dot{\omega}\,\Delta t",
    "pos_int":  r"x_{t+1} = x_t + v_x\,\Delta t,\qquad \psi_{t+1} = \psi_t + \omega\,\Delta t",
    "body2world": r"v_{x,\mathrm{next}} = v_{x,\mathrm{next}}^{\,b}\cos\psi - v_{y,\mathrm{next}}^{\,b}\sin\psi,\qquad v_{y,\mathrm{next}} = v_{x,\mathrm{next}}^{\,b}\sin\psi + v_{y,\mathrm{next}}^{\,b}\cos\psi",
    "world2body": r"v_x^{\,b} = v_x\cos\psi + v_y\sin\psi,\qquad v_y^{\,b} = -v_x\sin\psi + v_y\cos\psi",
    # Lane dynamics
    "lane_trans": r"\mathrm{wp}_{\mathrm{shift}} = \mathrm{wp} - (\Delta x_b,\,\Delta y_b)",
    "lane_rot":   r"\mathrm{wp}_{\mathrm{rigid}} = R(-\Delta\psi)\,\mathrm{wp}_{\mathrm{shift}}",
    "resample":   r"\mathrm{wp}_{\mathrm{new}}[i] \;=\; (1-\alpha)\,\mathrm{wp}_{\mathrm{rigid}}[i] + \alpha\,\mathrm{wp}_{\mathrm{rigid}}[i+1],\qquad \alpha = \Delta y_b / \Delta s",
    "front_ext":  r"\mathrm{wp}_{\mathrm{new}}[9] \;=\; \mathrm{wp}_{\mathrm{rigid}}[9] + \alpha\,(\mathrm{wp}_{\mathrm{rigid}}[9] - \mathrm{wp}_{\mathrm{rigid}}[8])",
    "residual":   r"z_{t+1} \;=\; z_{\mathrm{phys}} + r,\qquad r[0] = r[1] = 0 \;\;\mathrm{(mask\;on\;}x, y\mathrm{)}",
    # Param bound
    "sigmoid":    r"\theta = \theta_{\min} + (\theta_{\max} - \theta_{\min})\,\sigma(\theta_{\mathrm{raw}})",
    # Car->Body motion increments
    "body_motion": r"\Delta x_b = (v_x\cos\psi + v_y\sin\psi)\,\Delta t,\qquad \Delta y_b = (-v_x\sin\psi + v_y\cos\psi)\,\Delta t,\qquad \Delta\psi = \omega\,\Delta t",
    "dspace":     r"\Delta s = 5,\qquad \mathrm{target} = [-5, 0, 5, 10, \dots, 40]",
}

for k, v in FORMULAS.items():
    render_formula(v, f"vis/formulas/{k}.png")
print("All formulas rendered.")


# ===================================================================
# PDF builder
# ===================================================================
def make_styles(lang='en'):
    body_font  = 'Helvetica' if lang == 'en' else 'SimSun'
    bold_font  = 'Helvetica-Bold' if lang == 'en' else 'SimHei'
    s = {}
    s['title']  = ParagraphStyle('title', fontName=bold_font, fontSize=20,
                                  leading=24, alignment=TA_CENTER,
                                  textColor=HexColor('#0a3a6e'), spaceAfter=14)
    s['h1']     = ParagraphStyle('h1', fontName=bold_font, fontSize=15, leading=19,
                                  textColor=HexColor('#0a3a6e'),
                                  spaceBefore=10, spaceAfter=6)
    s['h2']     = ParagraphStyle('h2', fontName=bold_font, fontSize=12, leading=15,
                                  textColor=HexColor('#333'),
                                  spaceBefore=8, spaceAfter=4)
    s['body']   = ParagraphStyle('body', fontName=body_font, fontSize=10.5,
                                  leading=15, alignment=TA_JUSTIFY, spaceAfter=4)
    s['caption']= ParagraphStyle('cap', fontName=body_font, fontSize=9, leading=11,
                                  alignment=TA_LEFT, textColor=HexColor('#444'),
                                  spaceAfter=10, italic=True)
    return s, body_font, bold_font


def build_en(story, s, body_font, bold_font):
    story.append(Paragraph("PIWM-lane-v6: Latent &amp; Dynamics", s['title']))

    # ---- 1. Latent ----
    story.append(Paragraph("1. Latent Space (31 dim)", s['h1']))
    story.append(Paragraph(
        "State <b>z &isin; R<super>31</super></b> = [car part (11) | lane part (20)].",
        s['body']))

    t = Table([
        [Paragraph("<b>Index</b>", s['body']),
         Paragraph("<b>Symbol</b>", s['body']),
         Paragraph("<b>Description</b>", s['body'])],
        ["z[0:2]",  "x, y",         "Position in body frame of ref (t=0)"],
        ["z[2]",    "ψ",            "Yaw"],
        ["z[3:5]",  "v_x, v_y",     "World-frame velocity"],
        ["z[5]",    "ω",            "Angular velocity (yaw rate)"],
        ["z[6:10]", "w_0..w_3",     "Four wheel angular speeds"],
        ["z[10]",   "δ",            "Steering angle"],
        ["z[11:31]", "10 × (x_b, y_b)", "10 lane waypoints at s ∈ [-5, 0, 5, …, 40]"],
    ], colWidths=[2.5*cm, 3.0*cm, 10.0*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Body-frame convention (CarRacing): <b>body +y</b> is forward, "
        "matching world forward direction <i>(−sin ψ, cos ψ)</i>.",
        s['body']))
    story.append(Paragraph(
        "All latent components stored <b>normalized</b> (subtract mean, divide by std); "
        "statistics computed from ~19k training frames.",
        s['body']))

    # ---- 2. Dynamics ----
    story.append(Paragraph("2. Dynamics: one-step update <i>z<sub>t</sub> → z<sub>t+1</sub></i>", s['h1']))

    story.append(Paragraph("2.1 Car dynamics: dynamic bicycle model", s['h2']))
    story.append(Paragraph(
        "World → body frame projection:",
        s['body']))
    story.append(tex_img(FORMULAS["world2body"], "world2body", width_cm=14))
    story.append(Paragraph(
        "Slip angles of front / rear tires:",
        s['body']))
    story.append(tex_img(FORMULAS["slip_f"], "slip_f", width_cm=10))
    story.append(tex_img(FORMULAS["slip_r"], "slip_r", width_cm=10))
    story.append(Paragraph(
        "Linear tire force model:",
        s['body']))
    story.append(tex_img(FORMULAS["tire"], "tire", width_cm=10))
    story.append(Paragraph(
        "Newton's 2<super>nd</super> law in body frame "
        "(with <i>a<sub>fwd</sub></i> from a learned <tt>accel_net</tt>):",
        s['body']))
    story.append(tex_img(FORMULAS["newton_x"], "newton_x", width_cm=12))
    story.append(tex_img(FORMULAS["newton_y"], "newton_y", width_cm=12))
    story.append(tex_img(FORMULAS["newton_w"], "newton_w", width_cm=10))
    story.append(Paragraph("Euler integration (Δt = 1/50 s):", s['body']))
    story.append(tex_img(FORMULAS["euler"], "euler", width_cm=15))
    story.append(Paragraph("Rotate velocity back to world frame:", s['body']))
    story.append(tex_img(FORMULAS["body2world"], "body2world", width_cm=16))
    story.append(Paragraph("Position and yaw integration:", s['body']))
    story.append(tex_img(FORMULAS["pos_int"], "pos_int", width_cm=12))
    story.append(Paragraph(
        "Six physical parameters are learnable, bounded by sigmoid reparameterization:",
        s['body']))
    story.append(tex_img(FORMULAS["sigmoid"], "sigmoid", width_cm=10))

    t = Table([
        ["Param", "Range", "Learned value"],
        ["m",     "[500, 2000] kg",   "1680"],
        ["I_z",   "[500, 5000] kg·m²", "4338"],
        ["C_f",   "[5e4, 5e5] N/rad", "237 k"],
        ["C_r",   "[5e4, 5e5] N/rad", "233 k"],
        ["L_f",   "[1, 4] m",          "2.28"],
        ["L_r",   "[1, 4] m",          "2.22"],
    ], colWidths=[3*cm, 5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "A learned residual MLP corrects the prediction but is <b>masked on x, y</b>:",
        s['body']))
    story.append(tex_img(FORMULAS["residual"], "residual", width_cm=13))
    story.append(Paragraph(
        "This keeps the kinematic position integration as a hard physical constraint.",
        s['caption']))

    story.append(PageBreak())

    story.append(Paragraph("2.2 Lane dynamics: rigid transform + resample", s['h2']))
    story.append(Paragraph(
        "Given the updated car state, extract body-frame motion over this step:",
        s['body']))
    story.append(tex_img(FORMULAS["body_motion"], "body_motion", width_cm=17))

    story.append(Paragraph(
        "<b>Step 1 &mdash; Rigid-body transform.</b> Translate and rotate all 10 waypoints into "
        "the new body frame:",
        s['body']))
    story.append(tex_img(FORMULAS["lane_trans"], "lane_trans", width_cm=9))
    story.append(tex_img(FORMULAS["lane_rot"], "lane_rot", width_cm=8))

    story.append(Paragraph(
        "<b>Step 2 &mdash; Arc-length resample.</b> After the rigid transform, each slot's arc-length "
        "has decreased by Δy_b. We restore the fixed-s semantics via linear interpolation "
        "between adjacent slots:",
        s['body']))
    story.append(tex_img(FORMULAS["resample"], "resample", width_cm=14))
    story.append(Paragraph(
        "Sampling interval Δs = 5 ; target offsets = [−5, 0, 5, 10, …, 40].",
        s['caption']))

    story.append(Paragraph(
        "<b>Step 3 &mdash; Tangent extrapolation for the front-most slot</b> (no slot i+1 available):",
        s['body']))
    story.append(tex_img(FORMULAS["front_ext"], "front_ext", width_cm=13))

    story.append(Paragraph(
        "<b>Step 4 &mdash; Small learned residual.</b> A <tt>lane_residual</tt> MLP "
        "with output scale 0.1 applies final corrections. "
        "<b>Steps 1-3 use zero learnable parameters</b>; the analytical backbone "
        "is pure geometry.",
        s['body']))

    # Summary
    story.append(Paragraph("Summary of components", s['h2']))
    t = Table([
        ["Component", "Type", "Params"],
        ["World↔body frame",            "Analytical",     "0"],
        ["Slip angles",                  "Analytical",     "0"],
        ["Tire forces (linear model)",   "Analytical",     "0"],
        ["Newton's 2nd law",             "Analytical",     "0"],
        ["Euler integration",            "Analytical",     "0"],
        ["Lane rigid transform + resample + tangent extrap.", "Analytical", "0"],
        ["Physical params (m, I_z, C_f, C_r, L_f, L_r)",      "Bounded sigmoid", "6"],
        ["accel_net (a_fwd)",            "MLP", "~5k"],
        ["wheel_net (Δw, Δδ)",           "MLP", "~5k"],
        ["residual_net (car, masked)",   "MLP", "~6k"],
        ["lane_residual (scale 0.1)",    "MLP", "~8k"],
    ], colWidths=[9*cm, 4*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


def build_cn(story, s, body_font, bold_font):
    story.append(Paragraph("PIWM-lane-v6 潜变量与动力学实现", s['title']))

    # 1. Latent
    story.append(Paragraph("1. 潜变量 (31 维)", s['h1']))
    story.append(Paragraph(
        "状态 <b>z &isin; R<super>31</super></b> = [车身部分 (11) | 车道部分 (20)]。",
        s['body']))

    t = Table([
        [Paragraph("<b>索引</b>", s['body']),
         Paragraph("<b>符号</b>", s['body']),
         Paragraph("<b>物理含义</b>", s['body'])],
        ["z[0:2]",  "x, y",            "参考帧 (t=0) 车身坐标系下的位置"],
        ["z[2]",    "ψ",               "Yaw (朝向)"],
        ["z[3:5]",  "v_x, v_y",        "世界系速度"],
        ["z[5]",    "ω",               "角速度 (yaw rate)"],
        ["z[6:10]", "w_0..w_3",        "四个轮子的角速度"],
        ["z[10]",   "δ",               "转向角"],
        ["z[11:31]", "10 × (x_b, y_b)", "10 个车道 waypoint, s ∈ [-5, 0, 5, …, 40]"],
    ], colWidths=[2.5*cm, 3.0*cm, 10.0*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "车身坐标系约定 (CarRacing)：<b>body +y 为前进方向</b>，"
        "对应世界系前进方向 <i>(−sin ψ, cos ψ)</i>。",
        s['body']))
    story.append(Paragraph(
        "所有潜变量以<b>归一化</b>形式存储 (减均值、除标准差)，"
        "统计量由约 19k 训练帧得到。",
        s['body']))

    # 2. Dynamics
    story.append(Paragraph("2. 动力学：单步更新 <i>z<sub>t</sub> → z<sub>t+1</sub></i>", s['h1']))

    story.append(Paragraph("2.1 车身动力学：动力学自行车模型", s['h2']))
    story.append(Paragraph("世界系 → 车身系速度投影：", s['body']))
    story.append(tex_img(FORMULAS["world2body"], "world2body", width_cm=14))
    story.append(Paragraph("前/后轮侧偏角：", s['body']))
    story.append(tex_img(FORMULAS["slip_f"], "slip_f", width_cm=10))
    story.append(tex_img(FORMULAS["slip_r"], "slip_r", width_cm=10))
    story.append(Paragraph("线性轮胎力模型：", s['body']))
    story.append(tex_img(FORMULAS["tire"], "tire", width_cm=10))
    story.append(Paragraph(
        "车身系牛顿第二定律 (其中 <i>a<sub>fwd</sub></i> 来自学习的 <tt>accel_net</tt>)：",
        s['body']))
    story.append(tex_img(FORMULAS["newton_x"], "newton_x", width_cm=12))
    story.append(tex_img(FORMULAS["newton_y"], "newton_y", width_cm=12))
    story.append(tex_img(FORMULAS["newton_w"], "newton_w", width_cm=10))
    story.append(Paragraph("Euler 积分 (Δt = 1/50 s)：", s['body']))
    story.append(tex_img(FORMULAS["euler"], "euler", width_cm=15))
    story.append(Paragraph("速度旋回世界系：", s['body']))
    story.append(tex_img(FORMULAS["body2world"], "body2world", width_cm=16))
    story.append(Paragraph("位置与朝向积分：", s['body']))
    story.append(tex_img(FORMULAS["pos_int"], "pos_int", width_cm=12))
    story.append(Paragraph(
        "六个物理参数用 sigmoid 重参数化做边界学习：",
        s['body']))
    story.append(tex_img(FORMULAS["sigmoid"], "sigmoid", width_cm=10))

    t = Table([
        ["参数", "范围", "学到的值"],
        ["m",     "[500, 2000] kg",    "1680"],
        ["I_z",   "[500, 5000] kg·m²", "4338"],
        ["C_f",   "[5e4, 5e5] N/rad",  "237 k"],
        ["C_r",   "[5e4, 5e5] N/rad",  "233 k"],
        ["L_f",   "[1, 4] m",          "2.28"],
        ["L_r",   "[1, 4] m",          "2.22"],
    ], colWidths=[3*cm, 5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "一个残差 MLP 对预测做修正，但 <b>x, y 维被掩码</b>：",
        s['body']))
    story.append(tex_img(FORMULAS["residual"], "residual", width_cm=13))
    story.append(Paragraph(
        "使运动学位置积分保持为硬物理约束。",
        s['caption']))

    story.append(PageBreak())

    story.append(Paragraph("2.2 车道动力学：刚体变换 + 重采样", s['h2']))
    story.append(Paragraph(
        "由更新后的车身状态，算出这一步的车身系位移：",
        s['body']))
    story.append(tex_img(FORMULAS["body_motion"], "body_motion", width_cm=17))

    story.append(Paragraph(
        "<b>第 1 步 &mdash; 刚体变换。</b> 对 10 个 waypoint 做平移与旋转到新车身系：",
        s['body']))
    story.append(tex_img(FORMULAS["lane_trans"], "lane_trans", width_cm=9))
    story.append(tex_img(FORMULAS["lane_rot"], "lane_rot", width_cm=8))

    story.append(Paragraph(
        "<b>第 2 步 &mdash; 弧长重采样。</b> 刚体变换后每个 slot 的弧长减少了 Δy_b。"
        "用相邻 slot 间的线性插值恢复 fixed-s 语义：",
        s['body']))
    story.append(tex_img(FORMULAS["resample"], "resample", width_cm=14))
    story.append(Paragraph(
        "采样间隔 Δs = 5；目标偏移 = [−5, 0, 5, 10, …, 40]。",
        s['caption']))

    story.append(Paragraph(
        "<b>第 3 步 &mdash; 最前方 slot 切线外推</b> (无 slot i+1 可用)：",
        s['body']))
    story.append(tex_img(FORMULAS["front_ext"], "front_ext", width_cm=13))

    story.append(Paragraph(
        "<b>第 4 步 &mdash; 小残差修正。</b> 一个 <tt>lane_residual</tt> MLP "
        "输出乘以 0.1 做最终修正。<b>第 1-3 步零可学参数</b>，"
        "解析骨架完全由几何决定。",
        s['body']))

    # Summary
    story.append(Paragraph("组件总览", s['h2']))
    t = Table([
        ["组件", "类型", "参数量"],
        ["世界系 ↔ 车身系",                 "解析",         "0"],
        ["侧偏角",                         "解析",         "0"],
        ["轮胎力 (线性模型)",                "解析",         "0"],
        ["牛顿第二定律",                    "解析",         "0"],
        ["Euler 积分",                     "解析",         "0"],
        ["车道刚体变换 + 重采样 + 切线外推",  "解析",         "0"],
        ["物理参数 (m, I_z, C_f, C_r, L_f, L_r)", "带界 sigmoid", "6"],
        ["accel_net (a_fwd)",             "MLP",          "~5k"],
        ["wheel_net (Δw, Δδ)",            "MLP",          "~5k"],
        ["residual_net (车身, 掩码)",       "MLP",          "~6k"],
        ["lane_residual (scale 0.1)",     "MLP",          "~8k"],
    ], colWidths=[9*cm, 4*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d5e3f3')),
        ('GRID', (0, 0), (-1, -1), 0.5, grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), body_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


def build_pdf(lang, out_path):
    s, body_font, bold_font = make_styles(lang)
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        title="PIWM-v6 Latent & Dynamics" if lang == 'en' else "PIWM-v6 潜变量与动力学",
    )
    story = []
    if lang == 'en':
        build_en(story, s, body_font, bold_font)
    else:
        build_cn(story, s, body_font, bold_font)
    doc.build(story)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    build_pdf('en', 'report_v2_en.pdf')
    build_pdf('cn', 'report_v2_cn.pdf')
