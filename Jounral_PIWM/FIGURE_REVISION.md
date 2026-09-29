# 图片与会议结果整合记录（2026-09-27）

当前主稿为 `elsarticle-template-num.tex`；编译后 52 页，11 张图。

## 本次修改

- 重新绘制 3 张 TikZ 方法图：架构、道路状态/曲率编码、道路预测。
- 删除重复的双道路动机示意图，统一标题、字号、箭头与留白。
- 架构图明确区分相机提供的曲率预览和评估时的 GT 初始状态。
- 道路状态统一为 `(s, e_CTE, e_psi)`，曲率剖面单列；预测图使用完整 Frenet 进度方程。
- 主结果图分为均值曲线与 100 步的跨折分布，图例在绘图区外；误差范围不再遮挡所有曲线。
- 标签噪声图拆成三个共享坐标的子图，每幅只保留一个误差带。
- 补回 CartPole、Lunar Lander、DonkeyCar simulator 三个原始环境画面。
- 补回三个环境的 18 个预测子图，参数估计图、定性 rollout，以及会议附录 36 格控制器结果。
- 定性对比保留所有模型行，显示 k=5、15、30，去掉重复中间列及幻灯片页脚/页码。
- 参数图去掉没有已知真值参考的 DonkeyCar wheelbase 面板。
- imgs/ 只保留当前正文实际引用的 8 个外部 PDF；另外 3 图为主稿内的 TikZ。
- 15 个未引用旧图已移到 `../_archive/journal/unused_figures/`；修改前主稿与 PDF 在 `../_archive/journal/figure_revision_before/`。

## 数据来源与边界

- 会议源文件：用户提供的 CPS26_PIWM.zip，解压保留在 `../_archive/conference_source/`。
- 预测曲线直接使用 `resultsx2.pdf`、`cartresults.pdf` 内嵌图像，未对曲线采点、拟合或创造数据。保留原纵轴范围（包括原图在 RMSE=2 上界的截断），重新排版坐标、图例与文字。
- 原始环境/定性画面来自 `pred.pdf` 和 `cartviz.pdf` 的 ground-truth / model rows，未进行生成式补图或修图。
- 参数区间、中心标记和真值线来自 `allpara.pdf`；保留图形与原刻度位置。
- 控制器表来自 `sample-sigconf.tex` 的 `tab:2`，36 个 mean/dispersion 单元格已逐格比对。
- 会议图未说明 shading / interval / ± 的统计定义，因此正文不擅自称之为 SD、SE 或置信区间。
- `allpara.pdf` 的 CartPole 面板原标注为“length of the cart”，而同一源码的动力学算法使用 m_c。这一原始语义不一致需要作者核对；本次保留原图名称，不把图形重新解释为另一物理量。
- 期刊新图只读取 `../reports/matrix/folds_table_symmetric_curves.npz`。数据、checkpoint、表格主结果未改变，没有运行新训练。
- 会议状态 RMSE、会议控制器指标，与真车的米制位置误差分开说明。

## 复现

从仓库 piwm/ 运行（Windows 先设置 PYTHONUTF8=1）：

1. `python scripts/prepare_conference_assets.py`（需 pypdf、pdfplumber、reportlab、numpy、matplotlib、Pillow；本机可用 Codex 的 bundled Python 执行此步骤）
2. `.venv/Scripts/python.exe scripts/fig_main_cv.py`
3. `.venv/Scripts/python.exe scripts/fig_delta_cv.py`
4. `.venv/Scripts/python.exe scripts/fig_conference.py`

曲线脚本直接更新 `Jounral_PIWM/imgs/`。TikZ 图在主稿中继续编辑。

## 验收

- pdflatex → bibtex → pdflatex ×2 完成。
- 无 LaTeX error、无 undefined references/citations、无 overfull hbox/vbox。
- 已检查全部 52 页的整体版面和所有图页的高分辨率渲染。
- 保留原有参考文献中的 7 条字段完整性警告（缺 pages / volume-number 冲突），未编造页码。
- 作者和 CRediT 占位、原有全文科学论证不属于本次图件修订范围；当前版本不是已补齐全部投稿信息的终稿。

会议 ZIP SHA256：40d9aba01e6c625faf04ebf7b1b026e429675c712a765be36534765712a77e38

期刊五折缓存 SHA256：7fbf9cd23fb8a485331940d4a51d78f6eda881c0c2f4db54a7cf76eb19089cda
