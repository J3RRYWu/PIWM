## 2026-09-28 最新：曲率分支审计完成

此段优先于下面更早的进度。当前论文 40 页；完整模型仍是预先固定的性能参照。新增 27 个动力学训练，全部完成；累计新协议训练为 36 个 CNN、231 个动力学模型（不计历史开发模型），原始记录仍为同一赛道两条。

- 分支对照：`reports/curvature_branches/`；脚本 `scripts/run_curvature_branches.py`；新增模型 `src/baselines/branch_query.py`。主报告序列化的 NumPy 整数问题由独立 `scripts/summarize_curvature_branches.py` 适配，冻结源码与结果未改。
- 无重训诊断：`reports/road_branch_diagnosis/`（72 组训练／验证）及 `reports/road_branch_transfer/`（36 组测试）。两个来源测试均为动作范围外推；完整模型的精确道路替换仍未超过运动学的来源 E100。
- 时间 E100：full .359；response-off .378；pose-off .368；both-off .371。完整模型相对 both-off 约 3.3% 收益，不能把相对静态曲率的 9.9% 当成同一对照。新变体未在任一来源方向的平均 E100 超过运动学。
- 最新复审：`Jounral_PIWM/RAS_curvature_branch_review_2026-09-28.md`，Overall 3/5 Major Revision，尚非 weak accept。全部不利结果保留。
- 主稿更新由 `scripts/revise_branch_manuscript.py` 一次性执行；再次运行会拒绝覆盖。上一稿备份 `_archive/journal/curvature_branches_before/`。旧生成器可能回退当前稿，不要盲目重跑。
- 当前 PDF SHA256: `69491cbf6c9edaf08437abe33ed29fd805f7bfa7a6154508fad38614eb24a4f2`；40 页，编译及渲染核查通过。无新增训练进程、自动任务、提交或推送。几何后备方法训练和独立生成条件验证尚未做。
- 审阅后 commit；权重／数据等被忽略资产需单独备份。

# Current audited revision: 2026-09-28

The current PDF has 37 pages. Source: `elsarticle-template-num.tex`; generator:
`scripts/revise_corrected_manuscript.py` at repository root. It preserves the
immediate prior manuscript in `_archive/journal/corrected_query_before/` and
moves the six earlier controlled rounds to an appendix. Both generators
reconstruct the manuscript and overwrite manual edits; preserve author changes
before using either one.

The new corrected-label and query-control results are in
`reports/corrected_query/`; initialization sensitivity is in
`reports/corrected_initialization/`; corrected labels are separately versioned in
`reports/consistent_road_labels/`. The earlier `reproduce/` figure bundle does
not reproduce these suites. Training scripts and ignored checkpoints remain at
the repository root. The original raw data are one level above the repository.

Current build: no undefined references, duplicate labels or overfull boxes;
two BibTeX empty-page warnings remain (piwm_conf, karl2016deep). All 37 pages and
the key new tables were inspected. See
`reports/corrected_query/manuscript_render_manifest.json` for the PDF hash.

Author metadata, declarations and data/code release details are still unfinished.
The simulated review remains 3/5 Major Revision; a successful build does not imply
submission readiness or acceptance. See `RAS_revision_audit_2026-09-28.md`.
All prescribed computation is complete; no training or automation remains active.

---

# Build the revised PIWM manuscript

The main file is `elsarticle-template-num.tex`. External figures are in `imgs/`;
the retained architecture diagram is editable TikZ in the main file.
The audited revision currently builds to 37 pages. Author metadata and
declarations remain unfinished; compilation does not establish submission readiness.
Upload this directory to Overleaf and set that file as the main document, or run:

    pdflatex elsarticle-template-num.tex
    bibtex elsarticle-template-num
    pdflatex elsarticle-template-num.tex
    pdflatex elsarticle-template-num.tex

Use a current TeX installation with elsarticle, Latin Modern, TikZ, caption,
placeins and the other packages declared in the preamble. No external file paths
are required to compile the manuscript.

The `reproduce/` directory contains the plot generators, exact source assets and
the unchanged five-fold evaluation cache. To regenerate external figures:

    cd reproduce
    python -m pip install numpy matplotlib Pillow pypdf pdfplumber reportlab
    python scripts/prepare_conference_assets.py
    python scripts/fig_main_cv.py
    python scripts/fig_delta_cv.py
    python scripts/fig_conference.py

The scripts produce PDFs in `reproduce/Jounral_PIWM/imgs/`. Copy those PDFs back
to the top-level `imgs/` when updating the manuscript. A TeX installation on PATH
is used for plotting; the scripts also look for the usual user MiKTeX installation
on Windows. Preparation needs no training checkpoints or new model evaluations.

See FIGURE_REVISION.md for the exact sources, statistical limits of the inherited
conference graphics, archived files and the remaining author-information fields.

## Audited experiment update (2026-09-27)

The `reproduce/` bundle describes the earlier figure revision only. It does not
reproduce the six new experimental suites. Their protocols, code hashes, results
and audits are under the repository-level `reports/`; training scripts are under
`scripts/` and saved models under `checkpoints/`. See `reports/EXPERIMENT_LEDGER.md`.
New plots: `scripts/plot_conditioned_dynamics.py` and
`scripts/summarize_temporal_evidence.py`.

`scripts/revise_audited_manuscript.py` rebuilds the manuscript from the preserved
original and audited result files. It overwrites the current main TeX file; do
not run it after manual author edits unless those edits are first preserved.

The final build has no undefined references or overfull warnings. Five existing
BibTeX entries lack page fields; author details and disclosures still need review.
