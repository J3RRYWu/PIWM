from pathlib import Path
import re
p=Path('Jounral_PIWM/elsarticle-template-num.tex');s=p.read_text(encoding='utf-8')
s=s.replace('quantitative comparisons above.','quantitative comparisons above. The original controller diagnostic is also\nretained in Appendix~\\ref{sec:conf_control}.',1)
appendix=r'''
\section{Conference Controller Diagnostic}\label{sec:conf_control}
The conference study~\cite{piwm_conf} passed reconstructed observations through
the original fixed controller to assess task-relevant visual information.
Table~\ref{tab:conf_control} reproduces the archived results, separately from
trajectory-prediction error. The continuous intrinsic variant gives the best
controller agreement in these examples; the strongest prediction variant need
not have the strongest reconstruction-based control metric.

\begin{table}[htbp]
\centering
\caption{Conference controller results on reconstructed observations, reproduced
from~\cite{piwm_conf}. Entries retain the source's reported $\pm$ quantities;
the archived table does not define their dispersion statistic. Accuracy entries
are percentages. These results do not evaluate the new Frenet model.}
\label{tab:conf_control}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llccc@{}}
\toprule
Architecture & Latent & $\delta=0$ & $\delta=5\%$ & $\delta=10\%$ \\
\midrule
\multicolumn{5}{l}{\textit{DonkeyCar simulator: action RMSE $\downarrow$}}\\[2pt]
Intrinsic & Continuous & $0.12\pm0.04$ & $0.13\pm0.04$ & $0.15\pm0.05$\\
Intrinsic & Discrete   & $0.21\pm0.15$ & $0.29\pm0.16$ & $0.32\pm0.20$\\
Extrinsic & Continuous & $0.15\pm0.05$ & $0.16\pm0.05$ & $0.22\pm0.06$\\
Extrinsic & Discrete   & $0.15\pm0.04$ & $0.17\pm0.05$ & $0.19\pm0.05$\\
\midrule
\multicolumn{5}{l}{\textit{Lunar Lander: action accuracy (\%) $\uparrow$}}\\[2pt]
Intrinsic & Continuous & $93.0\pm1.8$ & $90.5\pm2.0$ & $87.1\pm2.2$\\
Intrinsic & Discrete   & $85.5\pm2.5$ & $82.1\pm2.8$ & $78.3\pm3.1$\\
Extrinsic & Continuous & $86.2\pm2.4$ & $83.5\pm2.6$ & $80.0\pm2.9$\\
Extrinsic & Discrete   & $91.5\pm2.1$ & $88.6\pm2.3$ & $84.5\pm2.5$\\
\midrule
\multicolumn{5}{l}{\textit{CartPole: action accuracy (\%) $\uparrow$}}\\[2pt]
Intrinsic & Continuous & $98.0\pm1.0$ & $96.5\pm1.2$ & $94.0\pm1.5$\\
Intrinsic & Discrete   & $95.0\pm1.6$ & $91.5\pm2.0$ & $87.2\pm2.5$\\
Extrinsic & Continuous & $95.5\pm1.5$ & $92.0\pm1.8$ & $88.0\pm2.2$\\
Extrinsic & Discrete   & $97.2\pm1.1$ & $95.0\pm1.4$ & $92.5\pm1.8$\\
\bottomrule
\end{tabular}
\end{table}

'''
s=s.replace(r'\bibliographystyle{elsarticle-num}',appendix+r'\bibliographystyle{elsarticle-num}')
p.write_text(s,encoding='utf-8')
p=Path('scripts/prepare_conference_assets.py');s=p.read_text(encoding='utf-8')
s=s.replace("pdfmetrics.registerFont(TTFont('CMR',str(ROOT/'.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/cmr10.ttf')))","""font=ROOT/'.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/cmr10.ttf'
if not font.exists():
    import matplotlib
    font=Path(matplotlib.get_data_path())/'fonts/ttf/cmr10.ttf'
pdfmetrics.registerFont(TTFont('CMR',str(font)))""")
# Original text is rebuilt outside the cropped panels. Remove it from source
# streams as well, so clipped slide titles cannot leak into extracted PDF text.
s=s.replace("c.save();page=PdfReader(buf).pages[0]", """c.save();page=PdfReader(buf).pages[0]
from pypdf.generic import ContentStream,NameObject
stream=ContentStream(source.get_contents(),source.pdf)
kept=[];in_text=False
for operands,op in stream.operations:
    if op==b'BT':in_text=True
    if not in_text:kept.append((operands,op))
    if op==b'ET':in_text=False
stream.operations=kept
source[NameObject('/Contents')]=stream
source['/Resources'].pop('/Font',None)""")
p.write_text(s,encoding='utf-8')
p=Path('scripts/fig_main_cv.py');s=p.read_text(encoding='utf-8');end=s.index('"""',3)+3
s='''"""Five-fold manuscript figure from the symmetric-selection evaluation cache.
Top: mean error trajectories. Bottom: individual fold endpoints, their mean
(diamond), and full min-max range. No rollouts or statistics are re-estimated.
Run from piwm/: .venv/Scripts/python.exe scripts/fig_main_cv.py
"""'''+s[end:];p.write_text(s,encoding='utf-8')
