from pathlib import Path
import re
p=Path('Jounral_PIWM/elsarticle-template-num.tex');s=p.read_text(encoding='utf-8')
def fig(label,new):
    global s
    pattern=r'\\begin\{figure\}\[t\](?:(?!\\begin\{figure\}).)*?\\label\{'+re.escape(label)+r'\}.*?\\end\{figure\}'
    s,n=re.subn(pattern,lambda _:new.strip(),s,flags=re.S)
    assert n==1,(label,n)
# Remove a redundant conceptual cartoon; its geometry is covered by fig:encoding.
fig('fig:partial_obs','')
s=s.replace(r'Figure~\ref{fig:partial_obs} illustrates this definition.', 'The road-relative geometry is illustrated in Figure~\\ref{fig:encoding}.')
fig('fig:arch',r'''
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{9}{11}\selectfont,
 box/.style={draw=black!40,rounded corners=2pt,line width=.5pt,align=center,inner sep=5pt},
 ar/.style={-{Stealth[length=1.8mm]},line width=.7pt,draw=black!65},
 lab/.style={font=\fontsize{8}{10}\selectfont,fill=white,inner sep=2pt}]
\node[anchor=west,font=\bfseries\fontsize{9}{11}\selectfont] at (0,4.05) {(a) Observe once at $t_0$};
\node[box,fill=black!3,text width=2.35cm,minimum height=1cm] (cam) at (1.4,3.15)
 {Camera history\\$y_{t_0-T+1:t_0}$};
\node[box,fill=blue!6,text width=2.4cm,minimum height=1cm] (enc) at (5.5,3.15)
 {Road encoder $\mathcal E_p$\\15-frame context};
\node[box,fill=frenetink!7,text width=3cm,minimum height=1cm] (preview) at (10.4,3.15)
 {Fixed curvature preview\\$\hat\kappa(\Delta),\quad 0\leq\Delta\leq4.5$ m};
\draw[ar] (cam) -- (enc);
\draw[ar] (enc) -- (preview);
\node[lab,anchor=south] at (5.5,4.13) {Privileged road labels (training only)};
\draw[ar,dashed] (5.5,4.1) -- (enc.north);
\node[anchor=west,font=\bfseries\fontsize{9}{11}\selectfont] at (0,1.65) {(b) Predict at every step};
\node[box,fill=orange!8,text width=2.8cm,minimum height=1.25cm] (state) at (1.7,.4)
 {$z_t^*=(v,\omega,s,d,e_\psi)$\\Ground-truth state at $t_0$\\for the reported evaluation};
\node[box,fill=frenetink!7,text width=3.15cm,minimum height=1.25cm] (dyn) at (7.0,.4)
 {Structured transition\\Vehicle: learned actuation\\Road: Frenet geometry};
\node[box,fill=orange!8,text width=1.8cm,minimum height=1.25cm] (next) at (11.6,.4)
 {Next state\\$\hat z^*_{t+1}$};
\draw[ar] (state) -- (dyn);
\draw[ar] (dyn) -- (next);
\draw[ar] (preview.south) -- (10.4,1.85) -- node[lab,above] {$\kappa_t=\hat\kappa(s_t-s_{t_0})$} (7,1.85) -- (dyn.north);
\node[lab] at (4.35,1.65) {Action $a_t$};
\draw[ar] (4.35,1.4) |- ([yshift=3mm]dyn.west);
\draw[ar,dashed] (next.south) -- (11.6,-1.05) -- node[lab,below] {Autoregressive state update; no new images after $t_0$} (1.7,-1.05) -- (state.south);
\end{tikzpicture}
\caption{PIWM-Frenet computation used in the driving evaluation. (a) The camera
history supplies a curvature preview once per rollout; road labels supervise
this encoder only during training. (b) The current state, action and indexed
curvature drive the structured transition. Predictions are fed back, while the
preview remains fixed. The reported rollouts use ground-truth initialization;
the figure separates this state input from the camera-derived road preview.}
\label{fig:arch}
\end{figure}
''')
fig('fig:encoding',r'''
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{9}{11}\selectfont,
 ar/.style={-{Stealth[length=1.8mm]},draw=black!65,line width=.6pt},
 lab/.style={font=\fontsize{8}{10}\selectfont,fill=white,inner sep=1pt}]
\node[anchor=west] at (0,3.35) {\textbf{(a)} Road-relative state};
\node[anchor=west] at (7.4,3.35) {\textbf{(b)} Perceived preview (schematic)};
\draw[line width=9mm,black!8] (.25,.9) .. controls (2.6,.9) and (3.9,1.3) .. (6.15,2.3);
\draw[black!45,dashed] (.25,.9) .. controls (2.6,.9) and (3.9,1.3) .. (6.15,2.3);
\draw[line width=.6pt,previewor] (2.2,.95) .. controls (3.3,1.1) and (4.8,1.7) .. (6.15,2.3);
\fill[frenetink] (2.05,1.64) -- (1.4,1.64) -- (1.55,1.3) -- cycle;
\draw[<->,black!65] (1.5,.93) -- (1.5,1.47);
\node[lab,anchor=east] at (1.32,1.19) {$d=e_{\mathrm{CTE}}$};
\draw[black!50,dashed] (1.75,1.48) -- (3.5,1.48);
\draw[frenetink] (1.75,1.48) -- (3.38,2.16);
\draw[black!70] (2.64,1.48) arc (0:23:0.89);
\node[lab] at (3.03,1.77) {$e_\psi$};
\draw[ar] (3.4,.35) -- (4.55,.35);
\node[lab,anchor=north] at (3.97,.2) {Arc length $s$};
\node[lab,anchor=north] at (1.05,.3) {Centerline};
\draw[black!40] (1.05,.38) -- (1.05,.85);
\node[lab,anchor=south] at (4.95,2.65) {Preview ahead};
\draw[black!40] (5.05,2.6) -- (5.35,2.01);
\draw[black!60] (7.8,.65) -- (12.65,.65);
\draw[black!60] (7.8,.65) -- (7.8,2.65);
\node[lab,anchor=south west] at (7.8,2.7) {$\hat\kappa(\Delta)$ ($\mathrm{m}^{-1}$)};
\node[lab,anchor=north] at (7.8,.48) {$0$};
\node[lab,anchor=north] at (12.3,.48) {$4.5$};
\node[lab,anchor=north] at (10.05,.03) {Preview offset $\Delta$ (m)};
\draw[frenetink,line width=.9pt] plot[smooth] coordinates {(7.8,.88)(8.3,.91)(8.8,.98)(9.3,1.12)(9.8,1.35)(10.3,1.61)(10.8,1.87)(11.3,2.1)(11.8,2.29)(12.3,2.45)};
\foreach \x/\y in {7.8/.88,8.3/.91,8.8/.98,9.3/1.12,9.8/1.35,10.3/1.61,10.8/1.87,11.3/2.1,11.8/2.29,12.3/2.45}
 {\fill[frenetink] (\x,\y) circle (1.5pt);}
\end{tikzpicture}
\caption{Road state and camera-derived context. (a) The road-relative state is
$z^r=(s,e_{\mathrm{CTE}},e_\psi)$: progress along the centerline, signed lateral
offset and heading relative to the road tangent. (b) The encoder supplies ten
curvature samples over a \SI{4.5}{m} preview. This schematic profile is separate
from the propagated state and is held fixed during rollout.}
\label{fig:encoding}
\end{figure}
''')
fig('fig:prediction',r'''
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{9}{11}\selectfont,
 ar/.style={-{Stealth[length=1.7mm]},line width=.65pt},
 lab/.style={font=\fontsize{8}{10}\selectfont,fill=white,inner sep=2pt}]
\node[anchor=west] at (0,3.8) {\textbf{(a)} Read a fixed road preview};
\node[anchor=west] at (7.25,3.8) {\textbf{(b)} Advance the road-relative state};
\draw[black!55] (.5,1.25) -- (6.55,1.25);
\draw[black!55] (.5,1.25) -- (.5,3.15);
\node[lab,anchor=south west] at (.5,3.15) {$\hat\kappa(\Delta)$};
\node[lab,anchor=north] at (3.5,.88) {$\Delta=s_t-s_{t_0}$};
\draw[black!45,line width=.9pt] plot[smooth] coordinates {(.65,1.55)(1.2,1.58)(2.2,1.8)(3.2,2.15)(4.2,2.47)(5.2,2.69)(6.3,2.8)};
\foreach \x/\y/\c/\t in {1.2/1.58/frenetink/t,3.2/2.15/teal/t+m,5.2/2.69/previewor/t+2m} {
 \draw[\c,dashed] (\x,1.25) -- (\x,\y);
 \fill[\c] (\x,\y) circle (2pt);
 \node[lab,anchor=south,text=\c] at (\x,\y+.16) {$\t$};}
\draw[ar,black!55] (1.2,.35) -- (5.2,.35);
\node[lab,anchor=north] at (3.2,.19) {Increasing predicted progress};
\node[draw=black!35,rounded corners=2pt,fill=frenetink!4,align=left,
 text width=5.0cm,inner sep=9pt] at (9.85,1.9) {
 Read: $\kappa_t=\hat\kappa(s_t-s_{t_0})$\\[7pt]
 Advance: $\dot s=\dfrac{v\cos e_\psi}{1-\kappa_t d}$\\[7pt]
 Update $d$ and $e_\psi$ with Frenet geometry.};
\end{tikzpicture}
\caption{Road prediction by re-indexing. (a) Successive states query the same
camera-derived curvature profile at increasing arc-length offsets. (b) Frenet
kinematics advance progress, lateral offset and heading error, with the learned
corrections described in the text. The profile itself is not extrapolated;
perception and state-integration errors can still affect the rollout.}
\label{fig:prediction}
\end{figure}
''')
# Figure and table caption font; maintain print readability without oversized labels.
s=s.replace(r'\usepackage{subcaption}',r'\usepackage{subcaption}'+'\n'+r'\captionsetup{font=small,labelfont=bf}'+'\n'+r'\usepackage[section]{placeins}')
# Restore experimental provenance and show the three original simulation environments.
a=s.index(r'\paragraph{Tier~1: Fully observable benchmarks}')
b=s.index(r'\paragraph{Tier~2:',a)
s=s[:a]+r'''\paragraph{Tier~1: Conference benchmarks}
The conference evaluation~\cite{piwm_conf} comprises \textbf{CartPole},
\textbf{Lunar Lander}~\cite{brockman2016openai}, and the \textbf{DonkeyCar
simulator}. CartPole and Lunar Lander provide controlled physical systems; the
simulator provides a visual driving benchmark with an approximate bicycle prior.
Figure~\ref{fig:conf_environments} shows their original observations. We retain
these experiments to document the original representation-learning results,
separately from the new road-context evaluation on the physical DonkeyCar.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_conference_environments.pdf}
\caption{Original conference environments: CartPole, Lunar Lander and the
DonkeyCar simulator. Frames are extracted from the ground-truth rows of the
conference qualitative figures~\cite{piwm_conf}; the simulated driving scene
is distinct from the physical platform evaluated in Tier~3.}
\label{fig:conf_environments}
\end{figure}

'''+s[b:]
s=s.replace('For each environment, we collect 60{,}000 trajectories of at least 50 steps,','The conference study collected 60{,}000 trajectories of at least 50 steps per environment,')
s=s.replace('Laps are split into disjoint training and validation sets; all numbers reported\nbelow are computed on the $201$ held-out validation windows, in real meters.',
'''The main comparison uses five folds over 71 episodes, totaling 2{,}483
validation windows; each fold holds out disjoint episodes. Position errors are
reported in meters. The legacy 201-window split is used only where explicitly
identified.''')
a=s.index(r'\subsection{Benchmark Results (Tier~1)}');b=s.index(r'\subsection{CarRacing (Tier~2)}',a)
s=s[:a]+r'''\subsection{Conference Benchmark Results (Tier~1)}\label{sec:conf_results}

We restore the original conference results~\cite{piwm_conf} in
Figures~\ref{fig:conf_cartpole}--\ref{fig:conf_rollouts}. These are inherited
experiments, not additional runs of PIWM-Frenet. They compare intrinsic and
extrinsic encoders with continuous or discrete latents over 30 prediction steps
and supervision-noise levels $\delta\in\{0,5\%,10\%\}$. The conference protocol
uses five-fold cross-validation. State RMSE in these benchmarks is distinct from
the meter-valued position error of the physical-car evaluation.

\paragraph{Prediction and architectural choice}
Figures~\ref{fig:conf_cartpole}, \ref{fig:conf_lunar} and
\ref{fig:conf_donkey} retain all six panels for each environment. Extrinsic
PIWM with discrete latents gives the lowest long-horizon error among the
extrinsic variants shown. The intrinsic setting has a different ranking:
continuous PIWM generally outperforms the discrete variant. Thus the conference
results support choosing the latent parameterization together with the encoder
architecture, rather than a universal advantage for quantization. The driving
encoder comparison in Table~\ref{tab:kappa_source} revisits this choice for the
new road-perception task.

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig_conference_cartpole.pdf}
\caption{Conference CartPole results~\cite{piwm_conf}. State RMSE over 30 steps
for (a) extrinsic and (b) intrinsic methods, under three supervision-noise
levels. Curves and shaded regions are the original embedded plot images;
labels and layout have been reset. Panel-specific vertical limits are retained.
The archived figure does not specify the statistical meaning of its shading.}
\label{fig:conf_cartpole}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig_conference_lunar.pdf}
\caption{Conference Lunar Lander results~\cite{piwm_conf}. The layout and
method groups follow Figure~\ref{fig:conf_cartpole}. Original curves and shaded
regions are preserved; all panels retain the source RMSE range $[0,2]$.}
\label{fig:conf_lunar}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig_conference_donkey.pdf}
\caption{Conference DonkeyCar-simulator results~\cite{piwm_conf}. The layout
follows Figure~\ref{fig:conf_cartpole}. Original curves and shaded regions are
preserved, including the source vertical limit of 2; curves beyond that limit
are clipped as in the original. These are 30-step state-RMSE results and must
not be read as the 100-step position errors of the physical car.}
\label{fig:conf_donkey}
\end{figure}

\paragraph{Parameter estimates and visual rollouts}
Figure~\ref{fig:conf_parameters} restores the parameter comparison for CartPole
and Lunar Lander. The reference lines make both agreement and departures from
the known parameters visible as supervision noise changes. We omit the original
DonkeyCar wheelbase panel from this recovery comparison because it has no
known-parameter reference. Figure~\ref{fig:conf_rollouts} retains all five
comparison rows from the original visual rollout example, with three evenly
spaced display columns selected for legibility. These examples illustrate visual
behavior and do not replace the quantitative comparisons above.

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig_conference_parameters.pdf}
\caption{Conference parameter estimates for the two controlled
benchmarks~\cite{piwm_conf}. Each panel retains the original colored intervals,
center marks and yellow ground-truth reference. Rows correspond to
$\delta=0,5\%,10\%$, from top to bottom. Parameter names and scales follow the
source figure; the source does not specify the interval statistic.}
\label{fig:conf_parameters}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig_conference_rollouts.pdf}
\caption{Conference visual rollouts at $\delta=5\%$~\cite{piwm_conf}.
Ground truth and the four original model rows are shown at $k=5,15,30$ for the
DonkeyCar simulator and Lunar Lander. Frames are extracted without retouching;
the intermediate display columns and the slide footer have been omitted.}
\label{fig:conf_rollouts}
\end{figure}
\FloatBarrier

'''+s[b:]
s=s.replace('All quantitative claims in this article therefore rest on the physical platform\nof Section~\\ref{sec:donkeyreal_results}, where the forward-facing camera makes',
'''All quantitative claims about the new road-context extension therefore rest on
the physical platform of Section~\\ref{sec:donkeyreal_results}, where the forward-facing camera makes''')
# Replace caption only; preserve figures and anchors.
a=s.index('    \\caption{Position error against rollout step on the physical DonkeyCar.')
b=s.index('    \\label{fig:donkey_main}',a)
s=s[:a]+r'''    \caption{Physical DonkeyCar prediction under five-fold cross-validation.
    (a) Mean position-error curves. (b) Error at step 100: small circles show the
    five fold means, diamonds their average, and horizontal segments their
    minimum--maximum range. All models use ground-truth initialization.
    SINDYc is omitted because every evaluated window diverges.}
'''+s[b:]
a=s.index('    \\caption{Robustness to corrupted supervision.');b=s.index('    \\label{fig:delta}',a)
s=s[:a]+r'''    \caption{Physical DonkeyCar robustness to corrupted supervision.
    Each panel shows one training-noise level for PIWM-Frenet; solid curves are
    means over five folds and shading spans their minimum--maximum range.
    The same dashed Vid2Param reference, trained with clean labels, is repeated
    in all panels. The three panels share both axes.}
'''+s[b:]
p.write_text(s,encoding='utf-8')
print('Revised manuscript, diagrams, captions and conference results.')
