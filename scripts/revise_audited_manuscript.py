"""Generate an evidence-aligned manuscript while preserving the original draft."""
from pathlib import Path
import re,json,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];J=R/'Jounral_PIWM';B=R/'_archive/journal/audited_revision_before'
B.mkdir(parents=True,exist_ok=True)
for n in ['elsarticle-template-num.tex','elsarticle-template-num.pdf']:
    if not (B/n).exists():shutil.copy2(J/n,B/n)
s=(B/'elsarticle-template-num.tex').read_text(encoding='utf-8')
def part(a,b):return s[s.index(a):s.index(b)]
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def block(env,label):
    for m in re.finditer(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',s,re.S):
        if '\\label{'+label+'}' in m[0]:return m[0]
    raise ValueError(label)
def caption(x,t):return re.sub(r'\\caption\{.*?\}\s*\\label',lambda m:'\\caption{'+t+'}\n\\label',x,count=1,flags=re.S)
for p in ['controlled_holdout','frenet_diagnostic','road_geometry_training']:
    assert read('reports/'+p+'/audit.json')['status']=='PASS'
ready=(R/'reports/conditioned_dynamics/audit.json').exists()
if ready:assert read('reports/conditioned_dynamics/audit.json')['status']=='PASS'
head=s[:s.index('\\begin{abstract}')]
head=re.sub(r'\\title\{.*?\}',lambda m:r'\title{Camera-Derived Road Context for Physically Interpretable Vehicle Prediction}',head,count=1,flags=re.S)
intro=r'''\begin{abstract}
Physically meaningful predictive states make learned vehicle models easier to
inspect, but road-aligned coordinates introduce geometric and numerical
assumptions. We extend the Physically Interpretable World Model (PIWM) framework
with a camera-derived finite road preview and structured road-relative dynamics.
The task is offline prediction conditioned on a known initial state and recorded
future actions. A road encoder predicts curvature and centerline shape from a
15-frame camera history; the controlled evaluation reconstructs motion without
querying the surveyed centerline during rollout. Initial road-relative state
remains privileged. We audit the formulation on two existing physical DonkeyCar
recordings, holding out each recording in turn and separating checkpoint
selection from evaluation. Matched controls include calibrated kinematics and
Cartesian residual models with and without the same visual road inputs.
The initial Frenet implementation gives 100-step endpoint errors of
1.086 and 1.463 m, compared with 0.936 and 1.158 m for calibrated kinematics.
Exploratory follow-ups diagnose unstable coordinate updates, inconsistent
curvature and shape parameterizations, and one nearly constant visual encoder.
Numerical safeguards improve optimization behavior, but the experiments do not
establish a consistent accuracy advantage from curvature-dependent dynamics.
The contribution is an explicit road representation together with a reproducible
assessment of its information requirements, geometric consistency, and empirical
limits. Repeated development on the same recordings constrains generalization
claims; neither unseen-track transfer nor camera-only deployment is demonstrated.
\end{abstract}
\begin{keyword}
physically interpretable prediction \sep road geometry \sep vehicle dynamics
\sep visual representation learning \sep retrospective evaluation
\end{keyword}
\end{frontmatter}

\section{Introduction}\label{sec:intro}
World models compress observations into states that can be propagated under
candidate actions~\cite{ha2018world,hafner2023mastering}. Physical interpretability
provides a useful complement to predictive accuracy: a predicted quantity should
have an identifiable meaning and unit, and its transition should expose the
assumptions connecting it to the modeled system. Our conference PIWM
framework~\cite{piwm_conf} investigated physical alignment under distribution-based
weak supervision. This article studies a road-context extension on physical
DonkeyCar records.

Road geometry matters for interpreting lateral displacement and heading relative
to a lane, and for the actions chosen by a driving controller. It does not, by
itself, change planar motion when the complete vehicle state, actuation sequence,
and physical conditions are fixed. We ask whether explicit visual road context
provides a useful inductive bias for finite-horizon action-conditioned prediction,
rather than claiming that identical physical states and actions must produce
different motion on differently painted roads.

Our state contains speed, yaw rate, progress, lateral offset, and road-relative
heading, accompanied by a finite road preview. The preview is perceived once and
queried at predicted progress. This avoids recurrent prediction of the profile,
but does not prevent errors in progress, perception, or actuation from
propagating. The Frenet chart can become singular, and independently predicted
curvature and centerline shape need not describe the same curve.

Two original recordings on one laboratory track provide 71 usable segments, not
71 independent acquisition sessions. The new evaluations separate fitting and
validation within one source and evaluate on the other. Both sources had already
informed development, so this analysis is retrospective. Earlier five-fold
results are preserved separately because their information and model-selection
conditions differ from those of the new comparisons. A subsequent purged temporal split includes both
sources in training and evaluates their later segments, while retaining all
source-transfer results.

The extensions beyond the conference study are: (i) a camera-derived curvature
and centerline preview with explicit road-relative state and map-free rollout
readout; (ii) analysis of coordinate validity, geometric consistency, and visual
optimization; and (iii) controlled source-separated and chronological experiments distinguishing
numerical stability from accuracy. The Frenet equations themselves are standard
coordinate transformations, not a new physical law. The evidence supports an
inspectable modeling approach and identifies limitations; it does not establish
universal superiority over calibrated dynamics.

Sections~\ref{sec:related} and~\ref{sec:preliminaries} provide background.
Sections~\ref{sec:problem}--\ref{sec:architecture} specify the implemented task
and models; Sections~\ref{sec:experiments}--\ref{sec:results} describe the protocol
and findings, followed by limitations and conclusions.

'''
related=part('\\section{Related Work}','\\section{Preliminaries}')
changes=[
('Compared with this body of work, PIWM learns interpretable representations','\\subsection{Representation Learning',r'''Our driving study examines named road variables and structured transitions.
Its controlled evaluation retains ground-truth initial state and direct map-derived
training labels; it does not demonstrate fully unaided visual driving.

'''),
('Compared with these approaches, PIWM explicitly aligns latent dimensions','\\subsection{World Models}',r'''The conference PIWM study uses distribution-based supervision. The driving
extension instead uses direct road labels to evaluate a manually specified
physical representation.

'''),
('The closest prior work, Vid2Param,','\\subsection{Physics-Informed',r'''Vid2Param estimates physical parameters from video~\cite{asenov2019vid2param}.
The new comparisons here use explicitly specified controls rather than claiming
full reproductions of these published architectures.

'''),
('PIWM differs in two ways aligned','\\subsection{Partial Observability',r'''We combine image-derived road geometry with road-relative pose kinematics and
learned actuation increments. This inductive bias requires comparison with
controls having comparable inputs and training conditions.

'''),
('This is the mechanism we use on the real DonkeyCar:','In driving specifically,',r'''Our preview encoder follows this training pattern, but rollout initialization
uses annotated pose and track geometry. Privileged information is therefore
reduced during prediction, not eliminated from the complete evaluation.

''')]
for a,b,t in changes:
    i=related.index(a);k=related.index(b,i);related=related[:i]+t+related[k:]
i=related.index('Compared with all of the above,')
related=related[:i]+r'''Our focus is the empirical role of explicit road context under action-conditioned
prediction, including numerical limitations. We do not claim priority for
combining lane geometry and vehicle physics in general.

'''
related+=r'''
Frenet-based trajectory representations predate the present study. Ye
et al.~\cite{ye2023frenet} use Frenet domain normalization to address scene
shift, while Hallgarten et al.~\cite{hallgarten2024stay} represent prediction
inputs and outputs relative to lane-centerline sequences to reduce off-road
forecasts. Their behavior-prediction setting differs from our supplied-action
vehicle rollout, so their reported scores are not directly comparable.
More recently, GeoWAM~\cite{lu2026geowam} forecasts visual scene geometry and
conditions ego-trajectory prediction on it. This is a broader geometric scene
representation than our compact road preview. These connections motivate a
specific contribution claim rather than claiming that road-aligned coordinates
or explicit geometric world states are themselves new.

'''
related=related.replace('These methods are transparent and certifiable, but they require an explicit\ndynamical model and a low-dimensional state, and they scale poorly to\nhigh-dimensional camera observations.', 'These approaches expose low-dimensional modeling assumptions. Their use with\ncamera observations requires an appropriate state-estimation interface, and\nany formal guarantee depends on the assumptions of the particular method.')
related=related.replace('These models, however, depend on handcrafted scene representations or HD-map\nannotations that are unavailable in many settings~\\cite{itkina2023interpretable,\nhsu2023interpretable}, and their internal states carry no physical meaning.', 'Scene representations and map inputs differ across these methods; availability\nof suitable annotations is one practical consideration~\\cite{itkina2023interpretable,\nhsu2023interpretable}. Physical interpretability must be assessed for each\nrepresentation rather than inferred from whether a model is learned.')
related=related.replace('guarantees\nbut do not make its latent state physically meaningful.', 'reliability analyses under their own assumptions; this is distinct from\nassigning physical semantics to a predictor\'s internal coordinates.')
related=related.replace('PIWM builds on this\nquantization but, unlike it, ties codebook components to physical values.', 'the conference PIWM formulation investigates physically supervised\nquantized representations. The current road encoder is continuous.')
prelim=part('\\section{Preliminaries}','\\section{Partial Observability in Driving}')
prelim=prelim.replace('PIWM enforces through the structured dynamics model.', 'PIWM encourages through its structured dynamics prior; learned residuals and numerical integration still require empirical checks.')
prelim+=r'''These definitions describe the conference framework. In the driving study,
the visual output is a road preview, not a reconstructed image, and the initial
vehicle state is supplied. Exact geometric supervision here differs from the
conference weak-supervision setting.

'''
method=r'''\section{Prediction Task and Information Conditions}\label{sec:problem}
At the beginning of each offline prediction window, the inputs are a 15-frame
camera history, initial speed $v_0$, a causal yaw-rate estimate $\omega_0$,
initial road-relative displacement $d_0$ and heading error $\psi_0$, and recorded
future steering/throttle actions. The output is planar motion in the initial
ego frame. Future images, future measured states, and controller feedback are
not supplied. Recorded actions are conditioning inputs, not predicted controls.

\subsection{Factorized state}\label{sec:factorized_state}
The state is $z=(s,d,\psi,v,\omega)$, accompanied by curvature samples and
centerline points at ten offsets from 0 to 4.5 m. Initial progress is zero.
Road context defines the coordinate system; we do not assert that this finite
preview is a sufficient Markov state for arbitrary driving environments.

The surveyed track and logged pose provide road labels and initial $d_0$ and
$\psi_0$. The controlled forecast readout uses predicted road points instead
of the surveyed centerline. Evaluation uses original logged planar positions,
not positions reconstructed from projected road labels. Thus label construction,
privileged initialization, forecast readout, and evaluation truth are distinct.
This is not a complete camera-only estimator or closed-loop driving system.

\section{Implemented Driving Models}\label{sec:architecture}
\subsection{Road-preview encoder}\label{sec:repr_learning}\label{sec:encoding}
A convolutional encoder receives the red channel of 15 consecutive $64\times64$
frames, stacked as channels and scaled to $[0,1]$. A 256-dimensional feature
vector feeds linear heads for ten curvature samples and twenty centerline
coordinates. The longitudinal shape output is a residual relative to nominal
offsets; the lateral output is direct. Joint supervised squared error is
normalized by training-set output standard deviations, floored at 0.05 in the
corresponding units. The controlled experiment does not use an image autoencoder,
VQ codebook, Transformer, or joint perception/dynamics fine-tuning.

The original encoder uses ReLU and outputs directly in physical units.
An exploratory repair uses LeakyReLU with slope 0.01 and
$\hat y=\mu_{\rm train}+\sigma_{\rm train}r_\theta$, retaining width, depth,
loss, and training budget. All six encoders are retrained under this rule.
Changing activation and output parameterization together prevents isolating
their individual causal contributions.

\subsection{Road-relative dynamics}\label{sec:prediction}
For an arc-length centerline within its valid coordinate chart, slip-free
kinematics gives
\begin{align}
 \dot s&=\frac{v\cos\psi}{1-d\kappa},\label{eq:frenet_s}\\
 \dot d&=v\sin\psi,\label{eq:frenet_d}\\
 \dot\psi&=\omega-\kappa\dot s.\label{eq:frenet_psi}
\end{align}
The independently predicted curve and legacy map labels only approximate
these arc-length identities; the original and guarded models are structured
predictors rather than exact physical coordinate transformations. The
consistent variants below explicitly address this distinction.
The implementation adds learned lateral and heading residuals and per-step
increments in speed and yaw rate. Three small multilayer perceptrons consume
normalized $d,\psi,v,\omega,\kappa$ and actions. These actuation increments are
not recovered physical bicycle parameters. The original variant uses Euler
updates, interpolated curvature from the visual head, and a separate predicted
centerline for reconstruction.

A cubic Hermite interpolant $P(s)$ uses centered interior tangents, an initial
road-$x$ tangent, and a final chord tangent. With tangent heading $\theta(s)$,
the vehicle position is $P(s)+d[-\sin\theta(s),\cos\theta(s)]^{\mathsf T}$ and
heading is $\theta(s)+\psi$. An ego transform removes initial translation and
rotation. Outside the 4.5 m preview, the position readout extends its endpoint
tangent while the independent curvature head holds its last value. These two
extensions are not generally geometrically consistent.

\paragraph{Numerical safeguards}
The guarded variant uses $\max(0.2,1-d\kappa)$, wraps heading, and bounds learned
increments with hyperbolic tangents. Speed and yaw-rate bounds are training-only
99.5th percentiles of absolute one-frame changes, floored at 0.01 in their
respective units. Lateral and heading residual limits are 0.1 m and 0.1 rad
per step. Positive denominator protection extends the numerical update outside
its valid chart; it does not make such states physically valid. We record states
with $1-d\kappa\leq0$. Finite increments imply neither bounded states nor
bounded prediction errors.

\paragraph{Geometric consistency}
Nominal arc-length knots do not guarantee a unit-speed predicted interpolant.
Let $q$ be its actual parameter, $g=\|P'(q)\|$, and
$\kappa_P=\det(P'(q),P''(q))/g^3$. Consistent dynamics requires
\begin{equation}
 \dot q=\frac{v\cos\psi}{g(1-d\kappa_P)},\qquad
 \dot\psi=\omega-\kappa_Pg\dot q.\label{eq:consistent_geometry}
\end{equation}
We evaluate this correction with Euler and explicit midpoint integration, using
the same curve in dynamics and readout. The independent curvature head is
ignored, $g$ is floored at 0.1, and the other safeguards are retained.
Tangent extension uses $g=1$ and $\kappa_P=0$ outside the preview.
Synthetic checks assess readout equivalence, finite-difference derivatives,
and the coordinate-to-Cartesian velocity identity. Consistency does not imply
that the predicted curve matches the real road.

\subsection{Controls and training}\label{sec:baselines}
A calibrated kinematic control learns seven global coefficients for throttle
gain, drag, effective length, steering gain and bias, drive bias, and yaw-response
lag. It receives initial speed, yaw rate, and future actions, without road inputs.
Two Cartesian residual controls have the same architecture; one receives the
initial road state and frozen visual preview, and the other zeros these inputs.
Both learn residuals to analytic planar kinematics. These implementations are
not claimed reproductions of DVBF, GOKU-net, or Vid2Param. Constant speed and yaw
rate provide an additional action-independent diagnostic.

Perception training uses 25 epochs, Adam with initial learning rate $10^{-3}$,
cosine decay, batch size 64, and frame stride 2. The checkpoint minimizes
validation normalized supervision loss. Dynamics use 40 epochs, 32-step windows,
batch size 128, stride 4, and a common physical-output loss. Initial learning
rates are $10^{-3}$ for neural dynamics and $10^{-2}$ for calibrated kinematics,
with cosine decay. Gradient norms are clipped at 1 for dynamics and 5 for
perception. Dynamics checkpoints minimize validation 100-step endpoint error.
All seeds (0, 1, 2) are retained.

The loss averages squared errors in ego-frame $x,y$, wrapped heading, speed,
and yaw rate, scaled by 0.5 m, 0.5 m, 0.5 rad, and training standard deviations
of speed and yaw rate (floored at 0.01). Evaluation errors do not choose
checkpoints within these runs; choices between modeling rounds are exploratory
because earlier results had already been inspected.

\section{Experimental Evaluation}\label{sec:experiments}\label{sec:setup}
\subsection{Data and source-separated protocol}
The physical archive contains two recordings with 5801 and 13597 frames at
approximately 22 Hz, including images, steering/throttle, logged planar pose,
speed, and a track map. Preprocessing yields 18 and 53 usable segments.
These segments share acquisition conditions and layout; they are not independent
sessions. Yaw rate uses at most five past yaw differences.

Direction A evaluates the first source, with 41 training, 11 validation, and one
purged segment from the second source. Direction B evaluates the second source,
with 13 training, 4 validation, and one purged segment from the first.
Validation is the final 20\% of ordered development-source segments; at least
115 original frames separate training and validation. Evaluation uses 844 and
1653 overlapping windows with stride 4 and horizon 100 (approximately 4.55 s).

Learned parameters and normalization statistics use training data only. Each
round freezes its protocol, implementation hashes, and required input checkpoints
before fitting; evaluation waits for all prescribed runs. Audits check data and
checkpoint hashes, validation selection, window identity, recomputed metrics,
and zero initial position error.

Both recordings were previously used in development. Retained track preprocessing
also includes orientation chosen using the first recording and historically
explored grid settings. Separation therefore concerns present fitting and
checkpoint selection, not a prospectively untouched complete pipeline.
No unseen-track or new-recording generalization claim follows.

\subsection{Metrics}
Endpoint error is
\begin{equation}
 E_{xy}(k)=\frac1N\sum_{i=1}^{N}\|\hat p_{i,k}-p_{i,k}\|_2.\label{eq:metric_exy}
\end{equation}
We report steps 25, 50, and 100, plus mean error over steps 1--100 (ADE), in meters.
Mean $\pm$ standard deviation summarizes three training seeds, not uncertainty
across independent recording sessions. Overlapping windows are not independent
trials for significance testing. The primary metric is 100-step endpoint error.

\subsection{Earlier experiments}
Conference CartPole, Lunar Lander, and DonkeyCar-simulator results are retained
with attribution~\cite{piwm_conf}. Their supervision and architectures differ
from the current road encoder. CarRacing was a development environment, but this
article has no completed quantitative CarRacing validation of the road-context
model. Earlier five-fold physical-car results are preserved in
Appendix~\ref{sec:historical}, under their original information and selection
conditions; they cannot be pooled with the new source-separated estimates.

\section{Results}\label{sec:results}
\subsection{Initial controlled comparison}\label{sec:donkeyreal_results}
Table~\ref{tab:controlled} shows lower mean Frenet endpoint error than the
road-conditioned Cartesian residual control in both directions, but calibrated
kinematics is better in both. These findings concern the specified implementations
and budgets, rather than all possible models in either coordinate system.
'''
method=method.replace(r'\subsection{Metrics}',r'''
\paragraph{Operating regimes and identifiability}
Across retained frames, median speeds are 0.434 and 0.774 m/s in the two records.
The first has constant throttle 0.566667, whereas the second ranges from
0.633333 to 0.766667. Source separation therefore also probes changes in operating
regime. In a model $\dot v=k_a u-k_dv+b$, constant $u=c$ identifies the combination
$k_ac+b$, not $k_a$ and $b$ separately: replacing them by $k_a+\Delta$ and
$b-c\Delta$ leaves the update unchanged. We do not interpret fitted coefficients
as uniquely recovered physical parameters. This limitation does not by itself
explain all prediction errors.

\subsection{Metrics}
''')
arch=block('figure','fig:arch')
arch=arch.replace(r'Curvature preview\\$\hat\kappa(\Delta)$',r'Curvature and shape\\$\hat\kappa(\Delta),\ P(\Delta)$')
arch=arch.replace('GT state at $t_0$','Known initial state')
arch=arch.replace(r'$\kappa_t=\hat\kappa(s_t-s_{t_0})$', 'Query at predicted progress')
arch=caption(arch, r'Implemented prediction pathways. The camera supplies curvature and shape once per rollout; annotated state initializes the dynamics. Recorded actions drive subsequent updates. The original and guarded variants query the curvature head; the consistent variant derives curvature and parameter speed from the shape curve. Position is decoded using that predicted curve, without surveyed-map queries during rollout. Initial road-relative state remains privileged.')
method=method.replace(r'\subsection{Road-preview encoder}',arch+'\n'+r'\subsection{Road-preview encoder}')
method=method.replace('Three small multilayer perceptrons consume', 'Three multilayer perceptrons, each with two 64-unit ReLU hidden layers, consume')
method=method.replace('vector feeds linear heads for ten curvature samples', 'vector feeds linear heads for ten curvature samples')
method=method.replace('frames, stacked as channels and scaled to $[0,1]$. A 256-dimensional feature', 'frames, stacked as channels and scaled to $[0,1]$. Four convolutions have\n32, 64, 128, and 256 channels, kernel size 4, stride 2, and padding 1. The\nflattened output feeds a 256-unit fully connected layer. This feature')
method=method.replace('Both learn residuals to analytic planar kinematics.', 'Their residual network has two 96-unit ReLU hidden layers and five outputs.\nIts inputs are scaled planar position, sine/cosine of heading, normalized speed\nand yaw rate, two actions, and 32 normalized road-context quantities. Both learn\nresiduals to analytic planar kinematics.')
method=method.replace('before fitting; evaluation waits for all prescribed runs.', 'before fitting; evaluation waits for all prescribed runs.')

def table(label,title,rows):
    text='\\begin{table}[t]\n\\centering\n\\caption{'+title+'}\n\\label{'+label+'}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrr}\n\\toprule\nModel & A: $E_{100}$ & A: ADE & B: $E_{100}$ & B: ADE \\\\\n\\midrule\n'
    for name,vals in rows:
        cells=['$'+f'{np.mean(v):.3f} \\pm {np.std(v,ddof=1):.3f}'+'$' for v in vals]
        text+=name+' & '+' & '.join(cells)+' \\\\\n'
    return text+'\\bottomrule\n\\end{tabular}\n\\end{table}\n'
def rows(path,names,suffix=''):
    data=read(path)['results'];out=[]
    for key,label in names:
        values=[[data[f'{f}/seed{i}/{key}{suffix}'][m] for i in range(3)] for f in ['outer0','outer1'] for m in ['E100','ADE']]
        out.append((label,values))
    return out
results=table('tab:controlled','Initial retrospective comparison. Errors in m; mean $\\pm$ sample SD over three seeds. A and B evaluate the first and second original recordings.',rows('reports/controlled_holdout/test_results.json',[('kinematic','Calibrated kinematics'),('cartesian_no_road','Cartesian, no road'),('cartesian_road','Cartesian, road'),('frenet','Frenet')],'/normal'))
results+=r'''
Replacing both curvature and shape by a straight road raises Frenet endpoint
error to $2.129\pm0.546$ and $2.415\pm0.071$ m. Setting curvature alone to zero
while keeping shape gives $0.994\pm0.126$ and $1.517\pm0.069$ m. The joint
intervention therefore does not isolate curvature-dependent dynamics.
Inference-time interventions also introduce distribution shift; separately
trained ablations provide a complementary comparison.

\subsection{Numerical stability and retrained ablations}
Two original Frenet runs deteriorate after their validation-optimal epochs:
final validation errors reach 99.875 and 21.681 m, with maximum pre-clipping
gradient norms above $10^{12}$ and $10^{13}$. Selected checkpoints reproduce the
initial comparison exactly in the diagnostic rerun. Guarded updates avoid these
extreme failures in the same runs, but can still leave the valid Frenet chart.

Table~\ref{tab:guards} reports models refit with fixed perception. Shape-only
sets curvature to zero throughout fitting and prediction. No-road additionally
replaces shape by a straight line and zeros initial road-relative displacement
and heading; it does not isolate shape alone. The oracle receives true curvature
and shape during fitting and evaluation; it is not a theoretical upper bound.
'''
results+=table('tab:guards','Exploratory numerical safeguards and retrained ablations using the original frozen encoders.',rows('reports/frenet_diagnostic/results.json',[('guarded_full','Guarded, full'),('guarded_shape','Guarded, shape only'),('guarded_none','Guarded, no road'),('guarded_oracle','Guarded, oracle')]))
results+=r'''
Shape-only improves on the full model in A but worsens it in B. Neither the
safeguards nor the oracle establishes a consistent advantage over calibrated
kinematics, so the remaining gap cannot be attributed solely to perception.

\subsection{Geometric consistency and integration}
A validation-only audit finds that predicted curves are not generally unit-speed
in their nominal parameter. For one encoder, 80.1\% of sampled locations have
$|g-1|>0.2$. Independent and shape-derived curvatures also disagree. Even the
coarse oracle interpolant is not exactly unit-speed. Synthetic checks verify
the coordinate velocity identity in Eq.~\eqref{eq:consistent_geometry}.
On a synthetic circle, zero-residual 100-step endpoint error falls from
0.001293 m with Euler to 0.000119 m with midpoint integration. This checks the
numerical implementation, not accuracy on recorded driving.

Table~\ref{tab:geometry} evaluates consistent dynamics with the original encoders.
The body midpoint control uses the same network structure and integrator, but
clears road inputs. Geometry helps relative to this control in B; A has the
opposite ordering. Calibrated kinematics still gives lower mean endpoint error
in both directions.
'''
results+=table('tab:geometry','Exploratory geometry and integration controls with the original encoders.',rows('reports/road_geometry_training/results.json',[('geometry_euler','Consistent, Euler'),('geometry_midpoint','Consistent, midpoint'),('body_midpoint','Body, midpoint')]))
results+=r'''
\subsection{Visual conditioning and matched dynamics}
Training and validation logs identify one encoder with nearly constant outputs.
Its best normalized validation loss is 0.8384, versus 0.1220 and 0.1280 for the
other seeds in that direction. This diagnoses representation collapse, not its
precise optimization mechanism. Uniform conditioning repair improves validation
loss in five of six runs; one changes from 0.1220 to 0.1371. The collapsed run
improves to 0.2953 with nonconstant outputs. Validation improvements alone do
not demonstrate improved trajectory generalization.
'''
if ready:
    results+=table('tab:conditioned','Matched dynamics with six uniformly retrained conditioned encoders. All dynamics are refit from scratch; all variants and seeds are retained.',rows('reports/conditioned_dynamics/results.json',[('guarded_full','Guarded, full'),('guarded_shape','Guarded, shape only'),('geometry_midpoint','Consistent, midpoint'),('cartesian_road','Cartesian, road')]))
    results+=r'''Table~\ref{tab:conditioned} uses the same previously examined sources. The
consistent model ignores the independent curvature head, while shape-only clears
that head during dynamics. Neither ablation isolates curvature supervision of
the jointly trained encoder. The kinematic and no-road Cartesian references in
Table~\ref{tab:controlled} are unaffected by encoder changes.
'''
else:
    results+=r'''\textbf{Development status:} The fixed matched-dynamics experiment with these
six encoders is still running. Its complete audited results must be inserted
before this draft is considered complete; no trajectory improvement is inferred.
'''
if ready and (J/'imgs/fig_conditioned_comparison.pdf').exists():
    results+=r'''
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_conditioned_comparison.pdf}
\caption{Complete conditioned-dynamics comparison and fixed controls. Lines are
means over three training seeds; shading is sample standard deviation across
seeds, not a confidence interval over recording sessions. All curves use the
same source-specific windows, privileged initialization, and recorded actions.
Both sources had been inspected in earlier development.}
\label{fig:conditioned_comparison}
\end{figure}
'''
temporal_ready=(R/'reports/temporal_holdout/audit.json').exists()
if temporal_ready:
    assert read('reports/temporal_holdout/audit.json')['status']=='PASS'
    pooled=read('reports/temporal_holdout/results.json')['results']
    specific=read('reports/temporal_holdout/per_source_results.json')['results']
    labels=[('kinematic','Calibrated kinematics'),('cartesian_no_road','Cartesian, no road'),('cartesian_road','Cartesian, road'),('guarded_full','Guarded, full'),('guarded_shape','Guarded, shape only'),('geometry_midpoint','Consistent, midpoint')]
    temporal_rows=[]
    for key,label in labels:
        vals=[[pooled[f'temporal/seed{i}/{key}'][m] for i in range(3)] for m in ['E100','ADE']]
        vals += [[specific[f'{source}/seed{i}/{key}']['E100'] for i in range(3)] for source in ['traj1_64x64','traj2_64x64']]
        temporal_rows.append((label,vals))
    results+=r'''
\subsection{Purged temporal prediction within both sources}\label{sec:temporal}
To distinguish source/regime transfer from forecasting within recorded operating
conditions, we add an exploratory chronological split separately within each
source. The first approximately 60\% of ordered segments train the models,
the next 20\% select checkpoints, and the last 20\% evaluate them. Boundary
segments are purged to leave at least 115 original frames between adjacent
partitions. This produces 9+30 training segments, 3+10 validation segments,
4+11 evaluation segments, and 2+2 purged segments.
All encoders and six dynamics variants are trained from scratch using the same
25/40-epoch budgets. The guarded full model and pooled-window endpoint metric
are fixed before fitting; no family is selected retrospectively as the winner.

This experiment was designed after inspecting the earlier studies and the
recorded operating ranges. It does not provide fresh independent confirmation,
and it does not replace the source-held-out stress tests. The narrower question
is prediction on later portions of the same sources, with both sources represented
during training. Training throttle nevertheless contains only 0.566667 and
0.633333, whereas the second full record reaches 0.766667. This split is not
asserted to be identically distributed or to cover every later operating condition. Table~\ref{tab:temporal} reports the pooled primary
metric and both source-specific endpoint errors.
'''
    tt=table('tab:temporal','Exploratory chronological holdout with both sources represented during training; later operating shifts may remain. Pooled-window E100 is primary; the last two columns expose the original sources separately. Errors in m, mean and sample SD over three seeds.',temporal_rows)
    tt=tt.replace('A: $E_{100}$ & A: ADE & B: $E_{100}$ & B: ADE','Pooled $E_{100}$ & Pooled ADE & Rec. 1 $E_{100}$ & Rec. 2 $E_{100}$')
    results+=tt
    def pooled_mean(name):return np.mean([pooled[f'temporal/seed{i}/{name}']['E100'] for i in range(3)])
    results+=f'The prespecified guarded full model gives a pooled endpoint error of {pooled_mean("guarded_full"):.3f} m, compared with {pooled_mean("kinematic"):.3f} m for calibrated kinematics and {pooled_mean("cartesian_road"):.3f} m for the road-conditioned Cartesian control. These values must be interpreted together with the per-source results and the cross-source stress tests, not as evidence for unseen-track generalization.\n'


coupled_ready=(R/'reports/coupled_road/audit.json').exists()
if coupled_ready:
    assert read('reports/coupled_road/audit.json')['status']=='PASS'
    cp=read('reports/coupled_road/results.json')['results'];cs=read('reports/coupled_road/per_source_results.json')['results']
    labels=[('arc_full','Coupled arc, full'),('arc_shape','Coupled arc, shape only'),('arc_independent_encoder','Arc, independent encoder'),('cartesian_road','Cartesian, coupled road')]
    coupled_rows=[]
    for key,label in labels:
        vals=[[cp[f'temporal/seed{i}/{key}'][m] for i in range(3)] for m in ['E100','ADE']]
        vals += [[cs[f'{source}/seed{i}/{key}']['E100'] for i in range(3)] for source in ['traj1_64x64','traj2_64x64']]
        coupled_rows.append((label,vals))
    results+=r"""
\subsection{A single unit-speed road representation}\label{sec:coupled}
The preceding consistency audit motivates an exploratory encoder whose curvature
and position describe one curve by construction. Ten predicted curvature knots
replace the independent curvature and shape heads. For the nine intervals of
length $\ell=0.5$ m, let $\bar\kappa_j=(\hat\kappa_j+\hat\kappa_{j+1})/2$ and
$\theta_j=\sum_{i<j}\bar\kappa_i\ell$. At distance $u\in[0,\ell]$ in interval $j$,
\begin{align}
 C(q_j+u)&=C(q_j)+u\,\operatorname{sinc}(\bar\kappa_j u/2)
 \begin{bmatrix}\cos(\theta_j+\bar\kappa_j u/2)\\
 \sin(\theta_j+\bar\kappa_j u/2)\end{bmatrix},\\
 \theta(q_j+u)&=\theta_j+\bar\kappa_j u,
\end{align}
where $\operatorname{sinc}(x)=\sin(x)/x$, extended continuously at zero.
Position and tangent are continuous; curvature can jump at interval boundaries.
The curve has unit parameter speed, and the same interval curvature drives
midpoint Frenet updates and position readout. Outside the preview, tangent
extension has zero curvature. This is a consistent implementation of standard
curve geometry, not a new Frenet identity or a stability guarantee.

The encoder retains the conditioned CNN backbone and is jointly supervised by
legacy curvature and shape targets; shape gradients propagate through the arc
integral. All three seeds are refit for 25 epochs. Four dynamics variants are
refit for 40 epochs on exactly the temporal split in Section~\ref{sec:temporal}.
The full coupled model and pooled $E_{100}$ are fixed before fitting. The
independent-encoder control uses the preceding frozen CNN's curvature output
with the same arc decoder and dynamics. The shape-only variant removes
curvature from the dynamics but retains the curved readout; it does not remove
all curvature information. Cartesian dynamics receives the same coupled preview.
"""
    tt=table('tab:coupled','Exploratory coupled representation on the unchanged chronological split. Mean and sample SD over all three seeds; all variants retained.',coupled_rows)
    results+=tt.replace('A: $E_{100}$ & A: ADE & B: $E_{100}$ & B: ADE','Pooled $E_{100}$ & Pooled ADE & Rec. 1 $E_{100}$ & Rec. 2 $E_{100}$')
    def cm(name):return np.mean([cp[f'temporal/seed{i}/{name}']['E100'] for i in range(3)])
    results+=f'The full coupled model has pooled endpoint error {cm("arc_full"):.3f} m, compared with {cm("arc_independent_encoder"):.3f} m for the independent-encoder arc control and {cm("cartesian_road"):.3f} m for Cartesian dynamics using the coupled preview. The unchanged calibrated-kinematic reference is {pooled_mean("kinematic"):.3f} m. These comparisons jointly expose coupling, geometry, and model-family effects.\n'
    if (J/'imgs/fig_temporal_comparison.pdf').exists():
        results+=r"""
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_temporal_comparison.pdf}
\caption{Chronological-holdout errors for every prescribed variant. The panels
separate the independent-head and coupled-road suites; calibrated kinematics
is repeated as a common reference. Means and shading summarize three seeds
(sample standard deviation), not independent recording uncertainty.}
\label{fig:temporal_comparison}
\end{figure}
"""
label_audit=read('reports/road_geometry/native_label_audit.json')
grid=label_audit['map'];val_labels=label_audit['partitions']['validation']
results+=r"""
\subsection{Compatibility of the geometric supervision}
The persisted centerline is smoothed after resampling and is not reparameterized
by arc length afterward. Its nominal grid length is """+f"{grid['nominal_length']:.3f}"+r""" m,
whereas the smoothed polygon length is """+f"{grid['polygon_length']:.3f}"+r""" m. At the native 0.05 m grid,
the median absolute deviation of parameter speed from one is """+f"{grid['native_speed_abs_error']['median']:.4f}"+r""".
Stored curvature is additionally smoothed. Thus these direct map-derived labels
are not exact, mutually consistent geometric ground truth.

On the temporal validation windows, integrating the stored curvature with the
piecewise-arc decoder differs from the stored shape by """+f"{val_labels['coordinate_RMSE']:.3f}"+r""" m
coordinate RMSE. This difference combines map parameterization, separate
curvature smoothing, and coarse interval approximation; it does not isolate one
cause. Joint supervision of an exact unit-speed representation therefore
introduces target mismatch. No test-window information is used in this audit,
and the existing targets and complete experimental results are retained.
This limits conclusions about the best attainable accuracy of geometrically
consistent learning from corrected labels.
"""


if temporal_ready:
    paired=read('reports/temporal_holdout/paired_segments.json')
    improvements=paired['comparisons']['kinematic'];count=improvements['segments_total']
    results+=f'For the fixed guarded full model, seed-averaged endpoint error improves on calibrated kinematics and both Cartesian controls in all {count} eligible temporal test segments. Only {count} of the 15 assigned segments are long enough for the prescribed 15-frame context and 100-step horizon, yielding 527 windows. Full dynamics improves on shape-only in {paired["comparisons"]["guarded_shape"]["segments_improved"]} of those segments. This is descriptive paired evidence: overlapping windows and segments from two recordings do not provide {count} independent acquisition sessions.\n'
    diag=read('reports/temporal_holdout/results.json')['diagnostics']
    invalid=max(diag[f'temporal/seed{i}/guarded_full']['invalid_chart_fraction'] for i in range(3))
    results+=f'The maximum fraction of monitored full-step states with an invalid Frenet denominator in the guarded full model is {100*invalid:.4f}\\%. Finite rollouts therefore do not establish chart validity. Midpoint internal stages are not included in this monitoring.\n'
    intro=intro.replace('Numerical safeguards improve optimization behavior, but the experiments do not\nestablish a consistent accuracy advantage from curvature-dependent dynamics.',f'In a subsequent purged chronological holdout of both sources, the fixed guarded\nmodel achieves {pooled_mean("guarded_full"):.3f} m pooled endpoint error, versus {pooled_mean("kinematic"):.3f} m for calibrated\nkinematics and {pooled_mean("cartesian_road"):.3f} m for the same-input Cartesian control. These improvements\nare specific to the chronological setting; the source-transfer tests do not\nestablish a consistent advantage from curvature-dependent dynamics.')
conf=block('figure','fig:conf_environments')+'\n'+part('\\subsection{Conference Benchmark Results (Tier~1)}','\\subsection{CarRacing (Tier~2)}')
conf=conf.replace('Conference Benchmark Results (Tier~1)','Retained Conference Benchmark Results').replace('physical platform evaluated in Tier~3','physical platform evaluated in the present study')
discuss=r'''
\section{Discussion}\label{sec:discussion}
The studies distinguish physical meaning, numerical behavior, and predictive
accuracy. The first two do not imply the third. Named variables expose coordinate
failures and inconsistent geometric representations, as the audits demonstrate.
Accuracy benefits depend on the recording, split, baseline, and implementation.
The chronological-holdout gains are substantial and reproducible from saved
weights, but the protocol was designed during development on reused sources.
They support within-source prediction under the stated information conditions,
while the source-separated results expose limited transfer.

Road geometry alone does not alter action-conditioned planar kinematics. As
context for learned actuation, it may encode correlations with driving conditions
or the behavior generating the data. These experiments do not identify such
correlations as causal effects. Re-indexing a fixed preview removes one learned
recurrent update but still propagates errors in progress and geometry.

The principal limitation is two reused recordings on one track. Seeds assess
optimization variability, not variation across independent tracks, vehicles,
or acquisition sessions. Successive repairs were informed by previous analysis;
their evaluation is exploratory even with validation-only checkpoint selection.
Fresh acquisition is needed for prospective confirmation of a selected method.
Other limitations are privileged initialization, map-derived encoder labels, recorded
future actions, finite preview, and absence of closed-loop assessment. Numerical
guards can remain finite outside the valid coordinate chart. No safety or
general bounded-error guarantee follows.

The direct road encoder differs from the conference autoencoding pipeline, whose
weak-supervision results do not establish weakly supervised road perception.
Historical reduced baselines also cannot establish superiority over the full
published methods. A deployment accuracy claim requires benefits over calibrated
dynamics across independent conditions and evaluation with an initial-state
estimator. Favorable subsets of these development experiments cannot satisfy
those requirements.

\section{Conclusion}\label{sec:conclusion}
We developed and audited a camera-derived road-context extension of PIWM for
offline action-conditioned prediction. Explicit geometry makes coordinate
assumptions and failures inspectable. Controlled studies distinguish input
information, reconstruction, numerical safeguards, and dynamics training.
The original Frenet predictor does not beat calibrated kinematics in the two
source-held-out directions. The guarded model substantially improves chronological prediction
over matched controls, with improvements in both recordings and across all
eligible test segments against the three principal baselines. This contrast
with source-transfer performance identifies a useful but limited operating
regime for the representation; geometric consistency alone is insufficient
to establish predictive superiority.
The evidence supports a limited empirical account of the representation, not
resolved partial observability, camera-only deployment, or guaranteed stability.

'''
meta=part('\\section*{Declaration of Competing Interest}','\\appendix')
meta=meta.replace('The authors declare that they have no known competing financial interests or\npersonal relationships that could have appeared to influence the work reported in\nthis article.','[AUTHOR CONFIRMATION REQUIRED: competing-interest declaration.]')
meta=meta.replace('The DonkeyCar trajectories, the derived road-context labels and the code used to\ntrain and evaluate the models will be made available in a public repository upon\npublication.','[AUTHOR CONFIRMATION REQUIRED: data/code release location, permissions, and access restrictions.]')
meta+=r'''\section*{Declaration of Generative AI Use}
[AUTHOR REVIEW REQUIRED BEFORE SUBMISSION.] GPT-6 (Codex) assisted with code
inspection, experimental implementation, analysis, and manuscript revision.
The final disclosure requires author verification of its scope and compliance
with the journal policy. This draft does not assert that the authors have
already verified or approved all assisted outputs.

'''
history=r'''\appendix
\section{Historical Physical-Car Development Results}\label{sec:historical}
These tables and figures preserve earlier five-fold development results.
The 71 segments come from the same two recordings. Displayed evaluation windows
also selected dynamics checkpoints, so these estimates are optimistic as measures
of generalization. The original Vid2Param visual encoder was shared across folds;
DVBF and GOKU-net were reduced to dynamics-only variants. Budgets and information
pathways differed. These are descriptive results under the original protocol,
not a clean ranking of published methods.

The 0.463 m result uses surveyed-map position reconstruction and privileged
initialization. The 0.516 m row uses perceived-shape readout with the original
selection and initialization conditions. Differences from the source-separated
study also involve splits, losses, horizons, and readout implementation; they
are not a controlled estimate of one isolated source of optimism.
'''
caps={
 'tab:donkey_main':r'Historical five-fold development errors (m). Displayed windows also selected checkpoints. Frenet uses map-assisted readout except in the map-free row; all rows use privileged initialization. Baseline names refer to local reduced implementations. These are not independent test estimates.',
 'tab:kappa_source':r'Historical curvature-source substitutions using the original dynamics and evaluation protocol. Curvature RMSE is in inverse meters; endpoint error is in meters. Camera refers to the curvature input only; readout and initialization retain the original privileged information. Limited sensitivity here does not establish absence of error propagation.',
 'tab:delta':r'Historical perturbation of dynamics supervision, mean and standard deviation over five development folds. Road-encoder labels are not perturbed, so this does not establish weakly supervised visual road learning. The Gaussian row is a matched-magnitude diagnostic, not evidence about all noise distributions.',
 'tab:noise':r'Historical initial-state perturbations with ten draws per window. Values are preserved from the original aggregation. The clean column differs slightly from the main historical table and should not be treated as an exact reproduction. Noise is relative to the original state-normalization scales.'}
for label,cap in caps.items():history+=caption(block('table',label),cap)+'\n'
for label,cap in [
 ('fig:donkey_main','Historical five-fold development curves with ground-truth initialization and checkpoint selection on the displayed windows. The principal Frenet curve uses map-assisted reconstruction. SINDYc is omitted after divergence.'),
 ('fig:delta','Historical dynamics-supervision perturbation curves. Solid curves average five development folds; shading spans their range. The repeated dashed reference uses clean-label Vid2Param. This is not noisy road-encoder supervision.')]:
    history+=caption(block('figure',label),cap)+'\n'
history+=r'''These results motivated follow-up studies. Small effects from curvature
replacement and gains under some noise conditions do not identify a unique
causal mechanism; they should be read alongside the controlled main-text results.

'''
append=s[s.index('\\section{Weak Supervision Noise Model}'):]
# Correct the retained appendix's supervision scope as well as the main text.
a=append.index('The same noise model is applied to the road-context supervision')
z=append.index('\\section{Conference Controller Diagnostic}',a)
append=append[:a]+r'''
The physical-car historical study in Table~\ref{tab:delta} perturbs dynamics
state supervision and initialization. It does not perturb the image-to-curvature
or image-to-shape training labels. The new controlled studies use direct map-derived
labels; their road preview is inferred from images, while initial road-relative
state remains privileged.

'''+append[z:]
append=append.replace('Tier~1 benchmark experiments','retained conference benchmark experiments')
head=head[head.index('\\documentclass'):]
output=head+intro+related+prelim+method+results+conf+discuss+meta+history+append
(J/'elsarticle-template-num.tex').write_text(output,encoding='utf-8')
print('Revised:',len(output.splitlines()),'lines; conditioned results included:',ready)
