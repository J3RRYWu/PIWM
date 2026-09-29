"""Rewrite the manuscript around the audited corrected-query evidence.

Preserve the immediate prior manuscript and every historical result. This stage
runs only after the declared fits, checkpoint checks and sensitivity evaluation.
"""
from pathlib import Path
import json,re,shutil
R=Path(__file__).resolve().parents[1];J=R/'Jounral_PIWM';OUT=R/'reports/corrected_query'
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def main():
    for p in ['corrected_query/audit.json','corrected_query/perception_reaudit.json','corrected_query/checkpoint_spotcheck.json','corrected_initialization/audit.json']:
        assert read('reports/'+p)['status']=='PASS'
    evidence=read('reports/corrected_query/evidence_summary.json');data=evidence['summary'];noise=read('reports/corrected_initialization/summary.json');label=read('reports/consistent_road_labels/manifest.json')
    B=R/'_archive/journal/corrected_query_before';B.mkdir(parents=True,exist_ok=True)
    for n in ['elsarticle-template-num.tex','elsarticle-template-num.pdf','sample-base.bib']:
        if not (B/n).exists():shutil.copy2(J/n,B/n)
    s=(B/'elsarticle-template-num.tex').read_text(encoding='utf-8')
    def number(model,fold='temporal',metric='E100'):return data[fold+'/'+model][metric]['mean']
    def cell(model,fold='temporal',metric='E100'):
        v=data[fold+'/'+model][metric];return '$'+f"{v['mean']:.3f} \\pm {v['sd']:.3f}"+'$'
    abstract=r'''\begin{abstract}
A visual road representation can support interpretable vehicle prediction, but
its geometric consistency and predictive value need not coincide. We extend
PIWM with a finite road preview inferred from a 15-frame image history, queried
at predicted progress, and connected to road-relative state updates and position
reconstruction. The task is offline prediction given an initial physical state
and recorded future actions. Road-relative initialization remains privileged;
rollout readout uses the predicted preview without querying the stored track.
We construct mutually consistent geometric labels from one arc-length curve and
compare independent-head, shared-curve, and unit-speed representations. A fixed
study covers two existing DonkeyCar recordings, three training seeds, a purged
chronological split, both source-transfer directions, retrained query controls,
and a Cartesian model with comparable numerical safeguards.
'''+f'''At 100 steps (approximately 4.55 s), the fixed guarded model obtains
{number('guarded_full'):.3f} m chronological endpoint error, compared with
{number('cartesian_guarded'):.3f} m for bounded Cartesian dynamics and
{number('kinematic'):.3f} m for calibrated kinematics.
'''+r'''The source-transfer results and initialization perturbations delimit this
comparison. Continuous label consistency does not remove errors from sparse
curvature sampling, and the query ablations distinguish geometric recursion from
neural curvature context. The contribution is a specific visual-to-dynamics
interface and its controlled empirical assessment, rather than a new Frenet
identity. Reused recordings on one track support retrospective evidence;
unseen-track generalization and camera-only closed-loop deployment remain
unverified.
\end{abstract}'''
    s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:abstract,s,count=1,flags=re.S)
    start=s.index(r'\section{Related Work}');end=s.index(r'\section{Preliminaries}')
    related=r'''\section{Related Work}\label{sec:related}
\subsection{Physics and learned vehicle prediction}\label{sec:rw_traj}\label{sec:rw_phys}
Embedding vehicle kinematics in a learned predictor is established practice.
Deep Kinematic Models~\cite{cui2020deep} connect learned motion prediction with
explicit vehicle kinematics. PRIME~\cite{song2022prime} combines constrained
model-based candidate generation with a learned trajectory evaluator. PERL~\cite{long2024perl}
adds a learned residual to a physical car-following model. These methods preclude
claiming either a kinematic layer or a physics-plus-residual decomposition as a
new contribution. Our task differs from multimodal behavior prediction: future
steering and throttle are supplied, and the object of study is a compact visual
road interface for recursive ego-vehicle prediction. This task difference does
not establish superior performance over those methods; their published scores
are not directly comparable to our data and information conditions.

\subsection{Road coordinates and geometric representations}\label{sec:rw_po}
Frenet domain normalization~\cite{ye2023frenet} addresses scene-geometry variation,
while Stay on Track~\cite{hallgarten2024stay} wraps motion predictors in
lane-relative coordinates. Thus road alignment and the use of lane geometry as
an inductive bias are not new. GeoWAM~\cite{lu2026geowam} studies a broader visual
geometric scene representation for ego-trajectory prediction. Our narrower
question concerns a finite road preview inferred once from images: how should
its shape, curvature, progress query and Cartesian reconstruction interact?
An independently predicted curvature profile may disagree with the curve used
for readout. A shared curve removes that particular discrepancy, but can impose
a different approximation or optimization cost. The present comparisons test
these tradeoffs under matched inputs rather than claiming a general geometry
or feasibility guarantee.

\subsection{Relation to world models and the conference study}\label{sec:rw_wm}\label{sec:rw_repr}
Latent world models support action-conditioned prediction and imagination-based
control~\cite{ha2018world,hafner2023mastering}. Physical variable alignment offers
a complementary objective: intermediate values have names and units that can
be inspected. The conference PIWM study~\cite{piwm_conf} investigated such
alignment using distribution-based weak supervision and visually encoded
physical states. Those representation-learning principles and benchmark
results are inherited work, not new journal contributions. The road extension
uses direct geometric supervision, a continuous convolutional encoder, and
explicit supplied-action dynamics; it does not experimentally establish weakly
supervised road perception. Its added evidence consists of road-query and
geometric-interface comparisons, matched safeguards, and sensitivity analyses
on physical-car records. Privileged initialization and a single reused track
limit claims about autonomous deployment or independent generalization.

'''
    s=s[:start]+related+s[end:]
    start=s.index(r'\section{Preliminaries}');end=s.index(r'\section{Prediction Task and Information Conditions}')
    prelim=r'''\section{Preliminaries}\label{sec:preliminaries}
For a physical state $x_t$ and action $a_t$, a predictive model approximates
$x_{t+1}=\phi(x_t,a_t)$ with a learned transition and an observation interface.
The conference PIWM framework~\cite{piwm_conf} aligns encoded quantities with
physical variables using distribution-based supervision. Here the driving
extension uses direct road labels and a supplied initial vehicle state; it
predicts physical trajectories rather than future camera images.

We distinguish state semantics from dynamical accuracy. Variables such as
lateral displacement, heading error and curvature have specified units and
coordinate meanings. That specification makes their errors inspectable, but
does not imply that every predicted value is correct or that every transition
is physically realizable. Analytic kinematics supplies an inductive bias;
learned residuals, numerical integration, curve approximation and protective
extensions must be assessed separately. A geometrically consistent readout
means that its curve and coordinate update use compatible derivatives within
the valid chart, not that the predicted road equals the physical road.
The conference weak-supervision noise model is retained in
Appendix~\ref{sec:interp_loss_def}; it is not the supervision used in the new
road-encoder comparisons.

'''
    s=s[:start]+prelim+s[end:]
    old_begin=s.index('The extensions beyond the conference study are:')
    old_end=s.index('Sections~',old_begin)
    contribution=r'''The journal extension has three components. First, it specifies a finite visual
road interface that separates the perceived preview from the road-relative
vehicle state and queries the preview during supplied-action rollout. Second,
it compares independent and shared geometric representations and distinguishes
continuous label consistency from sparse-preview approximation. Third, it tests
query mechanisms and comparable numerical safeguards with retrained controls,
while retaining both chronological and source-transfer results and reporting
initial-state sensitivity. The Frenet equations and residual-learning idea are
established; the novelty claim is the particular interface and its controlled
assessment. Physical alignment and weak-supervision principles from the
conference paper are inherited, not new journal contributions.

'''
    s=s[:old_begin]+contribution+s[old_end:]
    s=s.replace('All six encoders are retrained under this rule.','The corrected-label study trains both encoder families afresh for each of\nthree splits and three seeds (18 encoders). Earlier six-encoder results are\nretained in Appendix~\\ref{sec:development}.')
    insert=s.index(r'\section{Experimental Evaluation}')
    methods=r'''\subsection{Consistent supervision and query controls}\label{sec:corrected_method}
The final controlled study regenerates progress, lateral displacement, heading
error, curvature and local centerline shape from one curve. A periodic cubic
spline passes through the retained smoothed map points, initially parameterized
by chord length. Numerical integration of its tangent norm and inversion of the
arc map yield a true-distance parameter. Current logged position alone is
projected onto this curve; no future vehicle poses enter that projection.
Curvature is computed from first and second derivatives, and all preview labels
use this same curve. Logged speed, causal yaw rate, actions, images and raw
Cartesian evaluation targets are unchanged. The map and its historical
preprocessing are retained; this is not an independent new map acquisition.

For the shared-curve model, Eq.~\ref{eq:consistent_geometry} uses the predicted
Hermite shape and explicit midpoint integration. The coupled encoder instead
predicts curvature knots and generates both positions and tangents from one
piecewise circular curve. Interval curvature is the mean of its two neighboring
knots; exact circular-arc integration gives a unit-speed curve. Its dynamics
uses the same piecewise curvature and midpoint integration, with tangent
extension outside the preview. Continuous supervision does not imply exact
representability by ten 0.5 m-spaced curvature knots. Both families retain the
same perception and dynamics training budgets.

Two controls use the guarded model's identical architecture and are retrained
from scratch. The fixed-query control uses origin curvature in both kinematics
and the neural feature while still progressing along the readout curve; it
intentionally creates a geometric mismatch away from the origin. The static
neural-context control retains the correct progress query in kinematics but
fixes only the neural curvature feature at its origin value. The latter more
directly isolates the value of progress-dependent neural context.

The bounded Cartesian control retains the original Cartesian network and
inputs. It uses the same training-derived speed/yaw-rate increment limits,
wrapped heading and $0.1\tanh(\cdot)$ pose residuals. Cartesian dynamics has two
planar residual coordinates, whereas Frenet dynamics has one lateral residual;
these safeguards are comparable, not identical coordinate constraints. Neural
parameter counts are 13,733 for both Cartesian variants and 14,276 for all three
query variants. Calibrated kinematics has seven parameters. The independent
road encoder is shared within each split/seed; the coupled encoder is trained
separately, so differences between encoder families cannot be attributed solely
to a dynamics constraint.

'''
    s=s[:insert]+methods+s[insert:]
    insert=s.index(r'\subsection{Metrics}')
    protocol=r'''\subsection{Chronological comparison and sensitivity protocol}
The primary split uses ordered segments within each source, with approximately
60/20/20 percent assigned to training, validation and test before purging.
The retained partitions contain 39 training, 13 validation, 15 assigned test and
four purged segments; adjacent partitions have at least 115 original frames of
separation. Fourteen test segments are long enough for the specified context and
horizon, giving 527 windows. Both source-transfer directions remain mandatory
comparisons. The guarded full model and pooled chronological $E_{100}$ are fixed
before fitting; no model family is renamed as the primary method after testing.
All eight dynamics models are trained for each split and seed (72 fits).

The training throttle values are 0.566667 and 0.633333, while later portions of
the second source also contain larger values. The chronological split therefore
is not asserted to be identically distributed. Both records were explored
previously, so even a frozen follow-up protocol remains retrospective.

Initialization sensitivity uses the saved chronological checkpoints for guarded
full, bounded Cartesian and calibrated kinematics, without refitting. Independent
zero-mean Gaussian errors are added separately to $d,\psi,v,\omega$ and jointly.
The smaller standard deviations are 0.02 m, $2^\circ$, 0.02 m/s and $2^\circ$/s;
the larger values are 0.05 m, $5^\circ$, 0.05 m/s and $5^\circ$/s. Three common
noise realizations are shared across models and training seeds. These are fixed
sensitivity scenarios, not measured sensor distributions. Images, preview,
actions and the true ego evaluation frame are unchanged. In particular,
perturbing $\psi$ tests road-relative heading input error, not global-pose
localization error. Noise realizations are averaged within each training seed
before reporting seed dispersion; all separate and joint cases are retained.

'''
    s=s[:insert]+protocol+s[insert:]
    start=s.index(r'\section{Results}');end=s.index(r'\section{Discussion}')
    historical=s[start:end].replace(r'\section{Results}\label{sec:results}',r'\section{Earlier Controlled Studies}\label{sec:development}',1)
    labels={'kinematic':'Calibrated kinematics','cartesian_road':'Cartesian, original','cartesian_guarded':'Cartesian, bounded','guarded_full':'Guarded, dynamic query','guarded_fixed_query':'Guarded, fixed query','guarded_static_context':'Guarded, static neural','geometry_midpoint':'Shared curve, midpoint','arc_full':'Coupled circular arcs'}
    results=r'''\section{Results}\label{sec:results}
\subsection{Corrected geometry and the primary comparison}\label{sec:corrected_results}
All numerical values in this section use the corrected labels and unchanged raw
Cartesian targets. Earlier controlled rounds remain in
Appendix~\ref{sec:development}, and the historical five-fold study remains in
Appendix~\ref{sec:historical}; their different protocols are not pooled.
The road curve has length 9.743901 m instead of the legacy nominal 10.450 m.
Dense curvature integration and direct curve coordinates differ by less than
0.000043 m in the geometric check. However, the prescribed sparse circular-arc
reconstruction has validation coordinate RMSE 0.475 m. Thus correcting the
continuous labels removes their inconsistent definitions but leaves a material
sampling/representation approximation. It does not guarantee a better coupled
predictor.

\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Corrected-label comparison. Mean $\pm$ sample SD across three training seeds; all eight models retained. Errors in meters. Directions A/B hold out the first/second source.}
\label{tab:corrected}
\begin{tabular}{lcccc}\toprule
Model & Chron. $E_{100}$ & Chron. ADE & A $E_{100}$ & B $E_{100}$ \\ \midrule
'''
    for name,lab in labels.items():results+=lab+' & '+' & '.join([cell(name),cell(name,metric='ADE'),cell(name,'outer0'),cell(name,'outer1')])+r' \\'+'\n'
    results+=r'''\bottomrule\end{tabular}
\end{table}

'''
    for control,label_text in [('cartesian_guarded','bounded Cartesian'),('kinematic','calibrated kinematic')]:
        diff=number('guarded_full')-number(control);results+=f"The guarded model's chronological endpoint error is {abs(diff):.3f} m {'lower' if diff<0 else 'higher'} than the {label_text} control. "
    results+='The same model has source-transfer endpoint errors of '+f"{number('guarded_full','outer0'):.3f} and {number('guarded_full','outer1'):.3f} m. "+'These source-transfer results are retained regardless of the chronological ranking.\n\n'
    results+=r'''\subsection{What the query controls identify}
'''
    for control,label_text in [('guarded_fixed_query','fixed query'),('guarded_static_context','static neural context')]:
        p=evidence['paired']['temporal/'+control];direction='lower' if p['mean_difference_m']<0 else 'higher'
        results+=f"Against {label_text}, the dynamic-query model has {abs(p['mean_difference_m']):.3f} m {direction} mean chronological endpoint error and lower error in {p['primary_lower_seeds']} of the three paired training seeds. "
    for fold,direction_name in [('outer0','A'),('outer1','B')]:
        p=evidence['paired'][fold+'/guarded_static_context'];direction='lower' if p['mean_difference_m']<0 else 'higher'
        results+=f"In source-transfer direction {direction_name}, however, its mean endpoint error is {abs(p['mean_difference_m']):.3f} m {direction} than static neural context. "
    results+=r'''These are descriptive comparisons, not independent-session significance tests.
Fixing both curvature uses changes the geometric recursion as well as the
neural feature; it cannot alone establish that neural curvature context is
useful. The static-neural comparison isolates that feature more closely.
A gain over Cartesian dynamics still combines coordinate representation,
query, readout and architecture, and should not be attributed to one component.

'''
    for source in ['traj1_64x64','traj2_64x64']:
        source_name='first' if source.startswith('traj1') else 'second'
        results+=f"On the {source_name} source within the chronological test, guarded full obtains {number('guarded_full',source):.3f} m versus {number('cartesian_guarded',source):.3f} m for bounded Cartesian and {number('kinematic',source):.3f} m for calibrated kinematics. "
    results+='Both source-specific results supplement the pooled metric, whose weights follow the number of overlapping windows.\n\n'
    results+=r'''\subsection{Consistency and initialization sensitivity}
'''
    results+=f"The shared-curve and coupled-arc models obtain {number('geometry_midpoint'):.3f} and {number('arc_full'):.3f} m chronological endpoint error, respectively, compared with {number('guarded_full'):.3f} m for the independent-head guarded model. "
    results+='Their maximum seed fractions of full-step states with a nonpositive Frenet denominator are '+', '.join(f"{100*data['temporal/'+n]['max_seed_invalid_chart_fraction']:.4f}\\%" for n in ['geometry_midpoint','arc_full','guarded_full'])+', respectively. These fractions exclude midpoint internal stages; finite rollouts do not prove chart validity.\n\n'
    results+=r'''\begin{table}[t]
\centering\small
\caption{Chronological initialization sensitivity, joint perturbations. $E_{100}$ in meters; mean and sample SD across training seeds after averaging noise realizations. Individual perturbations and both sources are fully reported in the companion evidence.}
\label{tab:initialization}
\begin{tabular}{lccc}\toprule
Scenario & Kinematic & Bounded Cartesian & Guarded full \\ \midrule
'''
    for scenario,lab in [('clean','Clean'),('small_joint','Smaller joint'),('larger_joint','Larger joint')]:
        cells=[]
        for model in ['kinematic','cartesian_guarded','guarded_full']:
            row=noise[f'pooled/{scenario}/{model}']['E100'];cells.append('$'+f"{row['mean']:.3f} \\pm {row['sd']:.3f}"+'$')
        results+=lab+' & '+' & '.join(cells)+r' \\'+'\n'
    results+=r'''\bottomrule\end{tabular}
\end{table}
These perturbations quantify sensitivity of the specified initial-state inputs.
They neither replace evaluation of a visual state estimator nor establish a
sensor-noise tolerance or closed-loop safety claim. Nonfinite-window counts,
full-step chart monitoring and all individual perturbations are retained in the
companion results. The noise draws and clean predictions are independently
recomputed from the saved checkpoints by the same audit workflow.

\subsection{Reproducibility and retained evidence}
Audits verify the raw/prepared/label hashes, train-only normalizers, all validation
perception losses, validation-minimum checkpoint selection, original-frame
partition gaps and common evaluation windows. Twenty-four uniformly selected
windows per each of 72 dynamics checkpoints are recalculated; changing stored
map buffers leaves the road-model predictions unchanged. This is an internal
recomputation, not external replication. The unchanged kinematic control also
reproduces its preceding chronological results. Per-seed curves, all eight
models, both source-transfer directions and separate initialization scenarios
remain available in the local evidence package.

'''
    s=s[:start]+results+s[end:]
    pos=s.index(r'\appendix')+len(r'\appendix');s=s[:pos]+'\n\n'+historical+s[pos:]
    start=s.index(r'\section{Discussion}');end=s.index(r'\section*{Declaration of Competing Interest}')
    discussion=r'''\section{Discussion}\label{sec:discussion}
The experiments distinguish an interpretable state interface from a guarantee
of physical consistency or predictive accuracy. The implementation makes
progress, road-relative pose, perceived shape and curvature inspectable.
Matched safeguards and retrained query controls narrow possible explanations
for performance, but coordinate choice, residual architecture and reconstruction
still interact. Standard Frenet equations, arc-length reparameterization and
circular-arc integration are established mathematics; the proposed contribution
is their specific connection to a finite visual road preview and the evidence
for its use in supplied-action prediction.

Correcting the geometric labels is scientifically necessary for interpreting
consistency comparisons. It does not make a sparse curvature representation
exact. The integration diagnostic exposes a separate resolution limit, while
shared-curve and coupled-encoder models expose different learning constraints.
Their complete rankings should inform model choice rather than treating a more
constrained representation as automatically better. The neural query comparison must also be read across splits: an advantage in
the chronological setting does not establish an advantage under source/regime
transfer. The fixed guarded family remains the primary comparison; test results
do not select a replacement family.

The principal limitation is two repeatedly used recordings on one track.
Training seeds and overlapping windows do not supply independent environments.
Freezing each follow-up before its fits limits within-round selection but does
not make the overall development process prospective. Source-transfer results
and chronological results answer different questions; neither should be omitted
because it is less favorable. This study does not establish a causal physical
effect of road markings on motion given a complete vehicle state and actions.
Road context may also encode correlations with operating conditions or driving
behavior that the present records cannot disentangle.

Initial road-relative state remains privileged, road labels use a retained map,
and future recorded actions are supplied. Initialization perturbations test
specified errors without a real state estimator. No closed-loop controller,
new track or fresh acquisition validates deployment. Numerical guards can
produce finite states outside the valid coordinate chart, and the midpoint
internal stages are not fully covered by the reported chart monitor. No safety,
general bounded-error or camera-only autonomy guarantee follows. Conference
weak-supervision results concern a different encoder and are not evidence for
weakly supervised road learning in this extension.

\section{Conclusion}\label{sec:conclusion}
We studied a finite visual road-context interface for physically interpretable,
action-conditioned vehicle prediction. The corrected-label comparison retains
a calibrated physical control, bounded and unbounded Cartesian controls,
retrained query ablations and two geometrically coupled alternatives.
'''+f"The fixed guarded model yields {number('guarded_full'):.3f} m chronological endpoint error; the bounded Cartesian and calibrated controls yield {number('cartesian_guarded'):.3f} and {number('kinematic'):.3f} m. "+r'''Source-transfer comparisons, geometric approximation checks and initialization
perturbations delimit these results. The evidence supports a concrete,
inspectable representation and its retrospective empirical assessment, with
independent generalization and deployment left open.

'''
    s=s[:start]+discussion+s[end:]
    (J/'elsarticle-template-num.tex').write_text(s,encoding='utf-8')
    bib=(J/'sample-base.bib').read_text(encoding='utf-8')
    entries={
    'song2022prime':r'''@inproceedings{song2022prime,
 title={Learning to Predict Vehicle Trajectories with Model-based Planning},
 author={Song, Haoran and Luan, Di and Ding, Wenchao and Wang, Michael Y and Chen, Qifeng},
 booktitle={Proceedings of the 5th Conference on Robot Learning},
 series={Proceedings of Machine Learning Research}, volume={164}, pages={1035--1045},
 year={2022}, publisher={PMLR}, url={https://proceedings.mlr.press/v164/song22a.html}
}''',
    'long2024perl':r'''@article{long2024perl,
 title={A Physics Enhanced Residual Learning ({PERL}) Framework for Vehicle Trajectory Prediction},
 author={Long, Keke and Sheng, Zihao and Shi, Haotian and Li, Xiaopeng and Chen, Sikai and Ahn, Sue},
 journal={arXiv preprint arXiv:2309.15284}, year={2024},
 doi={10.48550/arXiv.2309.15284}, url={https://arxiv.org/abs/2309.15284}, note={Version 2}
}'''}
    for key,entry in entries.items():
        if '{'+key+',' not in bib:bib+='\n'+entry+'\n'
    (J/'sample-base.bib').write_text(bib,encoding='utf-8')
    print('Rewritten with all preceding results retained in appendices',flush=True)
if __name__=='__main__':main()
