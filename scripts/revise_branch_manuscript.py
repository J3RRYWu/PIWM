"""Append audited branch evidence to the current manuscript, preserving its predecessor."""
from pathlib import Path
import json,shutil,numpy as np
R=Path(__file__).resolve().parents[1];J=R/'Jounral_PIWM'
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def main():
    for p in ['curvature_branches/audit.json','road_branch_diagnosis/audit.json','road_branch_diagnosis/local_geometry_checks.json','road_branch_transfer/audit.json']:
        assert read('reports/'+p)['status']=='PASS'
    data=read('reports/curvature_branches/results.json')['results'];diag=read('reports/road_branch_transfer/summary.json')
    B=R/'_archive/journal/curvature_branches_before';B.mkdir(parents=True,exist_ok=True)
    current=J/'elsarticle-template-num.tex'
    if r'\label{sec:branch_factorial}' in current.read_text(encoding='utf-8'):raise RuntimeError('Already inserted; preserve manual edits')
    for n in ['elsarticle-template-num.tex','elsarticle-template-num.pdf','sample-base.bib']:
        if (B/n).exists():raise RuntimeError('Backup exists; inspect instead of overwriting')
        shutil.copy2(J/n,B/n)
    s=current.read_text(encoding='utf-8')
    method=r'''\subsection{Curvature feature allocation and local geometry}\label{sec:branch_factorial}
A follow-up factorial study separates curvature used by the speed/yaw-rate
networks from curvature used by the lateral/heading residual network. Each
branch receives either the dynamic curvature feature or zero in its normalized
curvature channel, corresponding to the training-label mean. Kinematic curvature
queries and the perceived position readout remain unchanged. The three new
variants are trained from scratch for 40 epochs on each of three splits and
three seeds, using the previously frozen independent encoders and normalizers.
The full-curvature model is an inherited reference. Architecture, parameter
count, loss, random-seed initialization, batch order, and validation selection
are matched; all 27 fits finish before evaluation. Initial tests verify bitwise
agreement of the full implementation's outputs and parameter gradients with
the preceding model. Removing curvature does not remove all road context:
$d$ and $\psi$ remain in the learned features.

The geometry also clarifies what a coordinate change can explain. On a smooth,
locally unique chart of a shared curve $P(q)$, let $g=\|P'(q)\|$ and
$h=1-d\kappa>0$, with tangent $\mathbf t$, normal $\mathbf n$, and road
heading $\theta$. In the continuous, zero-pose-residual, unclamped field,
\begin{equation}
\dot{\mathbf x}=gh\dot q\,\mathbf t+\dot d\,\mathbf n
=v\cos\psi\,\mathbf t+v\sin\psi\,\mathbf n,
\qquad \dot\alpha=\omega,\quad \alpha=\theta+\psi.
\label{eq:local_coordinate_identity}
\end{equation}
Thus consistent coordinates alone do not add a vehicle motion law. Learned
features, residual directions, finite-step integration and guards can change
predictions. Holding the state fixed gives the local partial derivatives
\begin{equation}
\frac{\partial\dot q}{\partial\kappa}=\frac{v\cos\psi\,d}{gh^2},\qquad
\frac{\partial\dot q}{\partial g}=-\frac{v\cos\psi}{g^2h},\qquad
\frac{\partial\dot\psi}{\partial\kappa}=-\frac{v\cos\psi}{h^2}.
\label{eq:local_curvature_sensitivity}
\end{equation}
These explain coordinate sensitivity near degeneracy, not a lower bound on
Cartesian error: consistent readout can cancel coordinate effects. Shared-curve
$g$ and $\kappa$ are correlated, and learned derivatives add further terms.
The identities exclude curve knots, guard transitions and ambiguous projections.
Actual-field/readout automatic-differentiation checks on 128 smooth-interior
float64 states agree with Eq.~\ref{eq:local_coordinate_identity} within
$1.8\times10^{-15}$; the partial derivatives agree within $4.5\times10^{-16}$.
This is a code check of standard geometry, not a new theorem or global stability
guarantee. Learned residuals are fixed-step increments; changing the step size
without rescaling and redefining them is not a convergence experiment.

'''
    anchor=r'\section{Experimental Evaluation}'
    assert s.count(anchor)==1;s=s.replace(anchor,method+anchor,1)
    def cell(name,fold):
        v=[data[f'{fold}/seed{i}/{name}']['E100'] for i in range(3)]
        return '$'+f'{np.mean(v):.3f} \\pm {np.std(v,ddof=1):.3f}'+'$'
    def mean(name,fold):return np.mean([data[f'{fold}/seed{i}/{name}']['E100'] for i in range(3)])
    result=r'''\subsection{Which learned branch uses curvature?}\label{sec:branch_results}
Table~\ref{tab:branch_factorial} isolates curvature feature allocation while
preserving dynamic kinematic queries and the original geometry. Every new
variant has 14,276 parameters. This retrospective follow-up uses the same
records; neither the factorial design nor repeated seeds create an independent
evaluation population.
\begin{table}[htbp]
\centering\small
\caption{Curvature feature factorial. Endpoint error in meters, mean and sample SD over three seeds. All geometry queries remain dynamic.}
\label{tab:branch_factorial}
\begin{tabular}{lccc}\toprule
Neural curvature features & Chronological & Source A & Source B \\\midrule
'''
    labels=[('guarded_full','Both branches (inherited)'),('curvature_response_off','Pose residual only'),('curvature_pose_off','Speed/yaw response only'),('curvature_both_off','Neither learned branch')]
    for name,label in labels:result+=label+' & '+' & '.join(cell(name,f) for f in ['temporal','outer0','outer1'])+r' \\'+'\n'
    result+=r'''\bottomrule\end{tabular}
\end{table}
'''
    result+=f"Removing curvature from both learned branches changes chronological error from {mean('guarded_full','temporal'):.3f} to {mean('curvature_both_off','temporal'):.3f} m, and source A/B errors from {mean('guarded_full','outer0'):.3f}/{mean('guarded_full','outer1'):.3f} to {mean('curvature_both_off','outer0'):.3f}/{mean('curvature_both_off','outer1'):.3f} m. "
    result+=r'''Against the no-neural-curvature control, the full model reduces mean chronological
endpoint error by only 3.3 percent, improves two of three paired seeds and eight
of fourteen segments. The earlier 9.9 percent comparison against static neural
curvature measures a different contrast; neither establishes that neural
curvature is essential for the large gain over Cartesian dynamics. Removing
pose-branch curvature improves source A in all three seeds, while removing
response-branch curvature improves source B in all three seeds. No new variant
beats calibrated kinematics in mean endpoint error in either transfer direction.
These comparisons identify input-allocation effects under the fixed training
procedure, not a causal influence of the road on physical actuation. The
companion results retain all paired seeds, per-source chronological results,
segment comparisons and the descriptive factorial interaction. They do not
select a replacement primary model by its test score.

\paragraph{No-refit diagnostic of transfer.}
A separately frozen intervention replaces both perceived road outputs by their
exact label counterparts, without refitting weights or changing initialization,
actions or raw-position truth. It is privileged diagnostic input, not a deployed
result or an oracle-trained upper bound. In Table~\ref{tab:branch_oracle},
all three seeds and both transfer directions are retained.
\begin{table}[htbp]
\centering\small
\caption{No-refit exact-road diagnostic for the fixed guarded model. Endpoint errors in meters.}
\label{tab:branch_oracle}
\begin{tabular}{lcc}\toprule
Split & Perceived road & Exact road \\\midrule
'''
    for f,label in [('temporal','Chronological'),('outer0','Source A'),('outer1','Source B')]:
        cells=[]
        for mode in ['predicted','exact']:
            vals=[diag[f'{f}_seed{i}']['guarded_full/'+mode]['E100'] for i in range(3)]
            cells.append('$'+f'{np.mean(vals):.3f} \\pm {np.std(vals,ddof=1):.3f}'+'$')
        result+=label+' & '+' & '.join(cells)+r' \\'+'\n'
    result+=r'''\bottomrule\end{tabular}
\end{table}
Exact-road input does not close the fixed guarded model's mean transfer gap to
calibrated kinematics: its source A/B errors remain 0.993/1.222 m versus
0.936/1.158 m. Thus this substitution alone is insufficient to remove the gap.
All diagnostic group boundaries are obtained from training partitions.
Action support means the training coordinate-wise steering/throttle ranges;
falling inside those ranges does not establish joint-distribution coverage.
In both source-held-out directions, every evaluated window contains at least
one action outside these ranges. These directions therefore confound recording
transfer with operating-regime extrapolation, rather than isolating visual
shift. This observation does not identify either factor's causal contribution.
Road-error, speed and action-support groupings are descriptive associations.
Exact-road substitutions may themselves change the input distribution and
cannot uniquely separate perception error from model mismatch.

Both the initial and midpoint vector-field evaluations of the shared-curve
model are now monitored in the supplemental diagnostic. Guard activation,
nonpositive $1-d\kappa$, small curve speed and preview extrapolation are counted
separately; failing windows remain in the scores. This supplements, rather than
redefines, the earlier full-step fractions. All 27 new selected checkpoints and
nine inherited full models are independently recomputed on 24 fixed windows;
all measured metrics, fingerprints and validation minima are checked. This
remains internal auditing, not external replication.

'''
    anchor=r'\section{Discussion}'
    assert s.count(anchor)==1;s=s.replace(anchor,result+anchor,1)
    s=s.replace('The final controlled study regenerates','The corrected-label study regenerates',1)
    s=s.replace('produce finite states outside the valid coordinate chart, and the midpoint\ninternal stages are not fully covered by the reported chart monitor. No safety,',
                'produce finite states outside the valid coordinate chart. Supplemental monitoring\nnow includes midpoint internal stages; nonpositive denominators still occur. No safety,',1)
    s=s.replace('retrained query ablations and two geometrically coupled alternatives.',
                'retrained query and curvature-branch ablations and two geometrically coupled alternatives.',1)
    s=s.replace('evaluation population.\n\\begin{table}', 'evaluation population.\n\n\\begin{table}')
    s=s.replace('all three seeds and both transfer directions are retained.\n\\begin{table}', 'all three seeds and both transfer directions are retained.\n\n\\begin{table}')
    s=s.replace('\\end{table}\nRemoving curvature','\\end{table}\n\nRemoving curvature')
    s=s.replace('\\end{table}\nExact-road input','\\end{table}\n\nExact-road input')
    current.write_text(s,encoding='utf-8')
    print('Updated manuscript; predecessor preserved at',B)
if __name__=='__main__':main()
