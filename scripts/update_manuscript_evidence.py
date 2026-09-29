from pathlib import Path
p=Path('scripts/revise_audited_manuscript.py')
s=p.read_text(encoding='utf-8')
needle="conf=block('figure','fig:conf_environments')"
assert needle in s
insert=r'''
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

'''
s=s.replace(needle,insert+needle,1)
s=s.replace('exact geometric\ntraining labels','direct map-derived\ntraining labels').replace('exact geometric\nlabels','direct map-derived\nlabels').replace('exact encoder labels','map-derived encoder labels').replace('uses exact road labels','uses direct road labels')
p.write_text(s,encoding='utf-8')
