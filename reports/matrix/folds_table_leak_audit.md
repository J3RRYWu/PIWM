# v6 encoder leak audit

```
v6 LEAK AUDIT (fold 0)
  clean (v6 never saw): [5, 8, 30, 50, 64, 69, 70]
  seen  (v6 trained on): [4, 16, 19, 35, 36, 53, 66]

-- subset clean --
fold 0: fold 0/5: 57 train / 14 val eps, val=[4, 5, 8, 16, 19, 30, 35, 36, 50, 53, 64, 66, 69, 70]
  5 rows x 201 windows

-- subset seen --
fold 0: fold 0/5: 57 train / 14 val eps, val=[4, 5, 8, 16, 19, 30, 35, 36, 50, 53, 64, 66, 69, 70]
  5 rows x 446 windows

======================================================================
E_xy@100 (m) by v6 exposure, fold 0, delta=0
======================================================================
model                 clean       seen   seen-clean   uses v6?
----------------------------------------------------------------------
V2P                   0.416      0.537       +0.121   YES (theta)
GOKU                  0.569      0.622       +0.054   no
DVBF                  0.643      0.648       +0.005   no
ours-a                0.416      0.462       +0.045   no
ours-c                0.457      0.514       +0.057   no

the actual test -- V2P's edge over each encoder-free model, per subset:
(if the leak helps V2P, its edge must be LARGER on 'seen')
----------------------------------------------------------------------
  V2P - GOKU    clean -0.153   seen -0.085   shift +0.067m (against V2P)
  V2P - DVBF    clean -0.227   seen -0.111   shift +0.116m (against V2P)
  V2P - ours-a  clean -0.000   seen +0.075   shift +0.076m (against V2P)

  mean shift +0.086m -- exposure buys V2P nothing measurable; the leak is inert
```
