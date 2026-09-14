# Aim 1: establishment and repair of normal organization

**Historical homogeneous baseline:** this page describes the original three-field control, whose fields converge to a flat state. The revised spatial aim is renewing-cell niches and connected bands; see [SPATIAL_AIM1.md](SPATIAL_AIM1.md). Passing the tests below does not establish that spatial architecture.

## Scientific question and claim boundary

Aim 1 asks for a minimal set of local feedback rules that supports a stable organized tissue, ongoing turnover, and recovery from a finite disturbance under one common normal parameterization. It must distinguish organization established from diverse initial conditions from maintenance of an imposed pattern.

The first executable model treats “normal organization” narrowly as a spatially uniform, bounded density and lineage composition on a closed spherical surface. This is a useful baseline because it tests homeostatic establishment and repair without a target map. It does **not** yet establish layered, polarized, or otherwise nonuniform anatomical organization. Those structures will require additional local state variables or boundary cues and new establishment tests.

The model is implemented separately in `tissue_growth.organization`; it does not alter the Schnakenberg pattern demonstration. Its normal parameterization is stored in `configs/aim1_normal.json`.

## Biological motivation and abstraction

The rules are motivated by three broad observations rather than assigned to a specific tissue:

- Lineage models show that feedback from downstream cells can regulate upstream proliferation and allow cell number and lineage fraction to recover through partly distinct controls ([PMID 19166268](https://pubmed.ncbi.nlm.nih.gov/19166268/), [PMID 29470992](https://pubmed.ncbi.nlm.nih.gov/29470992/), [PMID 30897261](https://pubmed.ncbi.nlm.nih.gov/30897261/)).
- Epithelial crowding can induce live-cell extrusion and buffer variation in cell number ([10.1038/nature10999](https://doi.org/10.1038/nature10999); [PMID 22504183](https://pubmed.ncbi.nlm.nih.gov/22504183/)).
- Mechanical stretch can stimulate epithelial division through Piezo1, providing a concrete example of low-density and high-density responses contributing to homeostasis ([10.1038/nature21407](https://doi.org/10.1038/nature21407); [PMID 28199303](https://pubmed.ncbi.nlm.nih.gov/28199303/)).

These sources motivate feedback signs, not the numerical parameter values or universal molecular identities. The first model is an abstract renewing tissue, as selected for this project stage. “Niche,” “mechanical feedback,” and “differentiated feedback” are operational model variables, not measurements of a particular pathway.

## State variables and local rules

The fields are renewing-cell density $R$, differentiated-cell density $D$, and permissive niche signal $N$. All are dimensionless. The local mechanical proxy is occupancy

$$
C=R+D.
$$

There is no prescribed target density in a reaction term. The carrying capacity $K$ sets the scale at which crowding suppresses renewal and activates extrusion; it is a scalar local material property, not a spatial target map.

The local fluxes are

$$
P=\beta[(1-w_N)+w_NN]R\max\left(0,1-w_M\frac{C}{K}\right),
$$

$$
F=\kappa R,\qquad
E=\eta w_M\max(0,C-K),
$$

where $P$ is renewing-cell proliferation, $F$ is differentiation, and $E$ is crowding-induced extrusion. Extrusion is divided between renewing and differentiated cells in proportion to their local densities. The feedback weights $w_N$ and $w_M$ equal one in the normal parameterization and are set to zero only in mechanism controls.

The surface equations on the fixed unit sphere are

$$
\partial_tR=D_R\Delta_\Gamma R+P-F-\mu_RR-E\frac{R}{C},
$$

$$
\partial_tD=D_D\Delta_\Gamma D+F-\mu_DD-E\frac{D}{C},
$$

$$
\partial_tN=D_N\Delta_\Gamma N+\alpha(1-N)-\gamma DN.
$$

The $E/C$ factors are defined as zero at zero occupancy. Differentiated cells suppress the niche through $\gamma DN$; the niche recovers locally toward one. The niche promotes renewing-cell proliferation, vacancy promotes it through the mechanical factor, and overcrowding removes cells. Differentiation and both turnover terms remain active at homeostasis.

All reaction terms depend only on the current local fields. Coordinates enter only when generating an initial condition or applying a lesion. Surface diffusion couples neighboring locations but does not compare the state with a stored pattern.

## Numerical method

The model uses the same closed triangular sphere and row-sum-lumped P1 mass matrix as the reference backend. Reactions are explicit and surface diffusion is implicit:

$$
(M_\ell+hD_iK)c_i^{n+1}=M_\ell[c_i^n+hQ_i(c^n)].
$$

The normal protocol uses `n=24`, `dt=0.02`, and the SciPy reference operator. A trial step is halved if the explicit source or solved state is nonfinite or negative. No clipping is applied. Accepted steps accumulate spatially averaged proliferation, differentiation, lineage loss, and extrusion fluxes. This makes ongoing turnover observable even when population means are stationary.

Aim 1 currently holds geometry fixed. The reported net population source is a candidate input for a later mechanically determined growth law; it does not yet change sphere radius. This separation keeps establishment and repair tests from being confounded by prescribed geometric growth.

## Common normal parameterization

| Rule | Parameter | Value |
|---|---|---:|
| Renewing diffusion | $D_R$ | 0.003 |
| Differentiated diffusion | $D_D$ | 0.001 |
| Niche diffusion | $D_N$ | 0.01 |
| Renewal | $\beta$ | 1.4 |
| Differentiation | $\kappa$ | 0.30 |
| Renewing turnover | $\mu_R$ | 0.03 |
| Differentiated turnover | $\mu_D$ | 0.18 |
| Niche recovery | $\alpha$ | 0.60 |
| Differentiated feedback | $\gamma$ | 0.80 |
| Carrying-capacity scale | $K$ | 1.0 |
| Crowding extrusion | $\eta$ | 2.0 |
| Niche/mechanical feedback weights | $w_N,w_M$ | 1.0, 1.0 |

These values are provisional and dimensionless. They are used unchanged for every normal establishment, turnover, and repair condition. They were chosen to create a numerically tractable positive homeostasis, not fitted to biological data.

## Preregistered evidence protocol

Run the reference protocol with:

```bash
tissue-growth aim1 --config configs/aim1_normal.json --output runs/aim1_normal.json
```

The command returns exit code zero only when all required evidence thresholds pass. It runs these experiments:

1. **Establishment:** low density, overcrowding, a hemispheric density/niche mosaic, lineage segregation, and a random smooth state evolve for 80 time units. Initial mean occupancies span more than eightfold. Every final occupancy coefficient of variation must be below 0.002; final mean occupancy and differentiated fraction must each agree across starts within 1%.
2. **Turnover:** a separately established tissue evolves for 20 additional time units. Integrated proliferation, differentiation, and differentiated-cell loss must each exceed 0.1 while mean occupancy drifts by less than 1%.
3. **Repair:** the same established state receives three Gaussian cell-depletion lesions: `(angular radius, central fraction) = (0.20,0.40), (0.35,0.60), (0.50,0.80)`. Both cell populations are depleted while niche is left unchanged. After 60 time units, occupancy must return within 1% of its pre-lesion mean, its coefficient of variation must be below 0.01, and the spatial defect must fall by at least 90%.
4. **Maintenance control:** an arbitrary latitudinal occupancy pattern is imposed only at initialization. Its spatial coefficient of variation must fall by at least 90%, and its final mean must match the common established mean within 1%. Decay demonstrates that the model does not preserve arbitrary supplied patterns or use them as restoring targets.

Thresholds are computed from density, composition, and fluxes. No pixel-wise comparison to a target field appears in the model or acceptance rules.

## Reference result

The full `n=24` protocol completed in 94.3 seconds in the development environment and passed every required criterion. The complete machine-readable result is [aim1-reference.json](aim1-reference.json).

| Initial condition | Initial mean occupancy | Settled mean occupancy | Settled differentiated fraction | Settled occupancy CV |
|---|---:|---:|---:|---:|
| Low density | 0.140 | 0.638806 | 0.625000 | 5.53e-9 |
| Overcrowded | 1.200 | 0.638806 | 0.625000 | 1.22e-8 |
| Hemispheric mosaic | 0.604 | 0.638806 | 0.625000 | 1.48e-7 |
| Lineage segregated | 0.600 | 0.638806 | 0.625000 | 1.93e-7 |
| Random smooth | 0.580 | 0.638806 | 0.625000 | 1.76e-8 |

Over the 20-unit homeostatic turnover interval, integrated mean proliferation was 1.5810, differentiation was 1.4373, renewing loss was 0.1437, and differentiated loss was 1.4373. Mean occupancy remained within the 1% drift criterion. The units are integrated dimensionless density flux, not cell counts.

| Lesion radius / central depletion | Post-lesion occupancy CV | Recovered mean occupancy | Recovered occupancy CV |
|---|---:|---:|---:|
| 0.20 / 0.40 | 0.0394 | 0.638806 | 1.32e-7 |
| 0.35 / 0.60 | 0.1015 | 0.638806 | 5.16e-7 |
| 0.50 / 0.80 | 0.1906 | 0.638805 | 1.12e-6 |

The imposed-pattern control began with occupancy CV 0.2970 and ended at 1.09e-8 while converging to the same mean occupancy. This result rejects maintenance of that arbitrary imposed pattern under the current rules; it does not prove that every possible imposed pattern must decay.

## Mechanism controls and H1

Three one-parameter controls set differentiated feedback, niche coupling, or mechanical feedback to zero. The remaining parameters are unchanged. Each control is established, receives the middle lesion, and is allowed to recover. These controls are causal probes, not alternative fitted normal parameterizations, and do not contribute to the common-normal pass/fail result.

The working prediction is that spatial smoothing can survive some feedback removals because diffusion and remaining local reactions still act, while the regulated population and lineage mixture shift. Mechanical-feedback loss is expected to permit severe overgrowth. Thus population regulation and spatial restoration are reported separately. Agreement with this prediction supports H1 only within this abstract model; it does not identify a biological pathway.

The reference controls matched that prediction. Removing differentiated feedback or disconnecting niche signal from renewal increased settled occupancy by 19.6%, yet the middle lesion still smoothed to occupancy CV below 2e-9. Removing mechanical feedback increased settled occupancy 6.10-fold; after lesion repair its occupancy remained 6.09-fold above normal even though spatial CV fell below 0.001. The result shows that low spatial variation is insufficient evidence of normal population control.

## Interpretation and next decisions

Passing the protocol supports the narrow statement that one target-free set of local rules establishes a uniform renewing tissue, maintains nonzero lineage turnover, and repairs finite depletion on this mesh and time step. It does not establish parameter identifiability, molecular validity, stochastic clonal behavior, or nonuniform anatomy.

Before moving to neoplasia, the normal evidence should be extended in this order:

1. repeat the protocol across seeds, `dt`, and `n`;
2. determine a maximum recoverable lesion size and recovery-time scaling without retuning parameters;
3. add clonal labels that are dynamically neutral, so turnover can be separated from population stationarity;
4. choose a tissue-specific architecture and establish it from diverse starts using local polarity or boundary cues;
5. couple net population source and mechanical state to geometry only after the fixed-domain controls are stable.
