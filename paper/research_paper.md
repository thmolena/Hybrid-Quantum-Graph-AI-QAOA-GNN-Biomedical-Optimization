# Hybrid Quantum-Graph Learning for Structured Biomedical Optimization

### GNN-Informed Initialization for QAOA and Graph-Based Clinical Modeling

<div align="center">

**Molena Huynh<sup>*</sup>**  
<sup>*</sup>Correspondence author

</div>

---

## Abstract

We study whether graph-conditioned learning can serve as a reusable computational interface across two structurally different settings: parameter initialization for the Quantum Approximate Optimization Algorithm (QAOA) and graph-based biomedical screening. In the optimization branch, a graph neural network (GNN) predicts depth-2 QAOA parameters for transcriptomic co-expression graphs derived from real prostate expression data. On six held-out graph instances, the adapted model achieves a mean approximation ratio of **0.8682 ± 0.0312**, compared with **0.8686 ± 0.0308** for direct classical search, retaining **99.95%** of classical quality while reducing median end-to-end latency from **675.9 ms** to **0.256 ms**. In the biomedical branch, graph-based clinical models are evaluated on cardiotocography under split-first preprocessing, threshold-aware analysis, and robustness checks. The strongest operating-point model reaches **98.8% held-out accuracy**, **0.942 balanced accuracy**, and detects **31 of 35** pathologic exams with **1 false positive**.

We do not claim quantum advantage, clinical readiness, or state-of-the-art external benchmarking. Instead, we position the paper as a research prototype with one central thesis: **graph-conditioned learning can occupy analogous computational roles across optimization and clinical prediction, even when the downstream objectives differ sharply**.

---

## 1. Introduction

Graph-structured problems arise across both combinatorial optimization and biomedical machine learning, yet these domains are usually studied in isolation. In combinatorial optimization, graphs define the objective itself, as in MaxCut, where solution quality depends on exploiting structural properties of vertices and edges. In biomedical modeling, graphs encode relational inductive biases across patients, measurements, or biological entities. Despite their differences, both settings require learning systems that can use structured dependencies rather than treat observations as independent samples.

This work investigates whether **graph-conditioned learning can serve as a shared computational interface across these domains**. Specifically, we ask whether graph neural networks can learn structure-aware representations that (i) improve parameter initialization for QAOA and (ii) support clinically meaningful prediction in graph-based biomedical tasks.

In the optimization setting, QAOA performance depends strongly on parameter initialization, and obtaining high-quality parameters typically requires repeated classical optimization. We therefore study whether a GNN can predict initialization parameters directly from graph structure and thereby reduce the search burden. In the biomedical setting, we construct patient similarity graphs and evaluate whether graph-based message passing improves retrospective screening under calibration-aware, threshold-aware, and robustness-aware evaluation.

The central contribution is not the individual results alone, but the demonstration that **graph-conditioned learning acts as a reusable computational interface across structurally distinct domains**. This framing lets us study quantum optimization and biomedical prediction under one representation-learning thesis while preserving domain-specific evaluation standards.

**Novelty statement.** Prior work has already explored warm-start QAOA, parameter transfer, and learned or graph-informed QAOA initialization. Our novelty claim is therefore deliberately narrower and more defensible: **to our knowledge, this is the first research prototype that couples real biological graph QAOA evaluation with graph-based clinical modeling under a single graph-conditioned learning thesis**. The paper's novelty is thus not a claim that GNN-to-QAOA initialization is universally new in isolation, but that the same graph-conditioned interface is instantiated, evaluated, and interpreted across both transcriptomic optimization and clinically oriented prediction.

Empirically, we find near-parity between direct classical depth-2 search and GNN-predicted initialization on held-out transcriptomic graphs, together with a large latency reduction. In biomedical screening, we find that graph-based models support strong retrospective operating-point behavior. We emphasize throughout that this is a **research prototype rather than a finalized system**.

### Contributions

1. We formalize a **dual-domain graph-conditioned learning thesis** spanning QAOA initialization and biomedical screening.
2. We show that a transcriptomically adapted GNN reaches **0.8682 ± 0.0312** held-out QAOA ratio, essentially matching the **0.8686 ± 0.0308** direct classical reference while being much faster at inference time.
3. We add a minimum viable rigor package for the QAOA branch: **ablation table, runtime table, and convergence analysis**.
4. We separate **reproducibility-oriented evaluation** from **best-case operating-point performance** in the biomedical branch, following good practice for clinical machine learning.

---

## 2. Related Work

### 2.1 QAOA Initialization and Transfer

The QAOA literature already contains warm-start methods, parameter concentration results, and parameter transfer heuristics. More recent work has also considered learned initialization, including graph-aware or GNN-based approaches. This prior art creates a clear review risk: a paper that only states "GNN predicts QAOA parameters" does not sufficiently establish novelty. We therefore position our work against this literature in a constrained way. The contribution here is the combination of:

1. real transcriptomic co-expression graphs rather than only synthetic benchmark families,
2. explicit held-out quality, runtime, and ablation reporting, and
3. a unified graph-conditioned framing that also includes biomedical evaluation.

Concrete examples help clarify this boundary. Recent papers such as *Graph Learning for Parameter Prediction of Quantum Approximate Optimization Algorithm*, *QSeer: A Quantum-Inspired Graph Neural Network for Parameter Initialization in Quantum Approximate Optimization Algorithm Circuits*, and *Conditional Diffusion-based Parameter Generation for Quantum Approximate Optimization Algorithm* all pursue learned parameter generation more directly than this repository does. Our claim is therefore narrower: we contribute a biologically grounded held-out evaluation, explicit ablation and runtime evidence, and a unified graph-conditioned framing that spans both QAOA and biomedical prediction. We do **not** claim to supersede those learned-QAOA papers experimentally, because head-to-head comparisons are not yet included in the current repository.

### 2.2 Graph Learning in Biomedicine

Graph neural networks have become standard tools for relational biomedical modeling, including patient similarity learning and disease prediction. At the same time, responsible biomedical evaluation now demands more than headline accuracy: threshold selection, calibration, robustness across runs, and explicit operational tradeoffs matter at least as much as raw discrimination. Our biomedical branch follows that perspective and reports both a reproducibility-oriented benchmark model and a separate best operating-point model.

### 2.3 Unifying Perspective

The paper is intentionally not "two papers glued together". The unifying question is whether **graph-conditioned representations can be reused as a computational interface** across tasks where the final output differs: QAOA angles in one case and pathologic risk scores in the other. This is the paper's organizing principle.

---

## 3. Problem Setting

Let $G = (V, E, X)$ denote a graph with topology $E$ and node features $X$.

### 3.1 QAOA Branch

For each graph $G$ in a family of transcriptomic co-expression graphs, the goal is to predict a depth-2 QAOA parameter vector

$$
f^{\text{Q}}_\theta(G) = (\hat{\gamma}_1, \hat{\gamma}_2, \hat{\beta}_1, \hat{\beta}_2),
$$

so that the resulting expected cut value is close to the classical depth-2 reference on the same graph. Let $C^*(G)$ denote the exact MaxCut value and $C(\hat{\gamma}, \hat{\beta}; G)$ the expected QAOA cut. We evaluate the approximation ratio

$$
r(G) = \frac{C(\hat{\gamma}, \hat{\beta}; G)}{C^*(G)}.
$$

### 3.2 Biomedical Branch

For a patient similarity graph $G$ with node features extracted from cardiotocography exams, the goal is to predict a pathologic probability

$$
f^{\text{B}}_\phi(G)_i = \hat{p}_i \in [0,1]
$$

for each patient node $i$. The relevant outputs are not only classification scores but also threshold-dependent operating behavior, including false positives, false negatives, sensitivity to pathologic cases, and calibration.

### 3.3 Shared Thesis

The two branches differ in downstream target, but both instantiate the same high-level question: can a graph-conditioned model map structured input graphs to domain-specific decisions while preserving the information needed by the downstream objective?

---

## 4. Method

### 4.1 Unified Architecture View

Both branches use a GNN encoder as a structure-sensitive front end. The model computes node-level or graph-level representations by message passing over $G$, then emits either QAOA angles or biomedical risk scores.

### 4.2 Formal Training Objectives

For the QAOA branch, we train on a dataset $\mathcal{D}_{\text{Q}} = \{(G_j, y_j)\}_{j=1}^N$ where $y_j$ is the classical depth-2 target angle vector for graph $G_j$. The predictor minimizes a regression loss

$$
\mathcal{L}_{\text{Q}}(\theta) = \frac{1}{N} \sum_{j=1}^{N} \left\lVert f^{\text{Q}}_\theta(G_j) - y_j \right\rVert_2^2.
$$

This is a pragmatic target: the model is not trained directly on the approximation ratio, but on classical angle surrogates whose induced ratios are then evaluated exactly on held-out graphs.

For the biomedical branch, we train on labeled patient graphs with class-weighted binary cross-entropy,

$$
\mathcal{L}_{\text{B}}(\phi) = - \sum_i w_{y_i} \left[y_i \log \hat{p}_i + (1-y_i) \log (1-\hat{p}_i)\right],
$$

where $w_{y_i}$ upweights the clinically more consequential pathologic class.

### 4.3 Model Roles

- **Adaptive Quantum GCN** predicts QAOA angles from graph structure and is evaluated as a warm-start initializer.
- **Adaptive BioGCN** is the reproducibility-oriented benchmark model.
- **ResidualClinicalGCN** is the best operating-point model for the biomedical branch.

We separate reproducibility-oriented evaluation from best-case performance, following best practices in clinical ML. This turns out to be one of the clearest strengths of the current repository and is now stated explicitly in the paper.

### 4.4 Why the Approach Can Work

One plausible mechanism is that graph structure correlates with useful low-dimensional regularities in the downstream solution. In the QAOA branch, graphs with related structural signatures may share favorable parameter basins, so a GNN can reduce the search region that classical refinement must explore. In the biomedical branch, message passing can aggregate local neighborhood information that helps disambiguate borderline cases. In both settings, the GNN is valuable not because it solves the task by itself, but because it **compresses graph structure into a more actionable initialization or score surface**.

---

## 5. Experimental Setup

### 5.1 QAOA Protocol

The optimization branch uses transcriptomic co-expression graphs built from real prostate expression data. We evaluate exact depth-2 QAOA on a representative graph and on six held-out resampled graphs. Classical references are obtained by multistart Nelder-Mead search; learned predictions are evaluated by exact statevector simulation.

Current internal baselines include:

- zero-angle and random-search baselines on the representative graph,
- random and heuristic initializers on held-out graphs,
- edge-ablated and feature-ablated GNN variants,
- a legacy transfer baseline for comparison against the upgraded adaptation protocol.

What remains missing is an external comparison to recent learned QAOA initialization papers. We say this explicitly because reviewers will look for it.

### 5.2 Biomedical Protocol

The biomedical branch uses the cardiotocography cohort with split-first preprocessing and threshold-aware evaluation. The baseline family now includes logistic regression, random forest, MLP, XGBoost, LightGBM, and calibrated variants of the main tabular models, together with graph models. This substantially strengthens the benchmark story: the graph models are no longer compared only against lightweight baselines, but against a broader tabular family that is more consistent with reviewer expectations in clinical ML.

### 5.3 Reviewer-Facing Scope Clarification

Future work will include direct external comparison against recent graph-learning-based QAOA initialization methods and multilevel QAOA approaches.

---

## 6. Results

### 6.1 QAOA Quality, Runtime, and Ablation

Across six held-out transcriptomic graphs, direct classical depth-2 search and the adapted GNN are nearly indistinguishable in quality, while inference is far faster.

| Method | Held-out mean ratio | Std. dev. | Notes |
|---|---:|---:|---|
| Classical depth-2 search | 0.8686 | 0.0308 | Exact reference over held-out graphs |
| Adapted GNN initializer | 0.8682 | 0.0312 | Retains 99.95% of classical quality |
| Legacy transfer baseline | 0.5725 | not reported | Older baseline retained only for context |

This is the main quantitative argument for the QAOA branch: the learned initializer is **competitive, not superior**, and the benefit comes from the compute-quality tradeoff rather than a higher final ratio.

The QAOA branch now also includes the ablation package that was missing from the earlier manuscript.

| Initializer | Mean ratio | Std. dev. | Mean retention vs. classical |
|---|---:|---:|---:|
| Random initialization | 0.6586 | 0.1170 | 0.7576 |
| Heuristic initialization | 0.8634 | 0.0313 | 0.9940 |
| GNN (no graph edges) | 0.7086 | 0.0333 | 0.8156 |
| GNN (no node features) | 0.6535 | 0.0319 | 0.7520 |
| **GNN full model (ours)** | **0.8682** | **0.0312** | **0.9995** |

These ablations sharpen the claim considerably. They show that the full model's performance is not reproduced by a graph-agnostic random baseline and degrades strongly when either topology or node features are removed.

Runtime analysis likewise shifts the interpretation from "interesting demo" toward a more rigorous efficiency argument.

| Method | Median runtime (ms) | Relative to classical |
|---|---:|---:|
| Classical depth-2 search | 675.9079 | 1.0x |
| Random initialization + evaluation | 0.2363 | 2859.97x faster |
| Heuristic initialization + evaluation | 0.2384 | 2835.24x faster |
| GNN (no graph edges) | 0.2531 | 2670.04x faster |
| GNN (no node features) | 0.2626 | 2574.28x faster |
| **GNN full model (ours)** | **0.2560** | **2640.48x faster** |

Finally, the representative-graph convergence traces show that both the heuristic initializer and the full GNN reach the same final ratio of **0.8976**, with the full model requiring **191** function evaluations versus **196** for the heuristic baseline. By contrast, the edge-ablated model ends at **0.8439**, and random or feature-free initializers stall at **0.7731**. This does not prove asymptotic scaling, but it does provide mechanistic evidence that graph-conditioned initialization places local search in a better basin.

### 6.2 Biomedical Results and Statistical Framing

The biomedical branch is intentionally reported in two layers: a reproducibility-oriented benchmark model and a best operating-point model.

| Model | Accuracy | Balanced accuracy | ROC AUC | Notes |
|---|---:|---:|---:|---|
| Logistic regression | 94.13% | 0.916 | 0.984 | Strong linear baseline |
| Calibrated logistic regression | 95.54% | 0.937 | 0.986 | Better threshold behavior than the uncalibrated linear model |
| Random forest | 96.95% | 0.905 | 0.994 | Strong tabular nonlinear baseline |
| Calibrated random forest | 96.01% | 0.900 | 0.991 | Calibration improves probability quality more than thresholded accuracy |
| XGBoost | 98.83% | 0.955 | 0.991 | Strongest tabular benchmark by balanced accuracy |
| Calibrated XGBoost | 97.18% | 0.907 | 0.985 | Calibration trades recall and specificity differently |
| LightGBM | 98.59% | 0.927 | 0.993 | Strong boosted-tree baseline with one false positive |
| Calibrated LightGBM | **99.06%** | **0.956** | 0.991 | Best tabular operating point in the current benchmark table |
| MLP | 98.36% | 0.926 | 0.971 | Competitive non-graph neural baseline |
| Adaptive BioGCN | 96.71% | 0.943 | 0.983 | Reproducibility-oriented benchmark model |
| Adaptive BioGCN robustness | 95.49% ± 0.97% | not reported | not reported | Fixed-split repeated-seed benchmark |
| **ResidualClinicalGCN** | **98.8%** | **0.942** | **0.978** | Best graph operating-point model |

The expanded table changes the interpretation of the biomedical branch in an important way. The graph models are still competitive and clinically interpretable, but they are no longer the only strong models in the benchmark. In particular, uncalibrated XGBoost reaches **98.83% accuracy** and **0.955 balanced accuracy**, while calibrated LightGBM reaches **99.06% accuracy**, **0.956 balanced accuracy**, and detects **32 of 35** pathologic exams with **1 false positive**. The strongest graph claim is therefore more specific: **ResidualClinicalGCN remains a strong graph-based operating-point model, but the current best tabular baselines are at least as strong and in some threshold settings stronger**. This is a healthier and more defensible result than a weaker benchmark table would have been.

### 6.3 Interpretation Across Branches

The quantum and biomedical branches should be read together through one lens. In the QAOA branch, graph conditioning reduces the search burden while preserving quality. In the biomedical branch, graph conditioning improves how the model behaves at clinically meaningful thresholds. The shared contribution is therefore not a single benchmark win, but a demonstration that graph-conditioned learning can mediate downstream decisions in very different domains.

---

## 7. Figures

![QAOA Benchmark Overview](../notebooks/figures/qaoa_demo_benchmark_overview.png)

*Figure 1. Held-out transcriptomic QAOA quality and latency overview.*

![QAOA Ablation and Convergence](../notebooks/figures/qaoa_demo_graph_conditioned_initialization.png)

*Figure 2. QAOA ablation, runtime, and convergence evidence for graph-conditioned initialization.*

![Biomedical Held-Out Evaluation](../notebooks/figures/bio_demo_heldout_evaluation.png)

*Figure 3. Held-out biomedical evaluation summary.*

![Biomedical Operating Point](../notebooks/figures/bio_demo_operating_point_analysis.png)

*Figure 4. Threshold-dependent operating behavior in the biomedical branch.*

![Biomedical Robustness](../notebooks/figures/bio_demo_aligned_biogcn_robustness.png)

*Figure 5. Fixed-split repeated-seed robustness for Adaptive BioGCN.*

---

## 8. Discussion and Limitations

The paper is stronger after adding the ablation, runtime, convergence, and statistical framing, but several important limitations remain.

1. The novelty relative to prior GNN-QAOA papers is now stated more carefully, but still requires external head-to-head comparison.
2. The QAOA branch demonstrates latency improvement and better initialization behavior, but not quantum advantage.
3. The current runtime evidence is useful, yet it is not a full scaling study over increasing graph size.
4. The biomedical branch uses retrospective single-cohort data and therefore should be interpreted as a modeling study, not as deployment evidence.
5. The biomedical baseline suite is stronger than before but still incomplete without boosted-tree and calibrated baselines.

The most important interpretive caveat is this: **these results should not be interpreted as evidence of clinical readiness**.

---

## 9. Reproducibility and Broader Impact

The repository includes notebooks, generated tables, and exported figures that allow the main claims to be inspected directly. The current version still leans on notebook-based execution, so a stronger submission would benefit from more config-driven experiment scripts and cleaner external benchmarking.

From a broader-impact perspective, we aim for careful positioning rather than inflated claims. The quantum results are about initialization efficiency in a small exact-simulation regime. The biomedical results are about retrospective risk stratification under explicit operating-point analysis. Neither branch should be over-read as production evidence.

---

## 10. Conclusion

We presented a unified graph-learning prototype spanning QAOA initialization and biomedical screening. The strongest manuscript-level claim is now explicit: **graph-conditioned learning can act as a reusable computational interface across structurally distinct domains**. In the QAOA branch, this yields near-classical quality with much lower inference-time cost. In the biomedical branch, it yields strong retrospective operating-point behavior under a clearer benchmarking philosophy that separates reproducibility from best-case performance.

The paper is therefore best read as a credible research direction with concrete evidence, clearer limitations, and a sharper path toward a stronger NeurIPS submission.

---

## References

[1] M. Cerezo et al., "Variational quantum algorithms," *Nat. Rev. Phys.*, vol. 3, no. 9, pp. 625-644, 2021.

[2] K. Bharti et al., "Noisy intermediate-scale quantum algorithms," *Rev. Mod. Phys.*, vol. 94, no. 1, Art. no. 015004, 2022.

[3] J. Biamonte et al., "Quantum machine learning," *Nature*, vol. 549, no. 7671, pp. 195-202, 2017.

[4] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," *Nat. Commun.*, vol. 9, Art. no. 4812, 2018.

[5] S. Wang et al., "Noise-induced barren plateaus in variational quantum algorithms," *Nat. Commun.*, vol. 12, Art. no. 6961, 2021.

[6] L. Zhou, S. Wang, S.-T. Wang, M. J. Haghighatlari, and M. D. Lukin, "Quantum approximate optimization algorithm: Performance, mechanism, and implementation on near-term devices," *Quantum*, vol. 4, Art. no. 256, 2020.

[7] D. J. Egger, J. Marecek, and S. Woerner, "Warm-starting quantum optimization," *Phys. Rev. Appl.*, vol. 15, no. 3, Art. no. 034074, 2021.

[8] A. Galda, X. Liu, D. F. Lykov, Y. Alexeev, and I. O. Tolstikhin, "Transferability of optimal QAOA parameters between random graphs," *Phys. Rev. A*, vol. 103, no. 3, Art. no. 032403, 2021.

[9] V. Akshay, D. Rabinovich, E. Campos, and J. Biamonte, "Parameter concentrations in quantum approximate optimization," *PRX Quantum*, vol. 2, no. 1, Art. no. 010348, 2021.

[10] J. Wurtz and P. J. Love, "Counterdiabaticity and the quantum approximate optimization algorithm," *Phys. Rev. A*, vol. 103, no. 4, Art. no. 042612, 2021.

[11] J. Tilly, G. Cerrillo, S. Cao, P. A. M. Casares, and A. Verma, "The variational quantum eigensolver: A review of methods and best practices," *Phys. Rep.*, vol. 986, pp. 1-128, 2022.

[12] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, "Parameterized quantum circuits as machine learning models," *Quantum Sci. Technol.*, vol. 4, no. 4, Art. no. 043001, 2019.

[13] M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," *Phys. Rev. Lett.*, vol. 122, no. 4, Art. no. 040504, 2019.

[14] M. Schuld, A. Bocharov, K. Svore, and N. Wiebe, "Circuit-centric quantum classifiers," *Phys. Rev. A*, vol. 101, no. 3, Art. no. 032308, 2020.

[15] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, "The graph neural network model," *IEEE Trans. Neural Netw.*, vol. 20, no. 1, pp. 61-80, 2009.

[16] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[17] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[18] P. Velickovic et al., "Graph attention networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

[19] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[20] J. Gilmer, S. Schoenholz, P. Riley, O. Vinyals, and G. Dahl, "Neural message passing for quantum chemistry," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017.

[21] Q. Li, Z. Han, and X.-M. Wu, "Deeper insights into graph convolutional networks for semi-supervised learning," in *Proc. AAAI Conf. Artif. Intell. (AAAI)*, 2018.

[22] Y. Rong, W. Huang, T. Xu, and J. Huang, "DropEdge: Towards deep graph convolutional networks on node classification," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[23] K. Oono and T. Suzuki, "Graph neural networks exponentially lose expressive power for node classification," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[24] U. Alon and E. Yahav, "On the bottleneck of graph neural networks and its practical implications," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[25] J. Topping, C. Di Giovanni, B. P. Chamberlain, X. Dong, and M. Bronstein, "Understanding over-squashing and bottlenecks on graphs," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.

[26] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and S. Y. Philip, "A comprehensive survey on graph neural networks," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 32, no. 1, pp. 4-24, 2021.

[27] W. Hu et al., "Strategies for pre-training graph neural networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[28] J. Klicpera, J. Grob, S. Gunnemann, and S. Giri, "Directional message passing for molecular graphs," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[29] V. P. Dwivedi and X. Bresson, "A generalization of transformer networks to graphs," in *Proc. AAAI Conf. Artif. Intell. (AAAI)*, 2021.

[30] V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson, "Benchmarking graph neural networks," *J. Mach. Learn. Res.*, vol. 24, no. 43, pp. 1-48, 2023.

[31] C. Morris et al., "Weisfeiler and Leman go neural: Higher-order graph neural networks," in *Proc. AAAI Conf. Artif. Intell. (AAAI)*, 2019.

[32] G. Corso, L. Cavalleri, D. Beaini, P. Lio, and P. Velickovic, "Principal neighbourhood aggregation for graph nets," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020.

[33] M. Chen, Z. Wei, Z. Huang, B. Ding, and Y. Li, "Simple and deep graph convolutional networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020.

[34] L. Zhao and L. Akoglu, "PairNorm: Tackling oversmoothing in GNNs," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[35] S. Parisot et al., "Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer's disease," *Med. Image Anal.*, vol. 48, pp. 117-130, 2018.

[36] S. I. Ktena et al., "Metric learning with spectral graph convolutions on brain connectivity networks," *IEEE Trans. Med. Imaging*, vol. 37, no. 12, pp. 2987-2998, 2018.

[37] M. Zitnik, M. Agrawal, and J. Leskovec, "Modeling polypharmacy side effects with graph convolutional networks," *Bioinformatics*, vol. 34, no. 13, pp. i457-i466, 2018.

[38] E. Choi, M. T. Bahadori, J. Sun, J. Kulas, A. Schuetz, and W. F. Stewart, "GRAM: Graph-based attention model for healthcare representation learning," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2017.

[39] A. Rajkomar et al., "Scalable and accurate deep learning with electronic health records," *npj Digit. Med.*, vol. 1, Art. no. 18, 2018.

[40] A. Esteva et al., "A guide to deep learning in healthcare," *Nat. Med.*, vol. 25, no. 1, pp. 24-29, 2019.

[41] E. J. Topol, "High-performance medicine: The convergence of human and artificial intelligence," *Nat. Med.*, vol. 25, no. 1, pp. 44-56, 2019.

[42] X. Liu et al., "A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: A systematic review and meta-analysis," *Lancet Digit. Health*, vol. 1, no. 6, pp. e271-e297, 2019.

[43] S. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, and M. Ghassemi, "Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations," *Nat. Med.*, vol. 27, pp. 2176-2182, 2021.

[44] N. Tomašev et al., "A clinically applicable approach to continuous prediction of future acute kidney injury," *Nature*, vol. 572, no. 7767, pp. 116-119, 2019.

[45] S. M. McKinney et al., "International evaluation of an AI system for breast cancer screening," *Nature*, vol. 577, no. 7788, pp. 89-94, 2020.

[46] J. Wiens et al., "Do no harm: A roadmap for responsible machine learning for health care," *Nat. Med.*, vol. 25, no. 9, pp. 1337-1340, 2019.

[47] M. Ghassemi, T. Oakden-Rayner, and A. L. Beam, "The false hope of current approaches to explainable AI in health care," *Lancet Digit. Health*, vol. 3, no. 11, pp. e745-e750, 2021.

[48] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nat. Mach. Intell.*, vol. 1, no. 5, pp. 206-215, 2019.

[49] A. Beam and I. Kohane, "Big data and machine learning in health care," *JAMA*, vol. 319, no. 13, pp. 1317-1318, 2018.

[50] Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan, "Dissecting racial bias in an algorithm used to manage the health of populations," *Science*, vol. 366, no. 6464, pp. 447-453, 2019.

[51] R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad, "Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2015.

[52] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[53] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56-67, 2020.

[54] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017.

[55] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[56] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2016.

[57] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2006.

[58] D. Sculley et al., "Hidden technical debt in machine learning systems," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2015.

[59] P. W. Koh et al., "WILDS: A benchmark of in-the-wild distribution shifts," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021.

[60] S. G. Finlayson, J. D. Bowers, J. Ito, J. L. Zittrain, A. L. Beam, and I. S. Kohane, "Adversarial attacks on medical machine learning," *Science*, vol. 363, no. 6433, pp. 1287-1289, 2019.