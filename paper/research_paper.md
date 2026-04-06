# Graph-Conditioned Parameterization for Optimization and Clinical Risk Scoring

### Transcriptomic QAOA Initialization and Clinical Similarity-Based Risk Modeling

<div align="center">

**Molena Huynh**

</div>

---

## Abstract

Many graph-learning systems do not terminate in a graph label. Instead, they emit a smaller set of variables that a downstream optimizer, simulator, or decision rule consumes. We study this setting through **graph-conditioned parameterization (GCP)**, in which a graph model maps structured input graphs to task-specific decision variables. The claim is deliberately narrow: the shared object across domains is the interface, not a common supervision signal or shared representation space.

We instantiate GCP in two settings. First, a graph neural network predicts depth-2 QAOA parameters for transcriptomic co-expression graphs derived from prostate expression data. On six held-out graphs, the adapted model achieves a mean approximation ratio of **0.8682 ± 0.0312**, compared with **0.8686 ± 0.0308** for direct classical search, while reducing median end-to-end latency from **675.9 ms** to **0.256 ms**. Second, a graph model emits node-level pathologic-risk scores on a cardiotocography similarity graph. The strongest graph operating point reaches **98.8%** held-out accuracy and **0.942** balanced accuracy, while the strongest tabular baseline, calibrated LightGBM, attains **99.06%** accuracy and **0.956** balanced accuracy.

Our contribution is therefore not a broad claim of superiority, but a reusable computational template,

$$
\text{graph} \rightarrow \text{parameterization} \rightarrow \text{downstream objective},
$$

and empirical evidence that this template remains effective in two materially different settings. Taken together, the results identify GCP as a compact formulation for graph models whose outputs parameterize downstream computation.

---

## 1. Introduction

Many graph-learning problems are only the first stage of a larger computational pipeline. Rather than producing a final label, the model emits a smaller set of quantities that steer downstream computation: optimizer controls, risk scores, thresholds, or structured coefficients. Despite how common this pattern is in practice, it is often described only indirectly, as if the graph model itself were the endpoint of interest. This paper studies that setting directly.

We refer to this role as **graph-conditioned parameterization**. Given an input graph $G$, a graph model produces a task-specific parameterization $\theta_T$ whose quality is determined by a downstream objective. Under this view, the model is best understood as a **structure-aware parameter generator** rather than only as an end-task predictor. That distinction matters whenever the utility of the model depends less on pointwise predictive error than on the behavior of a downstream system that consumes its output.

This perspective is relevant to both branches studied here. In the optimization branch, a graph encoder predicts depth-2 QAOA angles for transcriptomic MaxCut instances. In the biomedical branch, a graph encoder predicts node-level pathologic-risk scores for cardiotocography screening. The tasks differ in semantics, supervision, and evaluation, but they share the same computational role: graph structure is compressed into a downstream-actionable object.

Making that object explicit sharpens evaluation and clarifies comparison. Once the intermediate parameterization is treated as a first-class artifact, runtime, robustness, threshold sensitivity, and calibration become primary rather than auxiliary. A QAOA initializer and a clinical risk surface are not the same scientific object, but both instantiate the pipeline

$$
	ext{graph} \rightarrow \text{parameterization} \rightarrow \text{downstream objective}.
$$

This common structure motivates a single interface formulation across distinct downstream objectives. The contribution is the formulation itself together with two biologically grounded instantiations, explicit runtime accounting in the QAOA branch, and threshold-aware evaluation in the biomedical branch.

### Contributions

1. We define **graph-conditioned parameterization** as a role for graph models that map graph structure to decision-relevant variables consumed by downstream computation.
2. We instantiate that role in two settings: transcriptomic QAOA parameter generation and CTG pathologic-risk scoring.
3. We show that the QAOA instantiation reaches near-parity with direct classical depth-2 search while preserving a strong latency advantage and improving on weaker learned initializers.
4. We show that the biomedical instantiation remains competitive under stronger tabular benchmarking and threshold-aware evaluation, even though the best tabular model is slightly stronger on this split.

---

## 2. Related Work

### 2.1 Learned QAOA Initialization

Learned QAOA initialization is an active direction spanning initialization schemes, parameter concentration, parameter transfer, and graph-informed parameter prediction. Within that literature, the present work contributes a biologically grounded transcriptomic graph family, exact held-out depth-2 evaluation, explicit runtime accounting, and graph-specific ablations under a common graph-conditioned formulation.

### 2.2 Graph Learning in Biomedicine

Graph neural networks are already established in biomedical and clinical prediction tasks, especially where patient similarity, relational structure, or multimodal biological context is informative. The biomedical branch therefore focuses on evaluation protocol rather than architectural novelty: split-first preprocessing, threshold-aware reporting, calibration-sensitive interpretation, and benchmarking against strong tabular baselines.

### 2.3 Interface Positioning

The paper's organizing claim is that both branches instantiate the same graph-to-parameterization-to-objective pipeline. This places the work between application-specific benchmarking and broader methodological framing: the contribution is a reusable interface perspective supported by two materially different case studies.

---

## 3. Graph-Conditioned Parameterization

Let $G = (V, E, X)$ denote a graph with nodes $V$, edges $E$, and node features $X$. For a task $T$, graph-conditioned parameterization is a mapping

$$
\theta_T = f_{\phi, T}(G),
$$

where $\theta_T$ is not itself the final task output but the parameterization consumed by downstream evaluation.

Two aspects of the definition matter. First, the parameterization is task-specific: it may be continuous control variables, node-level probabilities, or a constrained set of structured coefficients. Second, the graph model is assessed through what the parameterization enables downstream, not only through a pointwise regression or classification loss. This distinction is important in both branches of our study. In the optimization setting, small angle errors matter only insofar as they change the final cut quality. In the screening setting, probability estimates matter only insofar as they support stable thresholding and acceptable operating behavior.

Under this view, GCP sits between representation learning and end-task decision making. The graph encoder must still extract useful structure, but the endpoint of interest is an intermediate control object. That interpretation favors evaluation protocols that include ablations, latency accounting, and threshold or calibration analysis rather than only a single end metric.

### 3.1 QAOA Instantiation

For a transcriptomic co-expression graph, the emitted parameterization is a depth-2 angle vector

$$
\theta_{\mathrm{Q}} = (\hat{\gamma}_1, \hat{\gamma}_2, \hat{\beta}_1, \hat{\beta}_2).
$$

This vector is passed to a depth-2 QAOA circuit for MaxCut. Let $C(\theta_{\mathrm{Q}}; G)$ denote the expected cut value under exact simulation and let $C^*(G)$ denote the exact MaxCut value. We evaluate the approximation ratio

$$
r(G) = \frac{C(\theta_{\mathrm{Q}}; G)}{C^*(G)}.
$$

Training uses classical depth-2 angles as regression targets, while evaluation uses the induced QAOA performance on held-out graphs.

This instantiation highlights why parameter quality and objective quality should be separated. Two predicted angle vectors may differ substantially in Euclidean distance yet produce similar expected cut values if they fall in a flat region of the optimization landscape. Conversely, apparently small deviations can be costly when the landscape is sharp. We therefore emphasize held-out approximation ratio and runtime rather than angle error alone.

### 3.2 Biomedical Instantiation

For a CTG patient-similarity graph, the emitted parameterization is a node-level risk vector

$$
\theta_{\mathrm{B}} = (\hat{p}_1, \ldots, \hat{p}_{|V|}), \qquad \hat{p}_i \in [0,1].
$$

These scores are later consumed by thresholding, calibration, and screening analysis. The downstream objective is therefore operational: discrimination, balanced accuracy, recall, false positives, and calibration quality.

The biomedical case clarifies that graph-conditioned parameterization is not restricted to control variables for numerical optimization. Here the emitted object is a clinical decision support surface over nodes in a patient-similarity graph. The scores are useful only after they are converted into an operating policy, so evaluation must consider threshold sensitivity and class-imbalance-aware metrics rather than raw probability output in isolation.

### 3.3 What Is Shared

The shared element is the computational structure

$$
G \rightarrow f_{\phi, T}(G) \rightarrow \theta_T \rightarrow \text{downstream objective}.
$$

What is **not** shared is supervision. We do not train a single model jointly across both tasks, nor do we claim transfer between them.

---

## 4. Experimental Setup

### 4.1 QAOA Protocol

The optimization branch uses transcriptomic co-expression graphs derived from prostate expression data. Graphs are solved under exact depth-2 statevector simulation, and classical references are obtained by multistart Nelder-Mead optimization. We compare four families of initializers: direct classical search, the adapted GNN, a legacy transfer baseline retained for context, and a lightweight prior-style graph-feature regressor inspired by learned QAOA initialization work. We also inspect graph and feature ablations.

The experimental design is intended to isolate whether graph-aware prediction adds value beyond cheaper heuristics. Direct classical search provides the strongest quality reference but is latency-heavy. The lightweight regressor provides a low-capacity learned baseline that does not fully exploit graph structure. The ablations then test whether message passing and node attributes contribute independently to the final objective. This setup makes the central comparison less about beating exhaustive optimization and more about whether graph-conditioned initializers occupy a useful middle ground between quality and speed.

### 4.2 Biomedical Protocol

The biomedical branch uses the UCI cardiotocography cohort under split-first preprocessing and symmetric $k$-NN graph construction. The positive class is pathologic fetal state. Baselines include logistic regression, random forest, MLP, XGBoost, LightGBM, and calibrated variants, alongside graph models. We report both a reproducibility-oriented graph benchmark model and the strongest graph operating-point model.

The split-first protocol is important because it limits leakage from preprocessing choices into held-out evaluation. The benchmark suite is intentionally broader than a graph-only comparison: if the graph model is to be interpreted as a useful decision interface, it should remain credible against well-tuned tabular methods that are natural alternatives for this cohort. Reporting both a reproducibility-oriented benchmark model and a strongest graph operating point helps separate a stable reference configuration from the most favorable graph result.

### 4.3 Scope

The experiments are designed to test whether graph-conditioned parameterization is a useful interface abstraction. They do **not** establish quantum advantage, clinical readiness, cross-domain transfer, or a complete state-of-the-art QAOA benchmark.

That scope restriction is deliberate. The paper is organized to answer a narrower question: can the same graph-to-parameterization pattern be instantiated credibly in two domains with different downstream objectives? This question can be addressed with comparatively small but well-instrumented studies, whereas claims about deployment readiness or broad benchmark leadership would require substantially larger evidence.

### 4.4 Overview of the Two Instantiations

| Branch | Input graph | Parameterization | Downstream objective | Main evaluation |
|---|---|---|---|---|
| QAOA | Transcriptomic co-expression graph | Depth-2 angle vector | Expected MaxCut value | Approximation ratio, runtime |
| Biomedical | CTG similarity graph | Node-level pathologic-risk scores | Thresholded screening behavior | Accuracy, balanced accuracy, ROC AUC |

This side-by-side view is intentionally simple, but it serves an important methodological role. It makes clear that the shared object across branches is not the label space, the loss, or the domain semantics. The common object is the intermediate parameterization that a downstream evaluator consumes. That is the level at which we claim comparability.

---

## 5. Results

### 5.1 QAOA: Near-Classical Quality at Much Lower Inference Cost

The QAOA branch is best interpreted as a compute-quality tradeoff rather than a claim of superiority over direct search. Across six held-out transcriptomic graphs, the adapted GNN nearly matches classical depth-2 search while remaining much faster.

| Method | Held-out mean ratio | Std. dev. | Median runtime (ms) |
|---|---:|---:|---:|
| Classical depth-2 search | 0.8686 | 0.0308 | 675.9079 |
| Adapted GNN initializer | 0.8682 | 0.0312 | 0.2560 |
| Prior-style graph-feature regressor | 0.8208 | 0.0678 | 0.5905 |
| Legacy transfer baseline | 0.5725 | not reported | not reported |

The adapted initializer retains **99.95%** of classical held-out quality while reducing median latency by roughly **2640x**. The learned-comparator result is also informative: a lightweight graph-feature regressor is fast, but the full graph-conditioned model gives materially better initializer quality.

This result is most persuasive when read as an inference-efficiency result. If a workflow requires many repeated evaluations across related graphs, the difference between sub-millisecond prediction and repeated multistart search becomes meaningful even when the approximation-ratio gap is small. The graph-conditioned model therefore functions as an instance-conditioned initializer: expensive reference optimization is incurred during target generation and model fitting, while test-time inference remains cheap.

The ablations indicate that both message passing and node features matter. Removing graph edges lowers the mean ratio to **0.7086**, and removing node features lowers it to **0.6535**. The strongest claim from this branch is therefore specific: **graph-conditioned structure improves the quality of emitted QAOA parameters relative to weaker learned initializers and graph-ablated variants.**

Equally important, the ablations show that the model is not merely memorizing a global angle prior. If it were, graph destruction and feature removal would not degrade quality as sharply. Instead, the observed drops suggest that the emitted parameters depend materially on the relational and attributed structure of each instance. That behavior is consistent with the interpretation of the model as a structure-conditioned generator rather than a static heuristic.

![QAOA Benchmark Overview](../notebooks/figures/qaoa_demo_benchmark_overview.png)

*Figure 1. Held-out transcriptomic QAOA benchmark overview.*

![QAOA Initialization Comparison](../notebooks/figures/qaoa_demo_graph_conditioned_initialization.png)

*Figure 2. Ablation and runtime evidence for graph-conditioned QAOA parameterization.*

### 5.2 Biomedical: Competitive Graph Operating Point Under Stronger Benchmarking

The biomedical branch is intentionally evaluated against stronger tabular baselines than in earlier versions. This changes the interpretation in a useful way: the graph model remains competitive, but it is no longer presented as the undisputed winner.

| Model | Accuracy | Balanced accuracy | ROC AUC | Role in analysis |
|---|---:|---:|---:|---|
| Logistic regression | 94.13% | 0.916 | 0.984 | Linear baseline |
| XGBoost | 98.83% | 0.955 | 0.991 | Strong uncalibrated tabular baseline |
| Calibrated LightGBM | **99.06%** | **0.956** | 0.991 | Strongest tabular operating point |
| Adaptive BioGCN | 96.71% | 0.943 | 0.983 | Reproducibility-oriented graph benchmark |
| ResidualClinicalGCN | 98.8% | 0.942 | 0.978 | Strongest graph operating point |

ResidualClinicalGCN detects **31 of 35** pathologic cases with **1 false positive** at its selected threshold. That is a strong operating point, but calibrated LightGBM remains slightly better in the benchmark table. This narrower interpretation is important: **the graph model is credible and empirically strong, but not the top benchmark winner on this split.**

The practical implication is that graph conditioning remains defensible even in a regime where a strong tabular baseline is hard to surpass. The graph model organizes patient relationships explicitly and produces an operating point with attractive error tradeoffs, but the benchmark table prevents overclaiming. In other words, the biomedical branch supports the interface thesis without requiring the graph model to dominate every comparator.

The calibration-sensitive interpretation matters here. In many screening contexts, the question is not simply which model has the highest aggregate score, but which model yields a controllable tradeoff under a clinically meaningful threshold. The graph model's value is therefore partly tied to how its score surface behaves when translated into a decision rule. That consideration fits naturally with the GCP perspective because the emitted probabilities are intermediate decision variables rather than an endpoint in themselves.

![Biomedical Held-Out Evaluation](../notebooks/figures/bio_demo_heldout_evaluation.png)

*Figure 3. Held-out biomedical evaluation summary for the graph operating-point model.*

### 5.3 Cross-Branch Interpretation

The two branches support the same interface claim from different directions. In the QAOA branch, graph-conditioned parameterization emits a compact control vector for a downstream optimizer. In the biomedical branch, it emits risk scores for downstream thresholding and calibration. The contribution is therefore not a shared benchmark win, but a shared **graph-to-parameterization-to-objective** pattern that remains meaningful across domains with different losses and evaluation criteria.

Seen together, the studies suggest a modest design principle. When a graph model is deployed upstream of a consequential computation stage, it can be valuable to evaluate the model as a parameter generator rather than only as a predictor. That shift encourages reporting on downstream sensitivity, decision latency, and operating behavior, which are often what matter most once the model is embedded in a larger system.

---

## 6. Discussion and Limitations

The manuscript is best interpreted as a conference-style interface paper. It shows that a graph model can be evaluated as a structure-aware parameter generator in two domains where downstream computation matters as much as upstream representation.

Several limits remain:

1. The QAOA study is restricted to small exact-simulation graphs and does not constitute a comprehensive benchmark against recent learned-QAOA methods.
2. The biomedical study is retrospective, cohort-specific, and not a deployment study.
3. The shared framing does not yet imply cross-domain transfer or shared representation learning.

These limitations narrow the claim, but they do not remove its value. The experiments still show that graph-conditioned parameterization is a useful lens for designing and evaluating graph models whose outputs are consumed by downstream solvers or decision policies.

An additional limitation is that the two branches are aligned conceptually rather than through a single shared training framework. We do not present a multi-task model, a common latent space, or an explicit transfer study. That absence is intentional, but it means the paper should be read as an interface formulation supported by two case studies rather than as evidence for universal graph-task coupling.

Future work could extend the framing in at least three directions. First, larger QAOA benchmarks could test whether the quality-latency tradeoff persists under broader graph families and deeper circuits. Second, prospective or multi-site biomedical evaluation could determine whether graph-conditioned risk surfaces remain stable under distribution shift. Third, one could study whether uncertainty estimates over emitted parameterizations improve downstream decision control in both domains.

Another useful extension would be a more explicit study of failure modes. In the optimization branch, this would include identifying graph regimes in which learned initializers collapse toward weak generic initializers. In the biomedical branch, it would include examining whether neighborhood construction amplifies spurious similarity structure or degrades minority-class sensitivity under cohort shift. These questions are downstream-interface questions as much as they are modeling questions, which further motivates the framing adopted here.

---

## 7. Conclusion

We introduced **graph-conditioned parameterization** as an interface perspective for graph-learning systems that emit decision-relevant variables rather than only end-task predictions. We instantiated this perspective in two settings: transcriptomic QAOA angle generation and CTG pathologic-risk scoring.

The empirical results are consistent across the two branches. In the QAOA branch, the learned graph model achieves near-classical held-out quality with a large reduction in inference latency, supporting the use of graph-conditioned outputs as learned initializers for downstream optimization. In the biomedical branch, the graph model yields a strong threshold-aware operating point, while stronger tabular baselines remain slightly superior on this cohort. Taken together, these results establish **graph-conditioned parameterization as a technically useful interface for structured decision systems whose outputs parameterize downstream computation.**

More broadly, the paper argues that graph models should sometimes be evaluated by the role they play inside larger computational pipelines. When a graph encoder emits variables that another module consumes, the interface itself becomes a legitimate object of analysis. That perspective does not replace standard graph learning objectives, but it does provide a sharper language for studying graph-based systems whose outputs are intrinsically decision-facing.

---

## References

[1] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate optimization algorithm," arXiv:1411.4028, 2014.

[2] D. J. Egger, J. Marecek, and S. Woerner, "Warm-starting quantum optimization," *Phys. Rev. Appl.*, vol. 15, no. 3, Art. no. 034074, 2021.

[3] A. Galda, X. Liu, D. F. Lykov, Y. Alexeev, and I. O. Tolstikhin, "Transferability of optimal QAOA parameters between random graphs," *Phys. Rev. A*, vol. 103, no. 3, Art. no. 032403, 2021.

[4] V. Akshay, D. Rabinovich, E. Campos, and J. Biamonte, "Parameter concentrations in quantum approximate optimization," *PRX Quantum*, vol. 2, no. 1, Art. no. 010348, 2021.

[5] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[6] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[7] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[8] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and S. Y. Philip, "A comprehensive survey on graph neural networks," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 32, no. 1, pp. 4-24, 2021.

[9] S. Parisot et al., "Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer's disease," *Med. Image Anal.*, vol. 48, pp. 117-130, 2018.

[10] A. Rajkomar et al., "Scalable and accurate deep learning with electronic health records," *npj Digit. Med.*, vol. 1, Art. no. 18, 2018.

[11] A. Esteva et al., "A guide to deep learning in healthcare," *Nat. Med.*, vol. 25, no. 1, pp. 24-29, 2019.

[12] J. Wiens et al., "Do no harm: A roadmap for responsible machine learning for health care," *Nat. Med.*, vol. 25, no. 9, pp. 1337-1340, 2019.

[13] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017.

[14] R. Caruana et al., "Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2015.

[15] M. Cerezo et al., "Variational quantum algorithms," *Nat. Rev. Phys.*, vol. 3, no. 9, pp. 625-644, 2021.
