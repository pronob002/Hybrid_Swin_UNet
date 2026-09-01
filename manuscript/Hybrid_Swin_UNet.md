# A Hybrid Swin-UNet for Few-Shot Cross-Domain Multi-Organ Segmentation

**Conference:** 2025 28th International Conference on Computer and Information Technology (ICCIT), 19-21 December 2025, Cox’s Bazar, Bangladesh  
**IEEE Citation Info:** 979-8-3315-7867-1/25/ $31.00 ©2025 IEEE  

---

### Authors
- **Pronob Saha**  
  *Department of CSE, Chittagong University of Engineering & Technology, Chattogram-4349, Bangladesh*  
  `pronobsahaa27@gmail.com`
- **Souhardyo Bhattacharjee**  
  *Department of CSE, Northern University Bangladesh, Dhaka, Bangladesh*  
  `souhardyo@nub.ac.bd`
- **Anamika Rani Nath**  
  *Department of CSE, International University of Business Agriculture & Technology, Dhaka-1230, Bangladesh*  
  `anamikanath.cse@iubat.edu`
- **Md. Ruhul Amin**  
  *Department of CSE, Southeast University, Dhaka, Bangladesh*  
  `dr.mdruhulamin@seu.edu.bd`

---

## Abstract
This paper presents Hybrid Swin-UNet, a deep learning framework designed for few-shot, cross-domain multi-organ segmentation that integrates a Swin Transformer encoder within a U-Net backbone. Hybrid Swin-UNet aims to enhance segmentation accuracy and robustness against domain shift, specifically targeting multiple abdominal organs using Computed Tomography (CT) data. The framework is adapted on a few-shot subset of the AMOS dataset and evaluated on the BTCV dataset to simulate a realistic cross-domain challenge. Unlike conventional U-Net architectures that rely on local convolutional features, the Hybrid Swin-UNet architecture leverages the Swin Transformer’s shifted-window self-attention, enabling the model to learn long-range spatial dependencies and global anatomical context. This approach mitigates the need for extensive manual annotation and enhances feature robustness against domain shift. The proposed model achieves a high cross-domain mean Dice Score of 85.1%, with a minimal generalization gap of 3.1%, significantly outperforming the conventional U-Net baseline. These results indicate Hybrid Swin-UNet’s potential to assist clinicians by providing an accurate and generalizable segmentation tool, thus accelerating the deployment of AI in diverse clinical environments. This research underscores the promise of hybrid transformer–CNN architectures within few-shot learning frameworks for advancing robust and practical medical image analysis.

**Index Terms** — Image Segmentation, U-Net, Swin Transformer, Few-Shot Learning, Domain Generalization, Medical Imaging.

---

## I. INTRODUCTION

According to recent estimates, the volume of medical imaging data is growing at an exponential rate, making manual analysis unsustainable for clinical workflows [1]. Abdominal organ segmentation from Computed Tomography (CT) scans is a cornerstone of modern medicine, critical for surgical planning, radiation therapy, and quantitative disease assessment. Traditionally, this process relies on manual delineation by expert radiologists [2]. These methods are essential but prone to inter-observer variability, are extremely time-consuming, and create a significant bottleneck in clinical practice, often resulting in delayed treatment planning.

Recently, deep learning (DL) techniques, especially U-Net and its variants, have transformed medical image analysis, greatly enhancing automatic organ segmentation [3]. CNN-based models can learn hierarchical features directly from raw imaging data, making them more accurate and efficient than traditional segmentation algorithms [4]. Yet, despite their success, many deep learning models still face critical challenges, particularly with the high cost of data annotation and a lack of robustness to data from new clinical sites [5], [6]. This "domain shift" problem, where a model trained at one hospital fails at another due to different scanner protocols, remains a major barrier to real-world deployment.

CNNs like the nnU-Net are popular for segmentation due to their strong performance when ample labeled data is available. Isensee et al. achieved state-of-the-art results across many tasks with a self-configuring pipeline [3]. However, such models are not explicitly designed for data-scarce settings or to handle domain shifts, limiting their practicality. In another study, Gibson et al. developed DenseVNet for multi-organ segmentation, establishing a solid baseline [2]. However, this pre-Transformer model struggles under strong domain variations and requires substantial training data.

The addition of Transformer architectures has shown potential for enhancing segmentation models by capturing long-range dependencies. Chen et al. [7] designed TransUNet, a hybrid model that showed consistent gains over standard U-Nets. Similarly, Hatamizadeh et al. [5] proposed UNETR for 3D segmentation, demonstrating strong performance by leveraging a Vision Transformer encoder. While powerful, these models are often data-hungry and were not originally validated under the dual constraints of few-shot learning and cross-domain generalization. Cao et al. [4] introduced the Swin-UNet, a pure Transformer architecture that outperformed CNN-based models, but its effectiveness in data-scarce 3D settings remains an open question. More recent works have specifically targeted the few-shot problem. Lin et al. [8] proposed CAT-Net using cross-attention, achieving state-of-the-art few-shot performance. However, these methods are often evaluated in episodic training protocols, with limited evidence of their direct, zero-shot generalization capability to entirely new domains.

The fundamental contributions of this research are as follows:
- This work proposes a Hybrid Swin-UNet architecture that combines a Swin Transformer encoder with a U-Net style decoder, adapted specifically for the challenging task of few-shot cross-domain multi-organ segmentation to learn robust, generalizable features.
- The framework is designed to reduce dependency on large, domain-specific labeled datasets by enabling adaptation to new clinical environments from only a small $k$-shot subset of annotated CT volumes drawn from the AMOS source domain.
- The proposed model demonstrates superior cross-domain performance and a substantially smaller generalization gap compared to a standard 3D U-Net trained and adapted under the same few-shot protocol.

The rest of the paper is organized as follows: Section II describes the datasets used in this study. Section III presents the proposed methodology, and Section IV reports the hyperparameter configuration. Section V discusses the experimental results and analysis and Section VI concludes the paper.

---

## II. DATASET DESCRIPTION

The study utilizes two publicly available datasets: the Abdominal Multi-Organ Segmentation Challenge 2022 (AMOS) [9] and the Beyond the Cranial Vault Challenge (BTCV) [10]. AMOS serves as the source domain for few-shot adaptation, while BTCV acts as the unseen target domain to evaluate generalization. Only the 500 CT scans from AMOS are used in this work; the 100 MRI scans are excluded. For each few-shot setting, the $k$-shot support subset is sampled from these AMOS CT volumes, as detailed in the adaptation protocol in Section III-F. Table I demonstrates the dataset description.

#### TABLE I: DATASET DESCRIPTION
| Dataset Name | Role | Content |
| :--- | :--- | :--- |
| **AMOS** | Source Domain (Adaptation) | 500 CT scans, 15 organ labels |
| **BTCV** | Target Domain (Evaluation) | 30 CT scans, 13 organ labels |
| **Total** | | **530 CT Scans** |

---

## III. METHODOLOGY

This section outlines the approach taken for developing and evaluating a Hybrid Swin-UNet for few-shot, cross-domain multi-organ segmentation using abdominal CT scans. Fig. 1 represents the workflow diagram proposed in this study.

The proposed Hybrid Swin-UNet framework for few-shot cross-domain multi-organ segmentation consists of four main stages: preprocessing, feature encoding via a Swin Transformer encoder, reconstruction through a U-Net style decoder, and adaptation with cross-domain evaluation. The pipeline is depicted in Fig. 1 and Fig. 2.

![Fig. 1: Workflow diagram of the proposed few-shot cross-domain segmentation framework.](./img1 (1).png)
*Fig. 1: Workflow diagram of the proposed few-shot cross-domain segmentation framework.*

### A. Model Architecture
The proposed model is a Hybrid Swin-UNet designed for 3D volumetric segmentation. It integrates a hierarchical Swin Transformer encoder with a CNN-based decoder to enhance feature representation. The Hybrid Swin-UNet processes $96 \times 96 \times 96$ input volumes, using shifted-window self-attention to efficiently model local and global dependencies. This patch size offers a practical trade-off between capturing sufficient multi-organ anatomical context within a single 3D crop and keeping the computational and memory cost manageable for volumetric training. Skip connections link the encoder stages to their corresponding decoder stages, allowing the model to combine high-level semantic features with low-level fine-grained details. By integrating the Swin Transformer’s global context with the U-Net’s spatial localization strengths, the model effectively learns robust features that generalize across imaging domains.

![Fig. 2: The proposed Hybrid Swin-UNet model architecture.](./img2.png)
*Fig. 2: The proposed Hybrid Swin-UNet model architecture.*

### B. Segmentation Problem Formulation
The segmentation task is formally defined as learning a voxel-wise classification function:

$$f_{\theta}: X \in \mathbb{R}^{H \times W \times D} \mapsto \hat{Y} \in [0,1]^{H \times W \times D \times C}, \quad (1)$$

where $X$ is a preprocessed CT volume of dimension $H \times W \times D$, $\hat{Y}$ is the predicted voxel-level probability distribution over $C$ organ classes, and $\theta$ denotes the trainable parameters of the model. The workflow can be expressed as a composition of operators:

$$\hat{Y} = D(E(P(X))), \quad Eval = A(\hat{Y}), \quad (2)$$

where $P$ is preprocessing, $E$ the encoder, $D$ the decoder, and $A$ the adaptation and evaluation protocol.

### C. Preprocessing ($P$)
To ensure statistical homogeneity across CT volumes from different scanners, preprocessing standardizes spatial resolution and intensity distribution. Each raw CT scan is resampled with trilinear interpolation to isotropic spacing of $1.5$ mm:

$$X' = R(X; \Delta x = \Delta y = \Delta z = 1.5), \quad (3)$$

and subsequently cropped or padded to a fixed spatial dimension $X'' \in \mathbb{R}^{96 \times 96 \times 96}$. Hounsfield Unit (HU) values are clipped to $[-125,275]$ and normalized to zero mean and unit variance:

$$\begin{aligned}
\hat{X} &= \frac{\min(\max(X'', -125), 275) - \mu}{\sigma}, \\
\mu &= \mathbb{E}[X''], \quad \sigma = \mathbb{V}[X''].
\end{aligned} \quad (4)$$

Stochastic data augmentation is applied using transformation operator $T$:

$$T(X) = \Delta I \big( R_\phi (S_\alpha (F(X))) \big), \quad (5)$$

where $R_\phi$ represents random rotations with $\phi \sim U[-15^\circ, 15^\circ]$, $S_\alpha$ random scaling with $\alpha \sim U[0.9,1.1]$, $F$ random flipping across axes, and $\Delta I$ random intensity perturbation of $\pm 10\%$. The final preprocessed input is:

$$\tilde{X} = T(\hat{X}). \quad (6)$$

### D. Encoder ($E$) – Swin Transformer Encoder
The encoder extracts hierarchical feature representations from $\tilde{X}$. The input $\tilde{X} \in \mathbb{R}^{96 \times 96 \times 96}$ is partitioned into cubic patches of size $M \times M \times M$, each projected into an embedding vector of dimension $d$, producing the token sequence:

$$P = PatchEmbed(\tilde{X}) \in \mathbb{R}^{N \times d}, \quad N = \frac{96^3}{M^3}. \quad (7)$$

Within each window of $M^3$ tokens, multi-head self-attention is applied:

$$Attention(Q, K, V) = Softmax \left( \frac{QK^\top}{d} + B \right) V, \quad (8)$$

where $Q, K, V \in \mathbb{R}^{M^3 \times d}$ are the query, key, and value matrices, $d$ is the attention head dimension, and $B$ is the relative positional encoding.

To extend receptive fields, the architecture alternates between standard W-MSA and shifted window self-attention (SW-MSA), in which windows are shifted by $\lfloor M/2 \rfloor$ voxels. The computational complexity is linear, $\mathcal{O}(N \cdot M^3)$, compared to the quadratic $\mathcal{O}(N^2)$ cost of global self-attention.

The encoder is hierarchical, with each stage halving spatial resolution and doubling embedding dimension:

$$F_l = E_l(F_{l-1}), \quad l = 1, 2, \dots, L, \quad (9)$$

where $F_l$ are progressively abstract feature maps.

### E. Decoder ($D$) – U-Net Style Reconstruction
The decoder reconstructs the segmentation map. At each resolution level $l$, the decoder upsamples features from deeper stage $D_{l+1}$:

$$U(D_{l+1}) \in \mathbb{R}^{2H \times 2W \times 2D}, \quad (10)$$

and concatenates them with encoder features $F_l$:

$$D_l = \phi_{dec} \big( U(D_{l+1}) \oplus F_l \big), \quad (11)$$

where $\oplus$ denotes concatenation and $\phi_{dec}$ is a convolutional block (Conv + Norm + GeLU). The final output is projected into class logits with a $1 \times 1 \times 1$ convolution, followed by Softmax:

$$\hat{Y} = Softmax(W \cdot D_0 + b). \quad (12)$$

### F. Optimization
Model parameters are trained using a hybrid Dice-CrossEntropy loss:

$$L = \lambda \left( 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i + \epsilon} \right) - (1-\lambda) \sum_i g_i \log(p_i), \quad (13)$$

where $p_i$ and $g_i$ denote the predicted probability and ground-truth label for voxel $i$. Optimization uses AdamW with learning rate $\eta = 10^{-4}$ and weight decay $\gamma = 10^{-5}$.

### G. Adaptation and Evaluation ($A$)
Few-shot adaptation is performed on a support set $S = \{ (X_s, Y_s) \}_{s=1}^k$ drawn from the AMOS CT training cases at the volume level, where $k \in \{1, 5\}$ denotes the 1-shot and 5-shot regimes. For each configuration, $k$ distinct scans are randomly sampled from the AMOS CT pool without additional organ-specific balancing beyond the native label distribution of the dataset.

Given a support set $S$, the Hybrid Swin-UNet is fine-tuned by minimizing the hybrid Dice–CrossEntropy loss defined in Section III-F:

$$\theta^* = \arg \min_\theta \sum_{(X_s, Y_s) \in S} L(f_\theta(X_s), Y_s). \quad (14)$$

The fine-tuned model $f_{\theta^*}$ is then evaluated on the complete BTCV dataset, which plays the role of an unseen target domain. No BTCV labels are used during adaptation, making this a strict cross-domain generalization test.

Performance metrics include the Dice Similarity Coefficient (DSC):

$$DSC(P, G) = \frac{2 |P \cap G|}{|P| + |G|}, \quad (15)$$

and the 95th percentile Hausdorff Distance (HD95):

$$\begin{aligned}
HD_{95}(P, G) = \operatorname{Quantile}_{0.95} \Big(&\max \{ \sup_{p \in P} \inf_{g \in G} \|p - g\|, \\
&\sup_{g \in G} \inf_{p \in P} \|g - p\| \} \Big).
\end{aligned} \quad (16)$$

These metrics quantify both volumetric overlap and boundary accuracy. To provide a fair comparison, the 3D U-Net baseline is trained and adapted under the identical few-shot protocol, using the same preprocessing steps, input size of $96 \times 96 \times 96$, and optimization hyperparameters.

---

## IV. Hyperparameter tuning

Table II lists the training parameters that are used to train and adapt the Hybrid Swin-UNet and the 3D U-Net baseline under the few-shot protocol. The fixed input size of $96 \times 96 \times 96$ is used for both models to ensure a fair comparison and to keep the 3D memory footprint tractable during optimization.

#### TABLE II: MODEL TRAINING PARAMETERS AND SPECIFICATIONS
| Parameter | Specifications |
| :--- | :--- |
| **Optimizer** | AdamW |
| **No of epochs (adaptation)** | 50 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | DiceCE (Dice + CrossEntropy) |
| **Input size** | $96 \times 96 \times 96$ |
| **Batch size** | 2 |
| **Split** | $k$-shot from AMOS; all of BTCV |
| **Activation** | GeLU |
| **Activation (output)** | SoftMax |

---

## V. RESULTS AND ANALYSIS

This part presents the outcomes of the suggested Hybrid Swin-UNet model. This study assessed the performance of a traditional 3D U-Net and the proposed Hybrid Swin-UNet model. The 3D U-Net achieved a mean cross-domain Dice Score of 72.8% and a mean HD95 of 28.9 mm. In contrast, the Hybrid Swin-UNet model significantly improved the mean Dice Score to 85.1% and reduced the mean HD95 to 14.2 mm. Table IV shows the overall performance comparison.

The qualitative results in Fig. 3 reveal the Hybrid Swin-UNet’s superior anatomical understanding. With the Swin Transformer, the model correctly separates adjacent organs with ambiguous boundaries. Without global context, the U-Net frequently makes anatomically implausible errors like merging distinct organs. The proposed Hybrid Swin-UNet model exhibits a minimal generalization gap of only 3.1%, compared to 12.3% for the U-Net.

![Fig. 3: Qualitative comparison of segmentation results on a challenging case from the BTCV dataset.](./img3 (1).png)
*Fig. 3: Qualitative comparison of segmentation results on a challenging case from the BTCV dataset.*

#### TABLE III: PER-ORGAN CROSS-DOMAIN PERFORMANCE FOR 3D U-NET AND HYBRID SWIN-UNET
| Class | 3D U-Net: Dice (%) | 3D U-Net: HD95 (mm) | Hybrid Swin-UNet: Dice (%) | Hybrid Swin-UNet: HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| **Spleen** | 89.5 | 16.1 | 94.2 | 8.7 |
| **Right Kidney** | 85.1 | 21.4 | 92.8 | 11.2 |
| **Left Kidney** | 86.3 | 19.8 | 93.1 | 10.5 |
| **Liver** | 92.2 | 15.5 | 96.5 | 7.9 |
| **Pancreas** | 50.3 | 49.3 | 73.9 | 24.1 |
| **Gallbladder** | 43.1 | 51.4 | 65.2 | 22.8 |

#### TABLE IV: OVERALL CROSS-DOMAIN EVALUATION METRICS
| Model | Mean Dice (%) | Mean HD95 (mm) | Gen. Gap (%) |
| :--- | :---: | :---: | :---: |
| **3D U-Net (Baseline)** | 72.8 | 28.9 | 12.3 |
| **Hybrid Swin-UNet (Proposed)** | **85.1** | **14.2** | **3.1** |

### A. State-of-the-art comparison
While fully-supervised models like nnU-Net achieve very high scores, they require large, domain-specific labeled datasets and are not designed for few-shot generalization [3]. Hybrid and transformer-based architectures such as TransUNet and UNETR have shown strong performance on multi-organ benchmarks by leveraging long-range dependencies through self-attention [5], [7]. More recent transformer-based 3D architectures, including Swin-UNETR and UNETR++, further improve accuracy and efficiency on datasets such as BTCV and Synapse by combining hierarchical attention with optimized attention blocks [11], [12]. Specialized abdominal multi-organ networks such as BA-Net report average Dice scores above 89% on AMOS CT under fully supervised training [13], illustrating how much performance can be gained when dense labels are available.

However, these methods are typically designed and evaluated in fully supervised, in-domain settings. They generally assume access to extensive labeled data from the target distribution, and thus do not directly address the scenario studied in this work, where only a small $k$-shot subset from the source domain is available and the model must generalize to an unseen target domain.

In parallel, recent few-shot segmentation frameworks such as CAT-Net primarily rely on episodic support–query training and are often validated on 2D or single-organ segmentation tasks [8]. Extending such episodic protocols to volumetric 3D multi-organ CT segmentation under cross-domain constraints remains non-trivial.

Compared to these lines of work, the proposed Hybrid Swin-UNet is evaluated under a strict few-shot cross-domain protocol: the model is adapted using only $k$ annotated AMOS CT volumes and then tested on the entire BTCV dataset without target-domain labels. Under this challenging zero-shot cross-domain setting, the Hybrid Swin-UNet achieves a highly competitive mean Dice score of 85.1%, with substantially reduced HD95 and a smaller generalization gap than the 3D U-Net baseline, demonstrating practical value for real-world clinical deployment where dense target-domain annotations are not available.

---

## VI. CONCLUSION
This study has introduced the Hybrid Swin-UNet, a deep learning framework that has enhanced few-shot, cross-domain multi-organ segmentation, achieving a high mean Dice Score of 85.1% on an unseen target domain. The model’s success has been largely due to the integration of a Swin Transformer encoder, which has captured long-range anatomical context for improved generalization across different imaging protocols. The Hybrid Swin-UNet has highlighted the potential of hybrid Transformer-CNN architectures in medical imaging, reducing reliance on extensive manual annotation and supporting the development of robust, generalizable segmentation tools. Future research has considered refining the framework by expanding its evaluation to other modalities and exploring advanced adaptation strategies like meta-learning to improve data efficiency and clinical applicability.

---

## REFERENCES
1. S. Bhattacharjee and A. R. Nath, "Sports video classification using vision transformer: A deep learning based approach," in *2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)*. IEEE, 2025, pp. 1–6.
2. E. Gibson, W. Li, C. Sudre, L. Fidon, D. Shakir, G. Wang, Z. Eaton-Rosen, R. Gray, T. Doel, Y. Hu et al., "Automatic multi-organ segmentation on abdominal ct with dense v-networks," *IEEE Transactions on Medical Imaging*, vol. 37, no. 8, pp. 1822–1834, 2018. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC6076994/
3. F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnu-net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, vol. 18, no. 2, pp. 203–211, 2021. [Online]. Available: https://www.nature.com/articles/s41592-020-01008-z
4. H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, "Swin-unet: Unet-like pure transformer for medical image segmentation," *arXiv preprint arXiv:2105.05537*, 2021. [Online]. Available: https://arxiv.org/abs/2105.05537
5. A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. Roth, and D. Xu, "Unetr: Transformers for 3d medical image segmentation," in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2022, pp. 1748–1758. [Online]. Available: https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html
6. Z. Niu, S. Ouyang, S. Xie, Y.-W. Chen, and L. Lin, "A survey on domain generalization for medical image analysis," *arXiv preprint arXiv:2402.05035*, 2024. [Online]. Available: https://arxiv.org/abs/2402.05035
7. J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, "Transunet: Transformers make strong encoders for medical image segmentation," *arXiv preprint arXiv:2102.04306*, 2021. [Online]. Available: https://arxiv.org/abs/2102.04306
8. Y. Lin, Y. Chen, K.-T. Cheng, and H. Chen, "Few shot medical image segmentation with cross attention transformer (cat-net)," in *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2023*, ser. Lecture Notes in Computer Science. Springer, 2023, pp. 233–243. [Online]. Available: https://arxiv.org/abs/2303.13867
9. Y. Ji, W. Bai, C. Ge et al., "Amos: A large-scale abdominal multi-organ benchmark," in *NeurIPS Datasets and Benchmarks Track*, 2022. [Online]. Available: https://arxiv.org/abs/2206.08023
10. "Multi-organ Abdominal CT Reference Standard Segmentations — zenodo.org," https://zenodo.org/records/1169361, [Accessed 27-09-2025].
11. Y. Tang, D. Yang, W. Li, H. R. Roth, B. Landman, D. Xu, V. Nath, and A. Hatamizadeh, "Self-supervised pre-training of swin transformers for 3d medical image analysis," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 20 730–20 740.
12. A. Shaker, M. Maaz, H. Rasheed, S. Khan, M.-H. Yang, and F. S. Khan, "Unetr++: delving into efficient and accurate 3d medical image segmentation," *IEEE Transactions on Medical Imaging*, vol. 43, no. 9, pp. 3377–3390, 2024.
13. S. Hu, Z. Liao, and Y. Xia, "Boundary-aware network for abdominal multi-organ segmentation," *arXiv preprint arXiv:2208.13774*, 2022.
