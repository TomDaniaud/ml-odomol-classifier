# Synthèse Technique & Architecture - Projet ML-Odomol-Classifier

Ce document fournit une vue d'ensemble complète, technique et autonome de ce dépôt. Il est spécialement structuré pour permettre à un développeur ou une IA externe n'ayant pas accès au code source de comprendre immédiatement l'architecture, les données, les choix algorithmiques, les patchs matériels bas niveau et l'historique des expérimentations.

---

## 1. Contexte et Objectif du Projet

* **Projet** : PRED (Projet de Recherche et Développement) – Polytech Nantes / ONIRIS.
* **Problématique** : Prédiction de la perception olfactive humaine à partir de la structure moléculaire chimique (odorat computationnel).
* **Tâche ML** : **Classification multi-labels** sur **138 descripteurs d'odeurs** (ex. fruité, boisé, floral, vanille, éthéré, soufré...).
* **Données sources** : `data/smiles_odor_dataset.csv` (~5 000 molécules uniques annotées avec 138 étiquettes binaires `0/1`).

---

## 2. Représentation des Données & Pipeline 3D

Le projet a évolué d'une approche par embeddings 2D pré-calculés vers une gestion fine de **géométrie moléculaire 3D multi-conformères** directement traitée par un Transformer de graphes 3D.

### A. Représentation Multi-Conformères 3D (`Conformer3DDataset`)
* **Génération hors-ligne** (`src/scripts/generate_3D_repr.ipynb` $\rightarrow$ `data/conformers_3d_dataset_100_000.pt`) :
  * Pour chaque molécule, génération de ~20 conformères 3D stables via RDKit (algorithme ETKDGv3 puis minimisation d'énergie par champ de force MMFF94).
  * Représente ~100 000 conformères 3D au total.
* **Format d'entrée Uni-Mol** (`src/data/components/unimol_transforms.py`) :
  Pour chaque conformère, 4 tenseurs sont générés :
  1. `src_tokens` : Tokens atomiques projetés selon le dictionnaire Uni-Mol (`molecule_all_h`), entourés des tokens spéciaux `[BOS]` et `[EOS]`.
  2. `src_coord` : Coordonnées cartésiennes $(x, y, z)$ centrées en $(0, 0, 0)$ (standard requis par Uni-Mol), avec $(0,0,0)$ pour les tokens spéciaux.
  3. `src_distance` : Matrice des distances euclidiennes interatomiques par paires via `torch.cdist(coords, coords, p=2.0)`.
  4. `src_edge_type` : Types d'arêtes combinant les indices d'atomes ($token_i \times V + token_j$).
* **Batching dynamique** (`unimol_tensor_collate_fn`) :
  * Padding 1D sur `src_tokens` et `src_coord` selon la molécule la plus longue du batch.
  * Padding 2D pour `src_distance` et `src_edge_type` avec masquage par `pad_idx`.

### B. Stratégies d'Augmentation de Données
* **En entraînement** :
  * **Échantillonnage de conformères** : Tirage aléatoire uniforme d'un conformère parmi les 20 disponibles à chaque époque.
  * **Permutation atomique** : Randomisation de l'ordre des atomes (`torch.randperm`), équivalent 3D de la randomisation SMILES, forçant l'invariance à l'ordre.
  * **Jittering 3D** : Bruit gaussien isotrope continu $\mathcal{N}(0, \sigma^2)$ injecté sur les coordonnées $(x, y, z)$ avec $\sigma \in [0.01, 0.04]$ Å.
  * **Contrôle dynamique** : Drapeau `dyn_augmentation: bool` pour activer/désactiver l'augmentation à la volée (l'expérience 11 a montré qu'un entraînement sans augmentation dynamique permettait une meilleure convergence du F1).
* **En validation** :
  * Déterministe : Dernier conformère fixe, aucune permutation, aucun bruit 3D.

---

## 3. Stratégies de Partitionnement & Prévention des Fuites (Data Leakage)

Le partitionnement est géré par `src/data/datamodule.py` selon deux modes :

1. **Split Stratifié Multi-Labels (`split_type="stratified"`)** :
   * Utilise `MultilabelStratifiedShuffleSplit` pour préserver l'équilibre des 138 classes d'odeurs entre Train (80%) et Val (20%).
   * Le split s'effectue strictement au niveau des molécules (aucun conformère d'une molécule du Train ne se retrouve en Val).
2. **Split par Squelettes de Bemis-Murcko (`split_type="scaffold"`)** :
   * Décompose chaque molécule en son squelette de base (cycles + liens) via RDKit `MurckoScaffold`. Pour les molécules acycliques, un squelette carboné générique est généré.
   * **Random Scaffold Split** : Les molécules partageant le même squelette sont regroupées, mélangées aléatoirement sans tri destructif, puis assignées au Train (80%) ou au Val (20%).
   * **Garantie** : Intersection strictement nulle de familles chimiques entre Train et Val (zéro fuite de sous-structures). Teste la capacité réelle de généralisation hors-distribution.

---

## 4. Gestion du Déséquilibre Extrême des Classes Multi-Labels

Les 138 descripteurs présentent un déséquilibre sévère (certaines odeurs sont présentes sur des centaines de molécules, d'autres sur moins de 10).

* **Pondération Dynamique de la BCE** :
  Calcul vectorisé du ratio négatifs/positifs **sur le Train set uniquement** dans `on_fit_start()` :
  $$\text{pos\_weight}_c = \min\left( \frac{N_{\text{train}} - N_{c}^{+}}{\max(N_{c}^{+}, 1)}, \text{max\_pos\_weight} \right)$$
* **Plafond anti-explosion (`max_pos_weight = 20.0`)** :
  Empêche les gradients des classes ultra-minoritaires de saturer l'optimiseur et de provoquer un surapprentissage destructeur.
* La fonction de perte est `torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)`.

---

## 5. Architecture du Modèle

Le modèle complet est encapsulé dans un `LightningModule` (`src/models/netmodule.py`) couplé à `src/models/components/odomol.py`.

```
Molecule 3D Batch
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Uni-Mol Graph Transformer Backbone                    │
│  - 15 couches d'attention spatiale 3D                  │
│  - 47.3 M paramètres (gelés ou dégel partiel N couches)│
└────────────────────────────────────────────────────────┘
       │  cls_repr [Batch, 512]
       ▼
┌────────────────────────────────────────────────────────┐
│  Décodeur MLP Modulaire (ClassificationHead)           │
│  - Couche 1 : Linear(512 -> 256) + LayerNorm + GELU    │
│               + Dropout(0.4)                           │
│  - Couche 2 : Linear(256 -> 128) + LayerNorm + GELU    │
│               + Dropout(0.4)                           │
│  - Couche 3 : Linear(128 -> 138)                       │
└────────────────────────────────────────────────────────┘
       │  Logits [Batch, 138]
       ▼
BCEWithLogitsLoss (avec pos_weights) & Métriques
```

* **Backbone Uni-Mol** :
  * Pré-entraîné sur des millions de structures moléculaires 3D (`mol_pre_all_h_220816.pt`).
  * Paramètre `unfreeze_last_n_layers: int` : Par défaut à `0` (gel complet du backbone), ajustable pour un fine-tuning progressif des gradients.
* **Décodeur / Tête de Classification (`ClassificationHead`)** :
  * Configurable via `num_layers` (ex: 2 ou 3) et `hidden_channels` (int ou liste `[256, 128]`).
  * Chaque couche intermédiaire comprend : `Linear` $\rightarrow$ `LayerNorm` $\rightarrow$ `GELU` $\rightarrow$ `Dropout`.

---

## 6. Métriques d'Évaluation & Optimisation du Seuil

1. **Métriques standard** :
   * `MultilabelF1Score(average="macro")` : Moyenne non pondérée des F1-scores sur les 138 classes d'odeurs.
   * `MultilabelAUROC(average="macro")` : Évalue la capacité de classement continu indépendamment du seuil de coupure.
2. **Recherche Vectorisée du Seuil Optimal (`on_validation_epoch_end`)** :
   * Le seuil conventionnel de $0.5$ est inefficace pour des classes très déséquilibrées.
   * En fin d'époque de validation, les probabilités et cibles sont concaténées.
   * Une grille vectorisée de 19 seuils $\tau \in [0.05, 0.95]$ est évaluée simultanément sur tenseur PyTorch :
     $$\text{TP}(\tau) = \sum (\hat{y} \ge \tau) \cdot y, \quad \text{FP}(\tau) = \sum (\hat{y} \ge \tau) \cdot (1 - y), \quad \text{FN}(\tau) = \sum (\hat{y} < \tau) \cdot y$$
   * Loggue dans TensorBoard :
     * `valid/best_threshold` : Meilleur seuil de décision global.
     * `valid/f1_opt_global` : F1 macro optimal obtenu à ce seuil.

---

## 7. Optimisation & Hyperparamètres

* **Optimiseur** : `AdamW` (`lr: 0.001`, `weight_decay: 0.05`). L'abandon d'Adam classique au profit d'AdamW avec pénalité $\ell_2$ découplée a été crucial pour limiter l'overfitting.
* **Taille de batch virtuelle** : `batch_size: 8` combiné à `accumulate_grad_batches: 8` simule un batch effectif de 64 sans saturer la VRAM.
* **Scheduler** : `CosineAnnealingLR` décroissant de $10^{-3}$ à $10^{-6}$ sur `max_epochs` (ou entraînement à taux fixe selon les expériences).
* **Précision** : `precision: 32-true` (ou FP32 matmul `high` / TF32).

---

## 8. Abstraction Matérielle & Patchs Bas Niveau Intel Arc (XPU)

Le fichier `src/utils/hardware.py` fournit une couche d'interopérabilité essentielle permettant au code de s'exécuter de façon transparente sur **CUDA**, **Apple Silicon (MPS)**, **CPU**, et particulièrement sur **Intel Arc GPU (XPU)** sous Windows.

### Patchs critiques pour Intel oneAPI Level-Zero sous Windows :
1. **Transferts bloquants (`apply_func_module._BLOCKING_DEVICE_TYPES += ("xpu",)`)** :
   Par défaut, Lightning applique `tensor.to(device, non_blocking=True)`. Sur le runtime Level-Zero Windows, le transfert asynchrone sans synchronisation de flux engendre des corruptions de mémoire partagée (les étiquettes cibles étaient écrasées par les coordonnées 3D). Le forçage en synchrone garantit l'intégrité des tenseurs.
2. **`XPUPrecision` (Contournement `foreach=False` dans le gradient clipping)** :
   `torch.nn.utils.clip_grad_norm_` active par défaut des kernels vectoriels multi-tenseurs (`torch._foreach_norm`). Sur les pilotes Level-Zero Windows, cela déclenche un crash critique `UR_RESULT_ERROR_DEVICE_LOST` (code 20). Le plugin force l'évaluation unitaire stable.
3. **Optimiseur unitaire (`foreach=False` dans Adam/AdamW)** :
   Sous Windows XPU, `torch._foreach_sqrt` génère une erreur `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (code 39). Forcer `opt_kwargs["foreach"] = False` résout le problème.
4. **Synchronisation de file SYCL (`SingleXPUStrategy`)** :
   Appels explicites à `torch.xpu.synchronize()` entre le forward/backward et l'optimizer step pour éviter l'épuisement des descripteurs de commandes (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, code 40).
5. **Reconfiguration UTF-8 sous Windows** :
   Reconfiguration forcée de `sys.stdout` et `sys.stderr` en `utf-8` dans `train.py` pour éliminer les `UnicodeEncodeError` (codec `cp1252`) provoqués par les barres de progression et emojis de la bibliothèque `rich`.

---

## 9. Historique Synthétique des Expériences (`experiments.md`)

| N° | Expérience | Modification clé | Observation / Résultat |
|:---|:---|:---|:---|
| 1 | Baseline initiale | BCE classique, Adam, 2 couches MLP | Overfitting massif, F1 très bas (~17%) |
| 2 | Weighted BCELoss | Ajout de `pos_weight` pour équilibrer les classes | F1 grimpe de 17% à 25% |
| 3 | Scheduler Plateau | `ReduceLROnPlateau` sur `val_loss` | Trop sensible à l'overfitting, le scheduler se bloque |
| 4-5 | AdamW + Weight Decay | AdamW, weight decay 0.01 puis 0.05 | Ralentissement significatif de la hausse de la perte de val |
| 6 | Dropout 0.5 | Augmentation du dropout sur la tête MLP | Réduction notable de l'overfitting |
| 7 | Scheduler Cosine Decay | Décroissance cosinusale du learning rate | Entraînement plus sain, F1 ATH à 27% |
| 8 | Multi-Conformères 3D | Backbone Uni-Mol + conformères RDKit + jitter 3D | Overfitting éliminé, AUROC bondit à 0.85, F1 en apprentissage continu |
| 9 | Seuil optimal dynamique | Recherche de seuil $\tau$ sur grille à chaque époque | Amélioration de la détection sur classes rares |
| 10 | Scaffold Split Bemis-Murcko | Découpage par squelettes chimiques stricts | Élimination totale du data leak chimique (F1 ~21% hors-distribution) |
| 11 | Sans augmentation dynamique | Désactivation de la permutation et du jitter 3D | F1 remonte à 23% (la dynamique introduisait trop de variabilité) |
| 12 | Sans scheduler | LR constant $10^{-3}$ sans décroissance | F1 équivalent (23%) |
| 13 | Décodeur MLP à 3 couches | Architecture pyramidale `[256, 128]` avec LayerNorm | Pas concluant : F1 à 22.5% (n'améliore pas les perfs) |

---

## 10. Organisation des Fichiers & Commandes Usuelles

### Arborescence clé
```
ml-odomol-classifier/
├── configs/                  # Hiérarchie de configuration Hydra
│   ├── data/                 # conformer.yaml, unimol.yaml
│   ├── experiment/           # mlp_3layers.yaml, without_dyn_augm.yaml, scaffold.yaml
│   ├── model/                # odomol.yaml (backbone + head), classifier.yaml
│   ├── trainer/              # Configurations Lightning Trainer
│   └── default.yaml          # Configuration racine Hydra
├── data/
│   ├── conformers_3d_dataset_100_000.pt  # Dataset 3D RDKit
│   ├── smiles_odor_dataset.csv            # Dataset tabulaire brut
│   └── unimol_embeddings.pt               # Embeddings CLS pré-calculés
├── src/
│   ├── data/
│   │   ├── components/       # conformer_dataset.py, unimol_transforms.py
│   │   └── datamodule.py     # OdomolDataModule (scaffold split, pos_weight, collate)
│   ├── models/
│   │   ├── components/       # head.py (MLP modulaire), odomol.py (fine-tuning net)
│   │   └── netmodule.py      # ClassifierLightningModule (BCE, F1, AUROC, seuil opt)
│   ├── utils/
│   │   └── hardware.py       # Stratégie XPU/Intel Arc, CUDA, MPS et patchs
│   └── train.py              # Point d'entrée principal Hydra + Lightning
├── experiments.md            # Journal de bord des métriques et runs
├── pyproject.toml            # Gestionnaire de dépendances UV & index PyTorch XPU
└── todo                      # Feuille de route des axes d'amélioration
```

### Commandes usuelles

* **Lancer un entraînement standard** :
  ```powershell
  uv run src/train.py experiment=mlp_3layers
  ```
* **Lancer un smoke test (1 batch, 1 époque)** :
  ```powershell
  uv run src/train.py experiment=mlp_3layers rn=smoke_test +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1
  ```
* **Lancer TensorBoard pour visualiser les courbes** :
  ```powershell
  uv run tensorboard --logdir logs/tensorboard
  ```
* **Linter et formater le code** :
  ```powershell
  uv run ruff check --fix
  uv run ruff format
  ```
