## list of experiments

1 - first training 
comments: super strong overfitting, bad f1 score
ckpt: None
log: logs/tensorboard/experiment/first

2 - weighted BCELoss
comments: f1 score increase from 17% to 25%
ckpt: None
log: logs/tensorboard/experiment/weighted-10-15_11-35-00

3 - scheduler plateau
comment: overfitting is too strong, hitting continually the plateau trigger, so useless feature if overfitting is not fixed
ckpt: -
log: logs/tensorboard/scheduler/plateau0.8-10-15_14-08-32

4 - adamW + weight_decay 0.01
comments: slightly lower loss increase
ckpt: -
log: logs/tensorboard/optimizer/adamW0.01-10-15_14-18-54

5 - adamW + weigth_decay 0.05
comments: again, slightly lower loss increase than 0.01 weigth_decay
ckpt: -
log: logs/tensorboard/optimizer/adamW0.05-10-15_14-21-51

6 - dropout 0.5
comments: learning is slower, but overfitting too. all time high (ATH) reached.
ckpt: -
log: logs/tensorboard/optimizer/dropout0.5-10-15_14-26-18

7 - scheduler cosine decay
comments: overfitting trigger to much the patience of the plateau scheduler, switching to the cosine one for healthier training. ATH 27% f1
ckpt: -
log: logs/tensorboard/scheduler/cosine-decay-10-15_14-33-39

8 - data augmentation (=> 8 batch size + workers=0 + 50epch)
comments: overfitting is completly gone, auroc is way better and growing steadly (ATH 0.85), f1 is way lower but still learning (0.21)
ckpt: -
log: logs/tensorboard/augmentation/test8batchsize-10-18_18-00-18

9 - optimal threshold (virtual batch size of 64 (using acc grad) so aligned with prev training)
comments: 
ckpt: -
log: logs/tensorboard/augmentation/8x8bs-threshold-dyn-10-19_08-50-31

10 - Optimization of the separation of 3D structures between train and val
comments: supposed to decrease data leak, but same f1 (0.207)
ckpt: -
log: logs/tensorboard/scaffold/random-scaffold-split-10-20_20-58-37

11 - no dyn_augmentation
comments: better f1, dynamic augmentation is superflue, we suppose that learning rate is descending too fast. f1(0.23)
ckpt: -
log: logs/tensorboard/augmentation/no-dyn-augm-10-21_14-38-21

12 - no scheduler
comments: same f1
ckpt: -
log: logs/tensorboard/scheduler/no-scheduler-10-21_14-59-44

13 - 3-layer MLP decoder (512 -> 256 -> 128 -> 138)
comments: test de 3 couches de MLP dans le décodeur après Uni-Mol avec régularisation, n'améliore pas le f1 (0.225)
ckpt: -
log: logs/tensorboard/decoder/mlp-3layers-10-21_22-03-49

14 - switching to asymetric loss / focal loss to emphasis on hard example (dyn aug off & new metrics: f1_micro)
comments: best auroc by far, f1 really good (0.25), the training is not fully converged and can be resumed
ckpt: outputs/2026-10-25/22-54-29/checkpoints/last.ckpt
log: logs/tensorboard/losses/asl+f1_micro+300epch-10-25_22-54-29

15 - running focal loss on 1000 epochs
comments: f1 even better (0.28), the training converged at the 850th epoch
ckpt: outputs/2026-11-03/11-02-31/checkpoints/last.ckpt
log: logs/tensorboard/losses/focal-loss-1000ep-11-03_10-56-22

16 - Uni-Mol pipeline fixes + tuned Asymmetric Loss (asl_fixed_unimol, 300 epochs)
comments: Reached 0.31 f1_opt_macro (0.37 per-class F1) in ~40-75 epochs (vs 0.28 at 850 epochs previously), then plateaus due to frozen backbone
ckpt: outputs/2026-11-04/23-46-30/checkpoints/last.ckpt
log: logs/tensorboard/losses/asl-fixed-unimol-11-04_23-46-30

17 - Dynamic 3D augmentation + Fine-tuning last 2 Uni-Mol layers (asl_dyn_aug_finetune, 120 epochs)
comments: Reached 0.35 f1_opt_macro, 0.40 f1_opt_per_class_macro, and 0.48 f1_opt_micro (ATH), with overfitting starting around step 2000 due to 6.3M unfrozen backbone params
ckpt: outputs/2026-11-05/00-42-51/checkpoints/last.ckpt
log: logs/tensorboard/augmentation/asl-dyn-aug-finetune-11-05_00-42-51

X - 
comments:
ckpt: -
log:
