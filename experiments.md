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

10 - dynamic augmentation off (FLO COMPLETE ICIIIIIIII)
comments:
ckpt: -
log:

11 - switching to asymetric loss to emphasis on hard example (dyn aug off & new metrics: f1_micro)
comments: best auroc by far, f1 really good (0.25), the training is not fully converged and can be resumed
ckpt: outputs/2026-10-25/22-54-29/checkpoints/last.ckpt
log: logs/tensorboard/losses/asl+f1_micro+300epch-10-25_22-54-29

12 - running 11 for a longer time / reduce the scheduler impact or delete it
comments:
ckpt: -
log:

X - 
comments:
ckpt: -
log:
