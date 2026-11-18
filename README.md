# Respiratory Sound Detection Using a Fine-Tuned BEATs Encoder

Code, split manifest and saved predictions for the paper:

> Z. K. Sadoon and H. H. Jabbar, "Respiratory Sound Detection Using a Fine-Tuned BEATs Encoder," submitted to the *International Journal of Computing and Digital Systems* (IJCDS).

The study fully fine-tunes the AudioSet-pre-trained BEATs encoder (iter3+, AS2M) with a frame-level head (16 ms) to detect six respiratory sound classes (inhalation, exhalation, crackle, wheeze, stridor, rhonchi) plus an auxiliary Abnormal output on **HF_Lung_V1**. It compares BEATs with four baselines retrained under an identical protocol (EfficientAT, AST, CNN-BiGRU, InceptionTime), and tests wheeze detection externally on **ICBHI 2017** and **SPRSound**.

## Repository structure

```
notebooks/
  01_BEATs_training_evaluation.ipynb                    BEATs 5-fold training, OOF thresholds, test evaluation, MLCM,
                                                        Grad-CAM, frame/event-level evaluation, matched EfficientAT
  02_AST_matched_baseline.ipynb                         AST retrained under the matched protocol
  03_CNN-BiGRU_InceptionTime_matched_and_audits.ipynb   CNN-BiGRU and InceptionTime (matched), Table 5, temporal
                                                        comparison, session/duplicate audit, frame-label rule check
  04_external_validation_uncertainty_calibration.ipynb  ICBHI 2017 / SPRSound external test, cluster bootstrap CIs,
                                                        PR/ROC and calibration, device analysis, CPU inference cost
data/
  split_manifest.csv        recording -> split (train/test), fold (0-4), device, start time, session_id
results/
  beats/                    OOF and test probabilities, thresholds (Accuracy- and F1-optimal), per-fold histories,
                            test metrics, MLCM, threshold scenarios
  matched_baselines/        OOF and 5-fold test probabilities and test metrics of the four baselines; table5_matched.csv
  frame_event/              frame- and event-level metrics (BEATs vs. CNN-BiGRU), reference test events
  external/                 ICBHI 2017 and SPRSound recording lists, probabilities, predictions and metrics
```

## Protocol (identical for all five models)

- 15-s recordings, 4 kHz audio (resampled to 16 kHz for the pre-trained encoders), 938 frames x 7 outputs.
- Session-level 5-fold cross-validation on the official training set: 382 training sessions, 143 test sessions, no session shared between folds or between training and test (`data/split_manifest.csv`).
- Loss: frame-level + recording-level weighted BCE, `pos_weight = clip(neg/pos, 1, 10)`.
- At most 25 epochs, early stopping (patience 5) on S = 0.5 x (six-class Macro-AUC + Abnormal AUC).
- Thresholds searched (0.01-0.99) on out-of-fold probabilities only (Accuracy-optimal and F1-optimal), then fixed before test evaluation; test probabilities are the mean of the five fold models.
- Seed 42 + fold index for all models except EfficientAT, whose seed was not fixed.

## Data

The databases are not redistributed here. Obtain them from their original sources:

- HF_Lung_V1: https://gitlab.com/techsupportHF/HF_Lung_V1
- ICBHI 2017: https://bhichallenge.med.auth.gr/
- SPRSound: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

Pre-trained weights: BEATs iter3+ (AS2M) from https://github.com/microsoft/unilm/tree/master/beats ; EfficientAT `mn10_as` from https://github.com/fschmid56/EfficientAT ; AST from the Hugging Face checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`.

Fine-tuned model weights (about 180 MB per BEATs fold) exceed GitHub limits and are available from the authors on reasonable request. All reported metrics can be recomputed from the saved probabilities in `results/`.

## Environment

Google Colab, PyTorch, NVIDIA A100 (BEATs) and T4 (baselines). The notebooks expect the data and outputs on Google Drive; adjust the paths at the top of each notebook.

## License

MIT (see `LICENSE`).
