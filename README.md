# Facial Emotion Recognition — Assignment 2 (SOTA CV + Demo)

**Computer Vision course · Team: Zainiddin, Nurdaulet, Olzhas**

Recognising **happy / sad / neutral** faces in our own photo dataset. Assignment 2 upgrades the Assignment 1 project with:

- a **Vision Transformer (ViT-B/16)** fine-tuned on our data, compared with the A1 baselines on a **leak-free split**;
- **attention-rollout** heatmaps that show where the model looks;
- **evaluation under animation** (VASA-1 concept, implemented with **LivePortrait**): unseen test photos are turned into talking-head clips and the model is checked frame by frame;
- a **web demo** (Gradio): upload / webcam → emotion → heatmap → latency and environment.

Full write-up: [`docs/A2_Facial_Emotion_Recognition_Report.pdf`](docs/A2_Facial_Emotion_Recognition_Report.pdf)

---

## Results (test set: 172 images, clean split)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| HOG + SVM (A1 baseline) | 0.715 | 0.659 |
| Simple CNN (A1 baseline) | 0.116 | 0.099 |
| MobileNetV2 (A1 model) | 0.547 | 0.427 |
| ViT-B/16, no augmentation | 0.878 | 0.877 |
| **ViT-B/16, with augmentation** | **0.884** | **0.891** |

**Evaluation under animation** (24 test photos → 48 synthetic clips, 624 frames per mode):

| Animation | Macro-F1 stills | Macro-F1 frames | Δmacro-F1 | Consistency |
|---|---|---|---|---|
| `pose` (head motion only) | 0.870 | 0.880 | **+0.010** | 0.963 |
| `lip` (talking mouth) | 0.870 | 0.483 | **−0.388** | 0.772 |

The ViT is robust to head motion but reads an open, talking mouth as a smile.

**Latency** (batch 1, Google Colab): ViT-B/16 — 18.9 ms/image on an NVIDIA T4 GPU, 474.5 ms/image on CPU (85.8 M parameters).

All tables and figures are in [`results/`](results/).

---

## Repository structure

```
app.py                         Gradio demo (upload/webcam -> emotion -> heatmap -> latency)
requirements.txt               packages for running the app locally
notebooks/
  A2_ViT_emotion_recognition.ipynb   data audit + leak-free split, baselines, ViT, ablation, heatmaps, latency, model export
  A2_talking_head_clips.ipynb        LivePortrait clips (VASA-1 concept) + per-frame evaluation, timelines
  A2_demo_app.ipynb                  starts app.py in Colab with a public link (no retraining)
results/                       metrics (csv/txt) and figures from the final run
docs/                          report (PDF)
```

**Not included:** the face photos and generated clips (privacy of the participants) and the trained weights
(`model.safetensors`, ~340 MB, above GitHub's file limit). The weights are produced by the first notebook in
`MyDrive/A2/results/vit_emotion/` and are available from the team on request.

---

## How to run

### A. Google Colab (recommended — this is how all results were produced)

Environment: Google Colab, NVIDIA T4 GPU, PyTorch, Hugging Face Transformers, scikit-learn, OpenCV, Gradio; seed 42.

1. Put the dataset in Google Drive as `MyDrive/Dataset-folder/dataset/{happy,sad,neutral}/` (label = folder name).
2. Open `notebooks/A2_ViT_emotion_recognition.ipynb` in Colab → Runtime → **T4 GPU** → **Run all** (~30 min).
   Writes all metrics/figures and the model (`vit_emotion/`) to `MyDrive/A2/results/`.
3. Open `notebooks/A2_talking_head_clips.ipynb` → T4 GPU → **Run all** (~20 min).
   Installs LivePortrait, writes clips to `MyDrive/A2/clips/` (clean) and `MyDrive/A2/clips_preview/` (watermarked), and the animation results to `MyDrive/A2/results/`.
4. Open `notebooks/A2_demo_app.ipynb` → T4 GPU → **Run all** → open the `https://....gradio.live` link.

### B. Local run guide (app only)

Requires Python 3.10+ and the trained model folder `vit_emotion/` (`model.safetensors`, `config.json`, `app_config.json`).

```bash
pip install -r requirements.txt
python app.py --model path/to/vit_emotion
# optional: also show the synthetic clips and timelines
python app.py --model path/to/vit_emotion --clips path/to/clips_preview --results path/to/results
```

Then open http://127.0.0.1:7860 in the browser. Add `--share` for a temporary public link.
On CPU one prediction (with the heatmap) takes about 0.5 s.

---

## Ethics and synthetic media

- All photographed participants are group members/volunteers who agreed to the use of their photos for this project.
- The talking-head clips are **AI-generated** (LivePortrait). They were made only from test photos of consenting participants,
  only to stress-test the classifier, and every clip and figure is labelled **SYNTHETIC**.
- The app does not store images. It is a course prototype and must not be used to make decisions about people.
- The dataset is small and demographically narrow; performance on other people and conditions is unknown.

## References

1. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
2. Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020.
3. Xu et al., *VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time*, arXiv:2404.10667, 2024.
4. Neurohive, *Microsoft VASA-1: Platform for Animating Static Images*, 2024 — https://neurohive.io/en/tag/animation/
5. Guo et al., *LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control*, arXiv:2407.03168, 2024.
