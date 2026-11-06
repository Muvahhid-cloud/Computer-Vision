"""Facial emotion recognition demo (happy / sad / neutral) - Assignment 2.

Upload a photo or use the webcam -> ViT prediction -> attention-rollout heatmap -> latency and environment.
A second tab plays the SYNTHETIC talking-head clips used for the evaluation under animation.

Usage:
    python app.py --model path/to/vit_emotion [--clips path/to/clips_preview] [--results path/to/results] [--share]
"""
import argparse, glob, json, os, platform, time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import ViTForImageClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit_emotion', help='folder with model.safetensors, config.json, app_config.json')
parser.add_argument('--clips', default='', help='optional: folder with watermarked synthetic clips (<emotion>/<clip>.mp4)')
parser.add_argument('--results', default='', help='optional: folder with emotion_timeline_*.png figures')
parser.add_argument('--share', action='store_true', help='create a temporary public link')
parser.add_argument('--port', type=int, default=7860)
args = parser.parse_args()

# ---------------- model ----------------
cfg = json.load(open(os.path.join(args.model, 'app_config.json')))
LABELS, SIZE = cfg['labels'], cfg['img_size']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViTForImageClassification.from_pretrained(args.model, attn_implementation='eager').to(device).eval()
tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])
DEVICE_NAME = torch.cuda.get_device_name(0) if device == 'cuda' else (platform.processor() or platform.machine() or 'CPU')
ENV = f'{DEVICE_NAME} | {platform.system()} | Python {platform.python_version()} | torch {torch.__version__}'
times = []


@torch.no_grad()
def predict_with_rollout(pil):
    # forward pass with attentions; attention rollout (Abnar & Zuidema, 2020) -> patch heatmap
    x = tf(pil)[None].to(device)
    out = model(pixel_values=x, output_attentions=True)
    att = torch.stack(out.attentions).float().mean(2)               # layers x 1 x tokens x tokens
    a = att + torch.eye(att.size(-1), device=att.device)
    a = a / a.sum(-1, keepdim=True)
    roll = a[0]
    for layer in range(1, a.size(0)):
        roll = a[layer] @ roll
    m = roll[0, 0, 1:]
    s = int(m.numel() ** 0.5)
    mask = (m / m.max()).reshape(s, s).cpu().numpy()
    probs = torch.softmax(out.logits.float(), -1)[0].cpu().numpy()
    return probs, mask


def overlay(rgb, mask, alpha=0.45):
    heat = cv2.applyColorMap((cv2.resize(mask, rgb.shape[1::-1]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return (rgb * (1 - alpha) + heat[:, :, ::-1] * alpha).astype(np.uint8)


def recognise(img, request: gr.Request = None):
    if img is None:
        return None, None, 'Upload a photo or take one with the webcam, then press Submit.'
    pil = Image.fromarray(img).convert('RGB')
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs, mask = predict_with_rollout(pil)
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    times.append(ms)
    browser = ''
    if request is not None:
        ua = dict(request.headers).get('user-agent', '')
        browser = next((b for b in ['Edg', 'OPR', 'YaBrowser', 'Chrome', 'Firefox', 'Safari'] if b in ua), 'unknown browser')
        browser = {'Edg': 'Edge', 'OPR': 'Opera', 'YaBrowser': 'Yandex Browser'}.get(browser, browser)
    heat = overlay(np.array(pil.resize((SIZE, SIZE))), mask)
    info = (f'This image: {ms:.0f} ms (prediction + explanation) | average over {len(times)} images: {np.mean(times):.0f} ms\n'
            f'Model runs on: {ENV}' + (f'\nYour browser: {browser}' if browser else ''))
    return {e: float(p) for e, p in zip(LABELS, probs)}, heat, info


# ---------------- optional: synthetic clips ----------------
CLIPS = sorted(glob.glob(os.path.join(args.clips, '*', '*.mp4'))) if args.clips else []
CLIP_NAMES = [os.path.relpath(p, args.clips) for p in CLIPS]
TIMELINES = sorted(glob.glob(os.path.join(args.results, 'emotion_timeline_*.png'))) if args.results else []

NOTICE = ('**Privacy & consent:** images are processed only for this prediction and are not stored. '
          'Use only photos of people who agreed to it. This is a course prototype - do not use it to make decisions about people.')

with gr.Blocks(title='Facial emotion recognition (ViT)') as demo:
    gr.Markdown('# Facial emotion recognition - happy / sad / neutral\n'
                'Vision Transformer (ViT-B/16) fine-tuned on our self-collected dataset, with attention-rollout explanations.')
    gr.Markdown(NOTICE)
    with gr.Tab('Recognise emotion'):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(sources=['upload', 'webcam'], type='numpy', label='Face photo (upload or webcam)')
                btn = gr.Button('Submit', variant='primary')
            with gr.Column():
                lab = gr.Label(num_top_classes=3, label='Predicted emotion')
                heat = gr.Image(label='Attention rollout (where the model looks)')
                info = gr.Textbox(label='Latency and environment', lines=3)
        btn.click(recognise, inputs=inp, outputs=[lab, heat, info])
    if CLIPS or TIMELINES:
        with gr.Tab('Synthetic clips (evaluation under animation)'):
            gr.Markdown('**SYNTHETIC MEDIA:** these clips were generated with LivePortrait (VASA-1 concept) from test photos '
                        'of consenting group members, only to test whether the prediction stays stable when the face moves. '
                        '`pose` = head motion only, `lip` = talking mouth.')
            if CLIPS:
                pick = gr.Dropdown(CLIP_NAMES, value=CLIP_NAMES[0], label='Choose a clip')
                vid = gr.Video(value=CLIPS[0], label='Synthetic clip')
                pick.change(lambda n: os.path.join(args.clips, n), inputs=pick, outputs=vid)
            for t in TIMELINES:
                gr.Image(value=t, label=os.path.basename(t) + ' (per-frame emotion probabilities)')

if __name__ == '__main__':
    print('Model:', args.model, '| device:', ENV, '| clips:', len(CLIPS))
    allowed = [os.path.abspath(p) for p in (args.clips, args.results) if p]
    demo.launch(share=args.share, server_port=args.port, allowed_paths=allowed)
