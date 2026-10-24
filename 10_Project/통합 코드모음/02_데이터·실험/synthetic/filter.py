"""생성 이미지의 그려 넣은 자리에 사람이 실제로 그려졌는지 CLIP으로 1차로 거른다.

YOLO와 다른 모델로 거르기 때문에, YOLO가 놓치는 이미지(학습에 가장 쓸모 있는 이미지)도 버려지지 않는다.
결과는 labels.json의 clip_person에 적는다. 남긴 이미지도 마지막에 사람이 한 번 더 확인한다.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

TEXTS = ["a photo of a person sleeping with their head on a desk",
         "a photo of an empty desk and chair",
         "a photo of a bag, a cap or clothes on a chair"]

out = Path(sys.argv[1])
labels = json.load(open(out / "labels.json"))
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for name, info in labels.items():
    crop = Image.open(out / name).crop(tuple(info["mask_box"]))
    inputs = proc(text=TEXTS, images=crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(-1)[0]
    info["clip_person"] = round(float(probs[0]), 3)
    print(f"{name:28s} {info['clip_person']:.2f}")

json.dump(labels, open(out / "labels.json", "w"), ensure_ascii=False, indent=1)
