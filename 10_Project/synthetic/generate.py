"""빈 방 프레임의 자리마다 Stable Diffusion 인페인팅으로 책상에 엎드린 사람을 그려 넣는다.

08_Gen_Synthetic_data 실습과 같은 stable-diffusion-v1-5/stable-diffusion-inpainting 모델을 쓴다.
자리 주변 512x512만 잘라 그리고, 마스크 안쪽만 원본 프레임에 다시 붙여서 나머지 화면은 그대로 둔다.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

PROMPT = ("photo of a young {who} sleeping, head resting on folded arms on a white desk, {outfit}, {view}, "
          "high angle view from a ceiling corner security camera, indoor fluorescent light, realistic photo")
# 사람 대신 가방이나 모자만 그리는 경우가 많아서 그런 결과를 부정 지시문에 넣는다
NEGATIVE = ("cartoon, painting, deformed, extra limbs, extra heads, blurry face, text, watermark, "
            "empty chair, empty desk, no people, backpack, hat only")
OUTFITS = ["black hoodie", "padded winter jacket", "gray sweatshirt and baseball cap", "white t-shirt"]
# --hard: 머리와 몸이 가려져 YOLO가 놓칠 만한 경우
HARD_OUTFITS = ["a winter coat draped over the head and shoulders",
                "a hoodie with the hood pulled over the head, face hidden",
                "face buried completely in the arms, only the back of the hood visible",
                "a blanket covering the head and back"]
WHO = ["man", "woman"]
# 카메라에서 본 자리별 방향
VIEWS = {"A": "seen from the left side", "B": "facing the camera",
         "C": "seen from behind, back of the head visible", "D": "seen from the right side"}
SIZE = 512


def crop_origin(box, w, h):
    """자리 박스를 가운데에 두는 SIZE x SIZE 잘라내기 위치. 화면 밖으로 나가지 않게 맞춘다."""
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    return min(max(0, cx - SIZE // 2), w - SIZE), min(max(0, cy - SIZE // 2), h - SIZE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", help="사람이 없는 빈 방 프레임")
    p.add_argument("--seats", default="seats.json", help="자리 이름별 박스 [x1, y1, x2, y2]")
    p.add_argument("--out", default="out")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=9.0, help="높을수록 지시문을 더 따른다 (diffusers 기본 7.5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--hard", action="store_true", help="옷차림 대신 머리와 몸이 가려진 경우를 그린다")
    args = p.parse_args()
    outfits = HARD_OUTFITS if args.hard else OUTFITS

    base = cv2.cvtColor(cv2.imread(args.base), cv2.COLOR_BGR2RGB)
    h, w = base.shape[:2]
    seats = json.load(open(args.seats))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16").to(args.device)
    pipe.enable_attention_slicing()

    labels = {}
    for seat, (x1, y1, x2, y2) in seats.items():
        left, top = crop_origin((x1, y1, x2, y2), w, h)
        crop = base[top:top + SIZE, left:left + SIZE]
        mask = np.zeros((SIZE, SIZE), np.uint8)
        mask[y1 - top:y2 - top, x1 - left:x2 - left] = 255
        # 경계를 부드럽게 섞어 붙인 자국이 덜 보이게 한다
        soft = cv2.GaussianBlur(mask, (21, 21), 0)[..., None] / 255.0
        for i, (who, outfit) in enumerate((wh, o) for wh in WHO for o in outfits):
            g = torch.Generator("cpu").manual_seed(args.seed + i)
            prompt = PROMPT.format(who=who, outfit=outfit, view=VIEWS[seat])
            gen = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                       image=Image.fromarray(crop), mask_image=Image.fromarray(mask),
                       height=SIZE, width=SIZE, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=g).images[0]
            frame = base.copy()
            frame[top:top + SIZE, left:left + SIZE] = (np.array(gen) * soft + crop * (1 - soft)).astype(np.uint8)
            name = f"{Path(args.base).stem}_{seat}_{i}.jpg"
            cv2.imwrite(str(out / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            labels[name] = {"seat": seat, "who": who, "outfit": outfit, "mask_box": [x1, y1, x2, y2]}
            print(f"{name}  {who}, {outfit}")

    # mask_box는 사람을 그려 넣으라고 지정한 영역일 뿐이다. 실제 사람 박스는 사람이 확인해서 다시 그린다
    json.dump(labels, open(out / "labels.json", "w"), ensure_ascii=False, indent=1)
