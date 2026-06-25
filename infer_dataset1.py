import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import createULite
from utils import intersect_dicts

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')

def infer():
    cfg_path = r'runs/dataset1_scratch/cfg.yaml'
    ckpt_path = r'runs/dataset1_scratch/best.pth.tar'
    src_dir = r'd:\algorithm\calculation\dataset1\test\images'
    dst_dir = r'd:\algorithm\calculation\dataset1_scratch_prediction'
    
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print("missing model files")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = createULite(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    csd = intersect_dicts(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, model.state_dict())
    model.load_state_dict(csd)
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_FORMATS)]
    print(f"found {len(files)} images")

    img_size = (256, 256)

    with torch.no_grad():
        for fn in tqdm(files):
            src_path = os.path.join(src_dir, fn)
            img = cv2.imread(src_path)
            if img is None: continue
            img = cv2.resize(img, img_size)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze_(0).to(device)
            out = model(t)
            hm = out[0][0].cpu().numpy()
            mask = (hm > 0.5).astype(np.uint8) * 255

            overlay = np.zeros_like(img)
            overlay[:, :, 1] = mask
            blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            result = np.hstack((img, blended, cv2.merge((mask, mask, mask))))
            pure = os.path.splitext(fn)[0]
            cv2.imwrite(os.path.join(dst_dir, f"{pure}_pred.jpg"), result)

    print(f"done: {dst_dir}")

if __name__ == "__main__":
    infer()
