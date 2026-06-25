"""处理 dataset/valid 单张图像"""
import os
import cv2
import yaml
import torch
import numpy as np
from models import createULite
from utils import intersect_dicts

cfg_path = r'runs/dataset1_scratch/cfg.yaml'
ckpt_path = r'runs/dataset1_scratch/best.pth.tar'
src_img = r'd:\algorithm\calculation\dataset3\valid\Bladder_Trans_Male_mp4-0003_jpg.rf.9f487084cc0f75f94311cd51edbefc33.jpg'
dst_dir = r'd:\algorithm\calculation'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
model = createULite(cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
csd = intersect_dicts(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, model.state_dict())
model.load_state_dict(csd)
model.eval()

img_bgr = cv2.imread(src_img)
orig_h, orig_w = img_bgr.shape[:2]
img_resized = cv2.resize(img_bgr, (256, 256))

rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
t = t.unsqueeze_(0).to(device)
with torch.no_grad():
    out = model(t)
hm = out[0][0].cpu().numpy()
mask = (hm > 0.5).astype(np.uint8) * 255

# 可视化：原图 + 预测绿色叠加
overlay_pred = np.zeros_like(img_resized)
overlay_pred[:, :, 1] = mask
pred_viz = cv2.addWeighted(img_resized, 0.6, overlay_pred, 0.4, 0)
contours_pred, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(pred_viz, contours_pred, -1, (0, 255, 0), 2)

# 拼接：原图 | 预测
panel_w = 256
header_h = 30
header = np.ones((header_h, panel_w * 2, 3), dtype=np.uint8) * 255
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(header, 'Original', (10, 20), font, 0.5, (0, 0, 0), 1)
cv2.putText(header, 'U-Lite Prediction', (panel_w + 10, 20), font, 0.5, (0, 0, 0), 1)

result = np.vstack((header, np.hstack((img_resized, pred_viz))))

pure = os.path.splitext(os.path.basename(src_img))[0]
out_path = os.path.join(dst_dir, f"{pure}_final_result.jpg")
cv2.imwrite(out_path, result)
print(f"done: {out_path}")
