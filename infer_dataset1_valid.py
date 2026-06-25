"""
dataset1/valid 推理 + YOLO 真值可视化对比
模型: dataset1_scratch (256x256)
输出: d:\algorithm\calculation\ 下 final result 文件
"""
import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import createULite
from utils import intersect_dicts

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')

def parse_yolo_poly(label_path, img_w=256, img_h=256):
    """解析 YOLO 多边形标注，返回像素坐标多边形点"""
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    coords = [float(x) for x in parts[1:]]
    pts = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * img_w)
        y = int(coords[i+1] * img_h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

def infer():
    cfg_path = r'runs/dataset1_scratch/cfg.yaml'
    ckpt_path = r'runs/dataset1_scratch/best.pth.tar'
    src_dir = r'd:\algorithm\calculation\dataset1\valid\images'
    label_dir = r'd:\algorithm\calculation\dataset1\valid\labels'
    dst_dir = r'd:\algorithm\calculation'

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
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_FORMATS)])
    print(f"found {len(files)} images in valid set")

    img_size = (256, 256)
    all_results = []

    with torch.no_grad():
        for fn in tqdm(files):
            src_path = os.path.join(src_dir, fn)
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, img_size)
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze_(0).to(device)
            out = model(t)
            hm = out[0][0].cpu().numpy()
            mask = (hm > 0.5).astype(np.uint8) * 255

            # ---- 可视化 ----
            # 预测 (绿色)
            overlay_pred = np.zeros_like(img_bgr)
            overlay_pred[:, :, 1] = mask
            pred_viz = cv2.addWeighted(img_bgr, 0.6, overlay_pred, 0.4, 0)
            contours_pred, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_viz, contours_pred, -1, (0, 255, 0), 2)

            # 真值 (蓝色)
            label_fn = os.path.splitext(fn)[0] + '.txt'
            label_path = os.path.join(label_dir, label_fn)
            gt_poly = parse_yolo_poly(label_path)
            gt_viz = img_bgr.copy()
            if gt_poly is not None and len(gt_poly) > 2:
                gt_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                cv2.fillPoly(gt_mask, [gt_poly], 255)
                overlay_gt = np.zeros_like(img_bgr)
                overlay_gt[:, :, 0] = gt_mask
                gt_viz = cv2.addWeighted(img_bgr, 0.6, overlay_gt, 0.4, 0)
                cv2.polylines(gt_viz, [gt_poly], True, (255, 0, 0), 2)

            # 三栏拼接: 原图 | 预测 | 真值
            panel_w = img_size[0]
            header_h = 30
            header = np.ones((header_h, panel_w * 3, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(header, 'Original', (10, 20), font, 0.5, (0, 0, 0), 1)
            cv2.putText(header, 'U-Lite Prediction', (panel_w + 10, 20), font, 0.5, (0, 0, 0), 1)
            cv2.putText(header, 'Ground Truth', (panel_w * 2 + 10, 20), font, 0.5, (0, 0, 0), 1)

            h_stack = np.hstack((img_bgr, pred_viz, gt_viz))
            result_img = np.vstack((header, h_stack))

            pure = os.path.splitext(fn)[0]
            out_path = os.path.join(dst_dir, f"{pure}_final_result.jpg")
            cv2.imwrite(out_path, result_img)

            all_results.append({
                'filename': fn,
                'pure_name': pure,
                'has_gt': gt_poly is not None and len(gt_poly) > 2
            })

    # ========== HTML 打包 ==========
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dataset1 Valid - U-Lite 分割结果</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; padding: 20px; }
    h1 { text-align: center; margin-bottom: 20px; color: #333; }
    .summary { text-align: center; margin-bottom: 30px; color: #666; font-size: 14px; }
    .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
    .card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        overflow: hidden;
        width: 800px;
    }
    .card img { width: 100%; display: block; }
    .card-body { padding: 12px 16px; }
    .card-body h3 { font-size: 14px; color: #333; margin-bottom: 6px; }
    .card-body .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        color: #fff;
    }
    .badge-green { background: #4caf50; }
    .legend { text-align: center; margin-bottom: 20px; font-size: 13px; color: #555; }
    .legend span { display: inline-block; margin: 0 12px; }
    .legend .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<h1>U-Lite \u819c\u80f1\u5206\u5272 - dataset1 Valid \u63a8\u7406\u7ed3\u679c</h1>
<div class="legend">
    <span><span class="dot" style="background:#4caf50;"></span>U-Lite \u9884\u6d4b (\u7eff\u8272)</span>
    <span><span class="dot" style="background:#2196f3;"></span>\u771f\u503c\u6807\u6ce8 (\u84dd\u8272)</span>
</div>
<div class="summary">
"""
    total = len(all_results)
    with_gt = sum(1 for r in all_results if r['has_gt'])
    html_content += f"    <p>\u5171 {total} \u5f20\u9a8c\u8bc1\u56fe\u50cf\uff0c{with_gt} \u5f20\u542b\u771f\u503c\u6807\u6ce8</p>\n"
    html_content += "</div>\n<div class=\"grid\">\n"

    for r in all_results:
        badge = '<span class="badge badge-green">\u6709\u771f\u503c</span>' if r['has_gt'] else '<span class="badge badge-gray">\u65e0\u771f\u503c</span>'
        html_content += f"""    <div class="card">
        <img src="{r['pure_name']}_final_result.jpg" alt="{r['filename']}" loading="lazy">
        <div class="card-body">
            <h3>{r['filename']}</h3>
            {badge}
        </div>
    </div>
"""
    html_content += """</div>
</body>
</html>"""

    html_path = os.path.join(dst_dir, 'dataset1_valid_final_results.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n=== \u5b8c\u6210 ===")
    print(f"\u5355\u5f20\u7ed3\u679c (*_final_result.jpg): {dst_dir}")
    print(f"HTML \u5305\u88c5: {html_path}")
    print(f"\u5171\u5904\u7406 {len(all_results)} \u5f20\u56fe\u50cf")

if __name__ == "__main__":
    infer()
