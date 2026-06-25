"""
dataset3/valid 推理 + COCO 真值可视化对比
模型: chula (512x640)
输出: d:\algorithm\calculation\ 下 final result 文件
"""
import os
import json
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import createULite
from utils import intersect_dicts

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')

def load_coco_annotations(json_path):
    """加载 COCO 标注，返回 {filename: segmentation_polygon} 字典"""
    with open(json_path) as f:
        data = json.load(f)
    
    # 建立 image_id -> filename 映射
    id_to_file = {}
    for img in data['images']:
        id_to_file[img['id']] = img['file_name']
    
    # 建立 filename -> segmentation 映射
    result = {}
    for ann in data['annotations']:
        fn = id_to_file[ann['image_id']]
        seg = ann['segmentation'][0]  # 多边形点列表 [x1,y1,x2,y2,...]
        # COCO 坐标已经是像素坐标，转为 numpy 数组
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        result[fn] = pts
    return result

def infer():
    cfg_path = r'runs/chula/cfg.yaml'
    ckpt_path = r'runs/chula/best.pth.tar'
    src_dir = r'd:\algorithm\calculation\dataset3\valid'
    coco_path = r'd:\algorithm\calculation\dataset3\valid\_annotations.coco.json'
    dst_dir = r'd:\algorithm\calculation'
    
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print("missing model files")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # 加载模型 (chula: 512x640)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = createULite(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    csd = intersect_dicts(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, model.state_dict())
    model.load_state_dict(csd)
    model.eval()
    
    # 加载 COCO 真值
    gt_dict = load_coco_annotations(coco_path)
    print(f"loaded {len(gt_dict)} ground truth annotations")
    
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_FORMATS)])
    print(f"found {len(files)} images in valid set")
    
    # 模型训练尺寸 512(H) x 640(W)
    model_h, model_w = 512, 640
    
    all_results = []
    
    with torch.no_grad():
        for fn in tqdm(files):
            src_path = os.path.join(src_dir, fn)
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                continue
            
            orig_h, orig_w = img_bgr.shape[:2]  # 256x256
            
            # 模型推理 (resize 到 512x640)
            img_resized = cv2.resize(img_bgr, (model_w, model_h))
            rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze_(0).to(device)
            out = model(t)
            hm = out[0][0].cpu().numpy()  # 512x640
            mask_raw = (hm > 0.5).astype(np.uint8) * 255
            
            # 将 mask 缩回原图尺寸 (256x256)
            mask = cv2.resize(mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # --- 可视化 ---
            # 原图
            img_disp = img_bgr.copy()
            
            # 预测叠加 (绿色半透明)
            overlay_pred = np.zeros_like(img_bgr)
            overlay_pred[:, :, 1] = mask
            pred_viz = cv2.addWeighted(img_bgr, 0.6, overlay_pred, 0.4, 0)
            contours_pred, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_viz, contours_pred, -1, (0, 255, 0), 2)
            
            # 真值叠加 (蓝色半透明)
            gt_viz = img_bgr.copy()
            if fn in gt_dict:
                gt_poly = gt_dict[fn].astype(np.int32)
                gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.fillPoly(gt_mask, [gt_poly], 255)
                overlay_gt = np.zeros_like(img_bgr)
                overlay_gt[:, :, 0] = gt_mask  # 蓝色通道
                gt_viz = cv2.addWeighted(img_bgr, 0.6, overlay_gt, 0.4, 0)
                cv2.polylines(gt_viz, [gt_poly], True, (255, 0, 0), 2)
            
            # 三栏拼接: 原图 | 预测 | 真值
            panel_w = orig_w
            header_h = 30
            header = np.ones((header_h, panel_w * 3, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(header, 'Original', (10, 20), font, 0.5, (0, 0, 0), 1)
            cv2.putText(header, 'U-Lite Prediction', (panel_w + 10, 20), font, 0.5, (0, 0, 0), 1)
            cv2.putText(header, 'Ground Truth (COCO)', (panel_w * 2 + 10, 20), font, 0.5, (0, 0, 0), 1)
            
            h_stack = np.hstack((img_disp, pred_viz, gt_viz))
            result_img = np.vstack((header, h_stack))
            
            # 保存 final result
            pure = os.path.splitext(fn)[0]
            out_path = os.path.join(dst_dir, f"{pure}_final_result.jpg")
            cv2.imwrite(out_path, result_img)
            
            all_results.append({
                'filename': fn,
                'pure_name': pure,
                'has_gt': fn in gt_dict
            })
    
    # ========== 生成 HTML 打包 ==========
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dataset3 Valid - U-Lite 分割结果</title>
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
<h1>U-Lite \u819c\u80f1\u5206\u5272 - dataset3 Valid \u63a8\u7406\u7ed3\u679c</h1>
<div class="legend">
    <span><span class="dot" style="background:#4caf50;"></span>U-Lite \u9884\u6d4b (\u7eff\u8272)</span>
    <span><span class="dot" style="background:#2196f3;"></span>COCO \u771f\u503c\u6807\u6ce8 (\u84dd\u8272)</span>
</div>
<div class="summary">
"""
    total = len(all_results)
    with_gt = sum(1 for r in all_results if r['has_gt'])
    html_content += f"    <p>\u5171 {total} \u5f20\u9a8c\u8bc1\u56fe\u50cf\uff0c{with_gt} \u5f20\u542b COCO \u771f\u503c\u6807\u6ce8</p>\n"
    html_content += "</div>\n<div class=\"grid\">\n"
    
    for r in all_results:
        badge = '<span class="badge badge-green">\u6709 COCO \u771f\u503c</span>' if r['has_gt'] else '<span class="badge badge-gray">\u65e0\u771f\u503c</span>'
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
    
    html_path = os.path.join(dst_dir, 'dataset3_valid_final_results.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n=== \u5b8c\u6210 ===")
    print(f"\u5355\u5f20\u7ed3\u679c (*_final_result.jpg): {dst_dir}")
    print(f"HTML \u5305\u88c5: {html_path}")
    print(f"\u5171\u5904\u7406 {len(all_results)} \u5f20\u56fe\u50cf")

if __name__ == "__main__":
    infer()
