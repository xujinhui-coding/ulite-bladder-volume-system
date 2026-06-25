import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2, torch, yaml, numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from models import createULite
from utils import intersect_dicts

def ulite_snake_pipeline():
    cfg = r'd:\algorithm\u-lite\runs\dataset1_scratch\cfg.yaml'
    ckpt = r'd:\algorithm\u-lite\runs\dataset1_scratch\best.pth.tar'
    src_dir = r'd:\algorithm\calculation\dataset1\test\images'
    dst_dir = r'd:\algorithm\calculation\ulite_snake_results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    with open(cfg) as f:
        cfg_d = yaml.safe_load(f)
    model = createULite(cfg_d).to(device)
    ckpt_d = torch.load(ckpt, map_location=device, weights_only=False)
    csd = intersect_dicts(ckpt_d['state_dict'] if 'state_dict' in ckpt_d else ckpt_d, model.state_dict())
    model.load_state_dict(csd)
    model.eval()
    
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    
    for fn in files:
        img_path = os.path.join(src_dir, fn)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # === Stage 1: U-Lite 粗分割 ===
        input_img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            hm = model(t)[0][0].cpu().numpy()
        
        # 动态阈值：取 heatmap 的 70% 分位作为阈值
        thresh = np.quantile(hm[hm > 0], 0.3) if (hm > 0).any() else 0.5
        mask = (hm > thresh).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))
        
        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 取最大连通域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # === Stage 2: Snake 精修 ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_smooth = gaussian(gray, sigma=2, preserve_range=False)
        
        result_img = img.copy()
        
        if contours:
            # 用 U-Lite 最大轮廓作为 Snake 初始值
            largest = max(contours, key=cv2.contourArea)
            
            # 用 minEnclosingCircle 生成初始 Snake 圆（略微放大）
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            r *= 1.1  # 外扩 10% 给 Snake 收缩空间
            
            s = np.linspace(0, 2*np.pi, 400)
            init = np.array([cy + r*np.sin(s), cx + r*np.cos(s)]).T
            
            # Snake 收敛
            snake = active_contour(img_smooth, init, alpha=0.01, beta=5, gamma=0.001, w_edge=2.0, max_num_iter=500)
            
            # 绘制结果
            pts = np.int32(snake[:, [1, 0]])  # swap col/row back
            cv2.polylines(result_img, [pts], True, (0, 255, 255), 2)  # 黄色 Snake
            
            # 也画上 U-Lite 原始轮廓做对比
            cv2.drawContours(result_img, [largest], -1, (255, 0, 0), 1)  # 蓝色 U-Lite
            
            cv2.putText(result_img, "Yellow=Snake  Blue=U-Lite", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # 如果 U-Lite 没检测到任何区域，用整个扇形区初始化 Snake
            _, bin_all = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            contours_all, _ = cv2.findContours(bin_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_all:
                fan = max(contours_all, key=cv2.contourArea)
                (cx, cy), r = cv2.minEnclosingCircle(fan)
                r *= 0.5
                s = np.linspace(0, 2*np.pi, 400)
                init = np.array([cy + r*np.sin(s), cx + r*np.cos(s)]).T
                snake = active_contour(img_smooth, init, alpha=0.015, beta=10, gamma=0.001, w_edge=2.0)
                pts = np.int32(snake[:, [1, 0]])
                cv2.polylines(result_img, [pts], True, (0, 255, 255), 2)
                cv2.putText(result_img, "Yellow=Snake (no U-Lite)", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(os.path.join(dst_dir, f"{os.path.splitext(fn)[0]}_snake.jpg"), result_img)
    
    print(f"done: {dst_dir}")

if __name__ == "__main__":
    ulite_snake_pipeline()
