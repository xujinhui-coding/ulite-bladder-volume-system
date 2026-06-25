import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import createULite
from utils import intersect_dicts

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')

def infer_bladder():
    # 训练好的模型权重路径
    cfg_path = r'runs/chula/cfg.yaml'
    ckpt_path = r'runs/chula/best.pth.tar'
    
    # 待推理图像目录: dataset3/test
    src_dir = r'd:\algorithm\calculation\dataset3\test'
    # 输出目录
    dst_dir = r'd:\algorithm\calculation\u_lite_prediction'
    
    if not os.path.exists(cfg_path):
        print(f"错误：找不到配置 {cfg_path}")
        return
    if not os.path.exists(ckpt_path):
        print(f"错误：找不到权重 {ckpt_path}")
        return

    # GPU 推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置和模型
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    model = createULite(cfg_dict).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        csd = intersect_dicts(ckpt['state_dict'], model.state_dict())
    else:
        csd = intersect_dicts(ckpt, model.state_dict())
    model.load_state_dict(csd)
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)

    # 获取所有图像文件
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_FORMATS)]
    print(f"找到 {len(files)} 张图像，开始推理...")

    img_size = (640, 512)  # 训练时使用的尺寸

    with torch.no_grad():
        for fn in tqdm(files):
            src_path = os.path.join(src_dir, fn)
            image = cv2.imread(src_path)
            if image is None:
                continue
            
            image_resized = cv2.resize(image, img_size)
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze_(0).to(device)
            
            mask_output = model(input_tensor)
            heatmap = mask_output[0][0].cpu().numpy()
            
            # 二值化 (阈值 0.5)
            mask = (heatmap > 0.5).astype(np.uint8) * 255
            
            # 可视化: 原图 + 绿色半透明遮罩
            overlay = np.zeros_like(image_resized)
            overlay[:, :, 1] = mask
            blended = cv2.addWeighted(image_resized, 0.7, overlay, 0.3, 0)
            
            # 拼接: 原图 | 叠加掩膜 | 掩膜
            result = np.hstack((
                image_resized,
                blended,
                cv2.merge((mask, mask, mask))
            ))
            
            pure_name = os.path.splitext(fn)[0]
            cv2.imwrite(os.path.join(dst_dir, f"{pure_name}_pred.jpg"), result)

    print(f"推理完成！结果保存在: {dst_dir}")

if __name__ == '__main__':
    infer_bladder()
