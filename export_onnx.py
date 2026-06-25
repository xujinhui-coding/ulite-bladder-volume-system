"""
导出 U-Lite .pt 模型为 .onnx 格式
用法: 修改下方 model_name 选择要导出的模型
输出: runs/<model_name>/<model_name>.onnx
"""
import os
import yaml
import torch
import onnx
from models import createULite
from utils import intersect_dicts

# ===== 选择要导出的模型 =====
# model_name = "dataset1_scratch"  # 256x256, 通用模型
model_name = "crop_chula"      # 64x64, 小图快速模型

# ===== 路径 =====
cfg_path   = f"runs/{model_name}/cfg.yaml"
ckpt_path  = f"runs/{model_name}/best.pth.tar"
onnx_path  = f"runs/{model_name}/{model_name}.onnx"
args_path  = f"runs/{model_name}/args.yaml"

# ===== 加载配置 =====
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
with open(args_path) as f:
    args = yaml.safe_load(f)

# ===== 加载模型 =====
device = torch.device('cpu')
model = createULite(cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
csd = intersect_dicts(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, model.state_dict())
model.load_state_dict(csd, strict=False)
model.eval()

# ===== 获取输入尺寸 =====
h, w = args['img_size']
print(f"Model: {model_name}, input size: {h}x{w}")

# ===== 构造 dummy input =====
dummy = torch.randn(1, 3, h, w)  # batch=1, CHW

# ===== 导出 ONNX =====
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    opset_version=11,
    do_constant_folding=True,
)

# ===== 验证 =====
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"[OK] ONNX exported: {onnx_path}")
print(f"     Input:  {onnx_model.graph.input[0].name} shape={dummy.shape}")
print(f"     Output: {onnx_model.graph.output[0].name}")

print(f"\n========== Done ==========")
print(f"ONNX model: {onnx_path}")
print(f"File size: {os.path.getsize(onnx_path)/1024:.1f} KB")

