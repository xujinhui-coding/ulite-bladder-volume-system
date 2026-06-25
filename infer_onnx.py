"""
ONNX 推理脚本 (Orange Pi Zero 3 / 任何 arm64 Linux)
无需 PyTorch, 只需 onnxruntime

用法:
  python infer_onnx.py <image_path>

示例:
  python infer_onnx.py test.jpg
  python infer_onnx.py /home/orangepi/Desktop/algorithm/images/1582.jpg
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

# ===== 配置 =====
# 选择模型 (根据需求切换)
ONNX_PATH = "dataset1_scratch_sim.onnx"  # 256x256, 通用
# ONNX_PATH = "crop_chula_sim.onnx"      # 64x64, 更快速

INPUT_SIZE = (256, 256)  # 与模型匹配: (W, H)
THRESHOLD = 0.5

def preprocess(img_path, input_size):
    """读取图像并预处理为模型输入"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, input_size)
    # BGR -> RGB -> CHW ->归一化
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))      # HWC->CHW
    tensor = np.expand_dims(tensor, axis=0)        # -> NCHW
    return tensor, img, orig_w, orig_h

def postprocess(output, orig_img, threshold=0.5):
    """将模型输出转为掩膜并可视化"""
    # output shape: (1, 1, H, W) 取第一个batch的第一个通道
    hm = output[0, 0]  # (H, W)
    mask = (hm > threshold).astype(np.uint8) * 255
    
    # 可视化: 绿色半透明叠加
    overlay = np.zeros_like(orig_img)
    overlay[:, :, 1] = mask
    viz = cv2.addWeighted(orig_img, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
    
    return mask, viz

def main():
    import argparse
    parser = argparse.ArgumentParser(description='U-Lite ONNX 推理')
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('--output', '-o', default=None, help='输出路径 (默认: 输入文件名+_result.jpg)')
    parser.add_argument('--model', '-m', default=ONNX_PATH, help=f'ONNX模型路径 (默认: {ONNX_PATH})')
    parser.add_argument('--size', '-s', nargs=2, type=int, default=INPUT_SIZE, help='输入尺寸 W H (默认: 256 256)')
    parser.add_argument('--threshold', '-t', type=float, default=THRESHOLD, help='分割阈值 (默认: 0.5)')
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        return
    
    # 加载ONNX模型
    print(f"加载模型: {args.model}")
    sess = ort.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    print(f"  输入: {input_name} {sess.get_inputs()[0].shape}")
    print(f"  输出: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
    
    # 预处理
    input_size = tuple(args.size)  # (W, H)
    tensor, orig_img, _, _ = preprocess(args.image, input_size)
    print(f"输入形状: {tensor.shape}")
    
    # 推理
    outputs = sess.run(None, {input_name: tensor})
    print(f"推理完成! 输出形状: {outputs[0].shape}")
    
    # 后处理
    mask, viz = postprocess(outputs, orig_img, args.threshold)
    
    # 结果拼接: 原图 | 预测叠加
    h, w = orig_img.shape[:2]
    header = np.ones((30, w * 2, 3), dtype=np.uint8) * 255
    cv2.putText(header, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(header, 'U-Lite (ONNX)', (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    result = np.vstack((header, np.hstack((orig_img, viz))))
    
    # 保存
    if args.output is None:
        basename = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"{basename}_onnx_result.jpg"
    cv2.imwrite(args.output, result)
    print(f"结果保存: {args.output}")

if __name__ == '__main__':
    main()
