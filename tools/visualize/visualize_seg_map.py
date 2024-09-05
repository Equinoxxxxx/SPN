import numpy as np
import cv2
import torch


def visualize_segmentation_map(image, one_hot_seg_map, alpha=0.5):
    """
    - image: (H, W, 3) ndarray
    - one_hot_seg_map: one-hot (H, W, num_classes) ndarray
    - alpha: [0, 1]
    """
    # 将one-hot分割图转换为类别索引形式
    seg_map = np.argmax(one_hot_seg_map, axis=-1)
    # 获取类别数量
    num_classes = one_hot_seg_map.shape[-1]
    # 生成颜色映射
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette  # 21 * 3
    colors = (colors % 255).numpy().astype("uint8")
    # colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    # 创建一个RGB图像用于显示分割结果
    seg_image = colors[seg_map]
    # import pdb; pdb.set_trace()
    # 叠加图像和分割图
    overlay = cv2.addWeighted(image, 1 - alpha, seg_image, alpha, 0)
    
    return overlay

