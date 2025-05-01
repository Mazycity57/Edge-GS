import cv2
import numpy as np

def compute_and_save_gradient(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 计算梯度（Sobel 算子）
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X方向梯度
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y方向梯度

    # 计算梯度幅值
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # 归一化到0-255范围并转换为uint8类型
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_image = gradient_normalized.astype(np.uint8)

    # 保存梯度图
    cv2.imwrite(output_path, gradient_image)
    print(f"梯度图已保存到：{output_path}")

# 输入图片路径和输出路径
input_image_path = "/home/wmy/proj/Scaffold-GS-main/image.png"  # 替换为你的输入图片路径
output_image_path = "/home/wmy/proj/Scaffold-GS-main/gradient_output.jpg"  # 替换为你的输出图片路径

# 计算并保存梯度图
compute_and_save_gradient(input_image_path, output_image_path)
