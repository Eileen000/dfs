import cv2

def morphological_operations(image_path):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # 闭运算
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # 保存闭运算后的图片
    cv2.imwrite('/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/close.png', closed_image)

    # 开运算
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

    # 保存开运算后的图片
    cv2.imwrite('/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/open.png', opened_image)

# 请替换为您的图片路径
image_path = "/home/eileen/Diffusion/DuduZhu/Diff_llie/outputs/psld-samples-llie/107/results/23.png"
morphological_operations(image_path)
