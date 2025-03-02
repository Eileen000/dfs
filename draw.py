# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun  30 2020

# @author: hanjin
# """

# import os

# import cv2
# import numpy as np
# from pathlib import Path

# height_lr = 0


# def tonemap(img):
#     img = (img - np.min(img)) / (np.max(img) - np.min(img))
#     return img ** (1 / 2.2) * 255
#     tmo = cv2.createTonemapReinhard(intensity=1.8, light_adapt=0.9, color_adapt=0.0)
#     return tmo.process(img.astype(np.float32)) * 255.0


# def draw(in_dir, out_dir, coor_src, color, thickness_src, number_of_rectangles):
#     files = os.listdir(in_dir)
#     for i, file in enumerate(files):
#         if file.startswith('._'):
#             continue
#         if file.endswith('jpg') or file.endswith('png'):
#             img = cv2.imread(os.path.join(in_dir, file))[:, :, ::-1]
#             # img = tonemap(img)
#             h, w = img.shape[:2]

#             # low-res
#             if h == height_lr:
#                 coor = (coor_src / 2).astype(int)
#                 thickness = int(thickness_src / 2)
#             else:
#                 coor = coor_src
#                 thickness = thickness_src

#             crop_images = []
#             for j in range(number_of_rectangles):
#             	#通过左上角和大小确定patch
#                 start_row = coor[j][0]
#                 start_col = coor[j][1]
#                 height = coor[j][2]
#                 width = coor[j][3]
#                 end_row = start_row + height - 1
#                 end_col = start_col + width - 1
#                 #thickness是方框的粗细
#                 if (start_row - thickness < 1) or (start_col - thickness < 1) or (end_row + thickness > h) or (
#                         end_col + thickness > w):
#                     print('Out of boundary!')
#                     break

#                 #                cv2.imwrite(os.path.join(out_dir, temp[0]+'.jpg'), rec_img[:,:,::-1])

#                 img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 0] = color[j][0]
#                 img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 1] = color[j][1]
#                 img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 2] = color[j][2]

#                 img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 0] = color[j][0]
#                 img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 1] = color[j][1]
#                 img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 2] = color[j][2]

#                 img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 0] = color[j][0]
#                 img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 1] = color[j][1]
#                 img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 2] = color[j][2]

#                 img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 0] = color[j][0]
#                 img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 1] = color[j][1]
#                 img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 2] = color[j][2]

#                 rec_img = img[start_row - thickness // 2:end_row + thickness // 2,
#                           start_col - thickness // 2:end_col + thickness // 2, :]

           
#                 # rec_img = cv2.resize(rec_img, (w // number_of_rectangles, w // number_of_rectangles), cv2.INTER_CUBIC)
#                 rec_img = cv2.resize(rec_img, (width, height), cv2.INTER_CUBIC)
#                 # rec_img = cv2.resize(rec_img, (h // number_of_rectangles, h // number_of_rectangles), cv2.INTER_CUBIC)
#                 # print('>', rec_img.shape)
#                 crop_images.append(rec_img)
#                 if j < number_of_rectangles-1:
#                     boundary = np.ones_like(rec_img)[:, :thickness*2]
#                     # boundary = np.ones_like(rec_img)[:thickness*2, :]
#                     crop_images.append(boundary*255)
                    

#             crop_images = np.concatenate(crop_images, axis=1)
#             # crop_images = np.concatenate(crop_images, axis=0)
#             # crop_images = cv2.resize(crop_images, (w, w// number_of_rectangles), cv2.INTER_CUBIC)
#             crop_images = cv2.resize(crop_images, (w, 100), cv2.INTER_CUBIC)
#             # crop_images = cv2.resize(crop_images, (width, height), cv2.INTER_CUBIC)
#             # crop_images = cv2.resize(crop_images, (h// number_of_rectangles, h), cv2.INTER_CUBIC)
#             boundary = np.ones_like(crop_images)[:thickness*2, :]
#             # boundary = np.ones_like(crop_images)[:, :thickness*2]
#             # print(crop_images.shape, img.shape)
#             img = np.concatenate([img, boundary*255, crop_images], axis=0)
#             # img = np.concatenate([img, boundary*255, crop_images], axis=1)
#             cv2.imwrite(os.path.join(out_dir, file), img[:, :, ::-1])

#     with open(os.path.join(out_dir, 'z_locations.txt'), 'w') as fd:
#         fd.write('{}\n'.format(coor))


# if __name__ == "__main__":
#     root = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/")
#     name = 'lolv2_real_700'

#     in_dir = str(root / name)
#     #    in_dir ='Npretrain2'
#     out_dir = str(root / f"{name}_bbox_new")

#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     # 3
#     number_of_rectangles = 2
    
    
#     #patch 左上角和左下角
#     # coor = np.array([[540, 130, 50, 50],
#     #                  [335, 320, 50, 50]])
#     # coor = np.array([[50, 90, 70, 70],
#     #                  [15, 420, 80, 80]])
#     # coor = np.array([[50, 80, 80, 80],
#     #                  [70, 230, 80, 80]])
#     # coor = np.array([[40, 100, 48, 120],
#     #                  [345, 450, 48, 72]])
#     coor = np.array([[30, 16 , 48, 120],
#                      [30 ,320  ,48 , 72]])
#     # coor = np.array([[85, 260, 50, 50],
#     #                  [305, 440, 50, 50]])
    
#    #边框的颜色
#     color = np.array([[223, 83, 39],
#                       [65, 138, 179],
#                       [0, 0, 255]])

#     thickness = 6

#     draw(in_dir, out_dir, coor, color, thickness, number_of_rectangles)

import cv2
import numpy as np
import os
from pathlib import Path

root = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/")
name = 'lolv2_700'

in_dir = str(root / name)
out_dir = str(root / f"{name}_bbox_new")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

color = np.array([[223, 83, 39],
                  [65, 138, 179],
                  [0, 0, 255]])

pt1, pt2, pt3, pt4 = 0, 0, 0, 0
files = os.listdir(in_dir)
for i, file in enumerate(files):
    if file.endswith('jpg') or file.endswith('png'):
        image = cv2.imread(os.path.join(in_dir, file))
        h, w, _ = image.shape

        # 第一个局部放大图
        pt1 = (70, 60)  # 长方形框左上角坐标
        pt2 = (140, 80)  # 长方形框右下角坐标
        cv2.rectangle(image, pt1, pt2, (179, 138, 65), 2)
        patch1 = image[60:80, 70:140, :]
        patch1 = cv2.resize(patch1, (w//2, 100))  # 调整宽度与图像相同

        # # 第二个局部放大图（如果需要）
        # pt1 = (215, 300)  # 长方形框左上角坐标
        # pt2 = (265, 335)  # 长方形框右下角坐标
        # cv2.rectangle(image, pt1, pt2, (179, 138, 65), 2)
        # patch2 = image[300:335, 215:265, :]
        # patch2 = cv2.resize(patch2, (w, 120))  # 调整宽度与图像相同

        # 第三个局部放大图
        pt3 = (330, 40)  # 长方形框左上角坐标
        pt4 = (400, 60)  # 长方形框右下角坐标
        # cv2.rectangle(image, pt3, pt4, (12, 92, 232), 2)
        cv2.rectangle(image, pt3, pt4, (0, 69, 255), 2)
        patch3 = image[40:60, 330:400, :]
        patch3 = cv2.resize(patch3, (w//2, 100))  # 调整宽度与图像相同

        # 拼接
        patch = np.hstack((patch1, patch3))
        # 将局部图像拼接到原图像的底部
        image = np.vstack((image, patch))

        cv2.imwrite(os.path.join(out_dir, file), image)

with open(os.path.join(out_dir, 'z_locations.txt'), 'w') as fd:
    fd.write('{}\n'.format(pt1))
    fd.write('{}\n'.format(pt2))
    fd.write('{}\n'.format(pt3))
    fd.write('{}\n'.format(pt4))
