#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 2020

@author: hanjin
"""

import os

import cv2
import numpy as np
from pathlib2 import Path

height_lr = 0


def tonemap(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img ** (1 / 2.2) * 255
    tmo = cv2.createTonemapReinhard(intensity=1.8, light_adapt=0.9, color_adapt=0.0)
    return tmo.process(img.astype(np.float32)) * 255.0


def draw(in_dir, out_dir, coor_src, color, thickness_src, number_of_rectangles):
    files = os.listdir(in_dir)
    for i, file in enumerate(files):
        if file.startswith('._'):
            continue
        if file.endswith('jpg') or file.endswith('png'):
            img = cv2.imread(os.path.join(in_dir, file))[:, :, ::-1]
            # img = tonemap(img)
            h, w = img.shape[:2]

            # low-res
            if h == height_lr:
                coor = (coor_src / 2).astype(int)
                thickness = int(thickness_src / 2)
            else:
                coor = coor_src
                thickness = thickness_src

            crop_images = []
            for j in range(number_of_rectangles):
                start_row = coor[j][0]
                start_col = coor[j][1]
                height = coor[j][2]
                width = coor[j][3]
                end_row = start_row + height - 1
                end_col = start_col + width - 1
                if (start_row - thickness < 1) or (start_col - thickness < 1) or (end_row + thickness > h) or (
                        end_col + thickness > w):
                    print('Out of boundary!')
                    break

                #                cv2.imwrite(os.path.join(out_dir, temp[0]+'.jpg'), rec_img[:,:,::-1])

                img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 0] = color[j][0]
                img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 1] = color[j][1]
                img[start_row - thickness:start_row - 1, start_col - thickness:end_col + thickness, 2] = color[j][2]

                img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 0] = color[j][0]
                img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 1] = color[j][1]
                img[start_row - thickness:end_row + thickness, start_col - thickness:start_col - 1, 2] = color[j][2]

                img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 0] = color[j][0]
                img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 1] = color[j][1]
                img[end_row + 1:end_row + thickness, start_col - thickness:end_col + thickness, 2] = color[j][2]

                img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 0] = color[j][0]
                img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 1] = color[j][1]
                img[start_row - thickness:end_row + thickness, end_col + 1:end_col + thickness, 2] = color[j][2]

                rec_img = img[start_row - thickness // 2:end_row + thickness // 2,
                          start_col - thickness // 2:end_col + thickness // 2, :]

                # if 'fast_light' in file:
                #     file = file.replace('fast_light', 'fastlight')
                # temp = file.split('.')
                # file_name = f"{temp[0]}_{j:d}_{i:d}.{temp[-1]}"
                # # print(file_name)
                # cv2.imwrite(os.path.join(out_dir, file_name), rec_img[:, :, ::-1])
                # # cv2.imwrite(os.path.join(out_dir, temp[0] + '_%d_%d' % (j, i) + '.jpg'), rec_img[:, :, ::-1])
                # # cv2.imwrite(os.path.join(out_dir, file_name), rec_img)

                rec_img = cv2.resize(rec_img, (h // number_of_rectangles, h // number_of_rectangles), cv2.INTER_CUBIC)
                crop_images.append(rec_img)
                if j < number_of_rectangles-1:
                    boundary = np.ones_like(rec_img)[:thickness*2, :]
                    crop_images.append(boundary*255)

            crop_images = np.concatenate(crop_images, axis=0)
            crop_images = cv2.resize(crop_images, (h // number_of_rectangles, h), cv2.INTER_CUBIC)
            boundary = np.ones_like(crop_images)[:, :thickness*2]
            # print(crop_images.shape, img.shape)
            img = np.concatenate([img, boundary*255, crop_images], axis=1)
            cv2.imwrite(os.path.join(out_dir, file), img[:, :, ::-1])

    with open(os.path.join(out_dir, 'z_locations.txt'), 'w') as fd:
        fd.write('{}\n'.format(coor))


if __name__ == "__main__":
    root = Path("/userhome/NAS-ChangPing/sherry/daq/lowev_synthetic_data_221108/teaser/")
    name = '221107_00002_save'
    #    in_dir = '/Users/hanjin/Study/Papers/CVPR2022_HDR/supp_materials/real/Rectangles/%s' % name
    #    out_dir = '/Users/hanjin/Study/Papers/CVPR2022_HDR/supp_materials/real/Rectangles/%s/Boxes/' % name

    in_dir = str(root / name)
    #    in_dir ='Npretrain2'
    out_dir = str(root / f"{name}_bbox_new")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #   1
    # number_of_rectangles = 2
    # coor = np.array([[140, 440, 80, 115],
    #                   [430, 780, 80, 115]])
    # color = np.array([[255, 0, 0],
    #                  [0, 255, 0],
    #                  [0, 0, 255]])

    # 2
    # number_of_rectangles = 2
    # coor = np.array([[80, 300, 80, 115],
    #                 [150, 660, 80, 115]])
    # color = np.array([[255, 0, 0],
    #                [0, 255, 0],
    #                [0, 0, 255]])

    # 3
    number_of_rectangles = 2

    coor = np.array([[280, 50, 50, 50],
                     [100, 420, 50, 50]])

    color = np.array([[235, 91, 142],
                      [26, 177, 218],
                      [0, 0, 255]])
    # 4
    # number_of_rectangles = 3
    # coor = np.array([[90, 670, 100, 160],
    #                  [500, 550, 100, 160],
    #                  [200, 300, 100, 160]])
    # color = np.array([[255, 0, 0],
    #                   [0, 255, 0],
    #                   [0, 0, 255]])
    thickness = 6

    # low = 1
    # if low == 1:
    #     coor = (coor / 2).astype(int)
    #     thickness = int(thickness / 2)

    draw(in_dir, out_dir, coor, color, thickness, number_of_rectangles)

# name = ''
# lr = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# bicu = cv2.resize(lr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# cv2.imwrite(name.replace('lr', 'bicubic'), bicu) 


# sr_dir = 'sr_delete012'
# new_sr_dir = 'sort_sr_delete012'
# if not os.path.exists(new_sr_dir):
#    os.makedirs(new_sr_dir)
# srs = sorted(os.listdir(sr_dir))
# for i, name in enumerate(srs):
#    basename = name.split('_')[0]
#    newname = basename + ('_%08d' % i) + '.jpg'
#    cmd = 'cp %s %s' % (os.path.join(sr_dir, name), os.path.join(new_sr_dir, newname))
#    os.system(cmd)
