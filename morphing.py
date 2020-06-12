# Single click on the image then to store the point press 'e'
# then to complete list, press 'f'

import argparse
import os
import sys

import cv2
import ffmpeg
import numpy as np

from utils import *

def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    triangleList = np.array(triangleList, dtype=np.int32)
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt2 = tuple(t[2:4])
        pt1 = tuple(t[:2])
        pt3 = tuple(t[4:])

        if check_in_rectangle(r, pt1) and check_in_rectangle(r, pt2) and check_in_rectangle(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)


def triangulation(img, given_file):
    delaunay_color = (0, 0, 255)
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    points = []
    with open(given_file) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))
    for p in points:
        subdiv.insert(p)

    draw_delaunay(img, subdiv, delaunay_color)
    cv2.imshow("ABC", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def input_from_mouse(given_img, file_name, source=False):
    file = open(file_name, "w+")

    def draw_circle(event, x, y, flags, param):
        global ix, iy
        if (event == cv2.EVENT_LBUTTONDOWN):
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            ix, iy = x, y

    img = cv2.imread(given_img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('e'):  # ASCII code for e
            if source:
                src_ctrl_pts.append([ix, iy])
            else:
                dest_ctrl_pts.append([ix, iy])
            file.write(str(ix) + " " + str(iy) + "\n")
        if k == ord('f'):  # ASCII code for f
            file.write("0 0\n")
            file.write("0 399\n")
            file.write("299 0\n")
            file.write("299 399")
            if source:
                src_ctrl_pts.append([0, 0])
                src_ctrl_pts.append([0, 399])
                src_ctrl_pts.append([299, 0])
                src_ctrl_pts.append([299, 399])
            else:
                dest_ctrl_pts.append([0, 0])
                dest_ctrl_pts.append([0, 399])
                dest_ctrl_pts.append([299, 0])
                dest_ctrl_pts.append([299, 399])
            break
    cv2.destroyAllWindows()


def delaunay_triangulation(frm_ctrl_pts, img_size):
    rect = (0, 0, img_size[1], img_size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in frm_ctrl_pts:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()
    triangleList = np.int32(triangleList)
    r = (0, 0, img_size[1], img_size[0])
    triangle_List = []
    for t in triangleList:
        pt1 = tuple(t[:2])
        pt2 = tuple(t[2:4])
        pt3 = tuple(t[4:])

        if check_in_rectangle(r, pt1) and check_in_rectangle(r, pt2) and check_in_rectangle(r, pt3):
            point1 = frm_ctrl_pts.index(pt1)
            point2 = frm_ctrl_pts.index(pt2)
            point3 = frm_ctrl_pts.index(pt3)
            triangle_List.append((point1, point2, point3))
    return triangle_List


def generateFrames(src_img, dest_img, num_frms):
    for i in range(0, num_frms+1):
        frm_ctrl_pts = []
        w = np.array([num_frms-i, i])/num_frms
        for j in range(0, len(src_ctrl_pts)):
            val = [int(w[0]*src_ctrl_pts[j][i]+w[1]*dest_ctrl_pts[j][i])
                   for i in range(2)]
            frm_ctrl_pts.append(tuple(val))

        img_size = src_img.shape
        triangle_List = delaunay_triangulation(frm_ctrl_pts, img_size)

        i_img = np.zeros((img_size[0], img_size[1], 3), np.uint8)

        for new_x in range(0, img_size[1]):
            for new_y in range(0, img_size[0]):
                for t in triangle_List:
                    # 3 Control points (x, y)
                    ctr1 = frm_ctrl_pts[t[0]]
                    ctr2 = frm_ctrl_pts[t[1]]
                    ctr3 = frm_ctrl_pts[t[2]]

                    if(inside_rect(ctr1[0], ctr1[1], ctr2[0], ctr2[1], ctr3[0], ctr3[1], new_x, new_y)):

                        val_a = np.array(
                            [[ctr2[i] - ctr1[i], ctr3[i] - ctr1[i]] for i in range(2)])
                        val_b = np.array([new_x - ctr1[0], new_y - ctr1[1]])

                        alpha, beta = np.linalg.solve(val_a, val_b)

                        # 3 Source Control points (x, y)
                        str1 = src_ctrl_pts[t[0]]
                        str2 = src_ctrl_pts[t[1]]
                        str3 = src_ctrl_pts[t[2]]

                        # New source points (x, y)
                        src_pts = [
                            int(alpha*(str2[i] - str1[i])+beta*(str3[i] - str1[i])+str1[i]) for i in range(2)]

                        # 3 Destination Control points (x, y)
                        dtr1 = dest_ctrl_pts[t[0]]
                        dtr2 = dest_ctrl_pts[t[1]]
                        dtr3 = dest_ctrl_pts[t[2]]

                        # New destination points (x, y)
                        dest_pts = [
                            int(alpha*(dtr2[i] - dtr1[i])+beta*(dtr3[i] - dtr1[i])+dtr1[i]) for i in range(2)]

                        i_img[new_y, new_x, :] = np.uint8(
                            w[0]*src_img[src_pts[1], src_pts[0], :]+w[1]*dest_img[dest_pts[1], dest_pts[0], :])

        path = './output/Images'
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = 'img' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(path, file_name), i_img)
        print(f"Frame {i}/{num_frms} generated")

src_ctrl_pts = []
dest_ctrl_pts = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True,
                        help="full path to source image")
    parser.add_argument("--image2", required=True,
                        help="full path to destination image")
    parser.add_argument("--mouseclick", required=False,
                        type=bool, help="Use mouseclick or not")
    arg = parser.parse_args()

    if not os.path.exists("./output"):
        os.mkdir("./output")

    img1 = cv2.imread(arg.image1)
    img2 = cv2.imread(arg.image2)

    # Resizing image
    re_img1 = cv2.resize(img1, (300, 400), interpolation=cv2.INTER_NEAREST)
    re_img2 = cv2.resize(img2, (300, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"./output/resized_{arg.image1}", re_img1)
    cv2.imwrite(f"./output/resized_{arg.image2}", re_img2)

    mouseclick = True
    num_frms = 250

    if(mouseclick):
        input_from_mouse(f"./output/resized_{arg.image1}",
                         f"./output/points_{arg.image1.split('.')[0]}.txt",
                         source=True)
        tri_img1 = triangulation(
            re_img1, f"./output/points_{arg.image1.split('.')[0]}.txt")

        input_from_mouse(f"./output/resized_{arg.image2}",
                         f"./output/points_{arg.image2.split('.')[0]}.txt",
                         source=False)
        tri_img2 = triangulation(
            re_img2, f"./output/points_{arg.image2.split('.')[0]}.txt")

        src_img = cv2.imread(f"./output/resized_{arg.image1}")
        dest_img = cv2.imread(f"./output/resized_{arg.image2}")

        generateFrames(src_img, dest_img, num_frms)

        (ffmpeg.input('./output/Images/img%d.jpg', framerate=30)
         .output('./output/video.mp4')
         .run()
         )

    cv2.imwrite(f"./output/{arg.image1.split('.')[0]}_triangle.jpg", tri_img1)
    cv2.imwrite(f"./output/{arg.image2.split('.')[0]}_triangle.jpg", tri_img2)
