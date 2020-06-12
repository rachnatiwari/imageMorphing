import cv2


def check_in_rectangle(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def area(x1, y1, x2, y2, x3, y3):
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def inside_rect(x1, y1, x2, y2, x3, y3, x, y):
    a1 = area(x1, y1, x2, y2, x3, y3)
    a2 = area(x, y, x2, y2, x3, y3)
    a3 = area(x1, y1, x, y, x3, y3)
    a4 = area(x1, y1, x2, y2, x, y)
    if(a1 == a2 + a3 + a4):
        return True
    else:
        return False


def draw_point(img, p, color):
    # https://stackoverflow.com/a/60546030
    cv2.circle(img=img, center=p, radius=1, color=color, thickness=-1, shift=0)
