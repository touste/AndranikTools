import numpy as np
import cv2 as cv

def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotateImage(image, angle):
    """
    Rotates the given image about it"s centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv.warpAffine(image, affine_mat, new_image_size, flags=cv.INTER_LINEAR)

    return result

def imshow_scaled(name, im, top=1, maxedge=1000):
    cv.imshow(name, im)
    h, w, _ = im.shape
    if h>w:
        w = int(maxedge*w/h)
        h = int(maxedge)
    else:
        h = int(maxedge*h/w)
        w = int(maxedge)
    cv.resizeWindow(name, (w, h))
    cv.setWindowProperty(name, cv.WND_PROP_TOPMOST, top)
    
    
    
# Pick line points on image
def lineinput(event, x, y, flags, params):
    lines, imtodraw, imtodrawbuf, postdrawfun, figname = params
    if event == cv.EVENT_LBUTTONDOWN:
        if len(lines) == 0 or len(lines[-1]) % 2 == 0: # First line point
            pointdraw(imtodraw, x,y)
            lines.append([])
            lines[-1].append((x,y))
        else: # Second line point
            pointdraw(imtodraw, x,y)
            lines[-1].append((x,y))
            linedraw(imtodraw, lines[-1])
            postdrawfun()
        imshow_scaled(figname, imtodraw)
    elif event == cv.EVENT_MOUSEMOVE:
        if len(lines) > 0 and len(lines[-1]) % 2 != 0:
            imtodrawbuf[:] = imtodraw[:]
            linedraw(imtodrawbuf, lines[-1] + [(x, y)])
            imshow_scaled(figname, imtodrawbuf)
            

def pointdraw(im, x, y):
    cv.circle(im, (x, y), radius=2, color=(0, 255, 255), thickness=-1)
    
def linedraw(im, pts):
    p0, p1 = pts
    cv.line(im, p0, p1, (0,0,255), 2)
    
    

# Draw roi on image
def draw_roi(im, roi):
    (x, y, w, h) = [int(v) for v in roi]
    cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw line between two rois
def draw_segment(im, roi1, roi2):
    c1 = ROIcenter(roi1)
    c2 = ROIcenter(roi2)
    cv.line(im, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 255, 0), 2)

# Get ROI center
def ROIcenter(roi):
    (x, y, w, h) = [v for v in roi]
    return (x+w/2, y+h/2)

# Compute length between centers of two rois
def compute_length(roi1, roi2):
    (x1, y1, w1, h1) = [v for v in roi1]
    (x2, y2, w2, h2) = [v for v in roi2]
    return np.sqrt((x1+w1/2 - x2-w2/2)**2 + (y1+h1/2 - y2-h2/2)**2)

# Compute displacement magnitude from initial point
def compute_disp(p1, p0):
    return np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

# Update plot with new data
def update_plot(line1, ax1, line2, ax2, fig, xdata, ydata1, ydata2):
    line1.set_xdata(xdata)
    line1.set_ydata(ydata1)
    ax1.relim()
    ax1.autoscale_view()
    line2.set_xdata(xdata)
    line2.set_ydata(ydata2)
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()