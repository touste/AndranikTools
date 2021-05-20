import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import csv
from tkinter import Tk, filedialog
from rich.console import Console    
from rich.rule import Rule
from rich.progress import track


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

console = Console()

# Init tkinter for file selection
root = Tk() # pointing root to Tk() to use it as Tk() in program.
root.withdraw() # Hides small tkinter window.
root.attributes("-topmost", True) # Opened windows will be active. above all windows despite of selection.

# Draw roi on image
def draw_roi(im, roi):
    (x, y, w, h) = [int(v) for v in roi]
    cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw line between two rois
def draw_segment(im, roi1, roi2):
    (x1, y1, w1, h1) = [int(v) for v in roi1]
    (x2, y2, w2, h2) = [int(v) for v in roi2]
    cv.line(im, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), (0, 255, 0), 2)

# Compute length between centers of two rois
def compute_length(roi1, roi2):
    (x1, y1, w1, h1) = [int(v) for v in roi1]
    (x2, y2, w2, h2) = [int(v) for v in roi2]
    return np.sqrt((x1+w1/2 - x2-w2/2)**2 + (y1+h1/2 - y2-h2/2)**2)

# Update plot with new data
def update_plot(line, ax, fig, xdata, ydata):
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


# Select image directory 
console.print(Rule())
console.print("Select image directory", style="green bold")
console.print("This is the folder with all the BMP images.", style="italic")
imdir = filedialog.askdirectory(title = "Select image directory")


# Get bmp files in imdir
filenames = [img for img in glob.glob(imdir + "/*.bmp")]
filenames.sort()

# Read first image
imref = cv.imread(filenames[0])

# Create window
cv.namedWindow("Images", cv.WINDOW_NORMAL)
imshow_scaled("Images", imref)

while True:
    # Make a copy of imref and work with this
    imref_cpy = imref.copy()
    # Select two ROIs and display them
    console.print(Rule())
    console.print("Select first marker", style="green bold")
    console.print("Start by clicking at the center of the marker, then grow a square around it. The square should be slightly bigger than the marker.", style="italic")
    ROI1 = cv.selectROI("Images", imref_cpy, fromCenter=True, showCrosshair=True)
    draw_roi(imref_cpy, ROI1)
    console.print(Rule())
    console.print("Select second marker", style="green bold")
    console.print("Start by clicking at the center of the marker, then grow a square around it. The square should be slightly bigger than the marker.", style="italic")
    ROI2 = cv.selectROI("Images", imref_cpy, fromCenter=True, showCrosshair=True)
    draw_roi(imref_cpy, ROI2)
    # Display segment between rois
    draw_segment(imref_cpy, ROI1, ROI2)
    imshow_scaled("Images", imref_cpy)
    console.print(Rule())
    console.print("Press ENTER to validate, else press any other key to start again.", style="yellow bold blink")
    key = cv.waitKey(0)
    if key == 13: # if ENTER is pressed, exit loop
        break



# Initial length
l0 = compute_length(ROI1, ROI2)

# Init trackers
tracker1 = cv.TrackerCSRT_create()
tracker2 = cv.TrackerCSRT_create()
tracker1.init(imref, ROI1)
tracker2.init(imref, ROI2)

# Init strain plot
plt.ion()
fig, ax = plt.subplots()
frames = []
strain = []
line1, = ax.plot(frames, strain, "+")
ax.set(xlabel="Image", ylabel="Strain [%]")
ax.grid()

imshow_scaled("Images", imref_cpy)

console.print(Rule())

# Loop for every file
for (nframe, file) in track(enumerate(filenames[1:]), total=len(filenames)-1, description="Tracking Frames: "):
    # Read image and track the rois
    im = cv.imread(file)
    (success1, ROI1) = tracker1.update(im)
    (success2, ROI2) = tracker2.update(im)
    # Check to see if the tracking was a success
    if success1 and success2:
        # Current length
        l = compute_length(ROI1, ROI2)
        # Update frames and strain values
        frames.append(nframe)
        strain.append(100*(l-l0)/l0)
        # Update plots and draw rois and line
        update_plot(line1, ax, fig, frames, strain)
        draw_roi(im, ROI1)
        draw_roi(im, ROI2)
        draw_segment(im, ROI1, ROI2)
        imshow_scaled("Images", im)
        # Wait 1ms for keypress
        key = cv.waitKey(1)
        if key == 27: # if ESC is pressed, exit loop
            break
    else:
        console.print("ERROR: tracking failed", style="red bold")
        break


# Write results
console.print(Rule())
console.print("Select result file", style="green bold")
ft = [("CSV files", "*.csv")]
outfile = filedialog.asksaveasfilename(title = "Result file",  initialfile = "tracking.csv", initialdir = imdir, filetypes = ft, defaultextension = ft)
with open(outfile, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Strain [%]"])
    for (t, e) in zip(frames, strain):
        writer.writerow([t, e])

console.print(Rule())
console.print("Success! Exit...", style="green bold")

cv.destroyAllWindows()