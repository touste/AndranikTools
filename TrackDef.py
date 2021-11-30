import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import dlib
import csv
import utils
from tkinter import Tk, filedialog
from rich.console import Console    
from rich.rule import Rule
from rich.progress import track


console = Console()

# Init tkinter for file selection
root = Tk() # pointing root to Tk() to use it as Tk() in program.
root.withdraw() # Hides small tkinter window.
root.attributes("-topmost", True) # Opened windows will be active. above all windows despite of selection.



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
imrefbw = cv.cvtColor(imref, cv.COLOR_BGR2GRAY)

# Create window
cv.namedWindow("Images", cv.WINDOW_NORMAL)
utils.imshow_scaled("Images", imref)

while True:
    # Select two ROIs and display them
    console.print(Rule())
    console.print("Select first marker", style="green bold")
    console.print("Start by clicking at the center of the marker, then grow a square around it. The square should enclose the marker.", style="italic")
    ROI1 = cv.selectROI("Images", imref, fromCenter=True, showCrosshair=True)
    utils.draw_roi(imref, ROI1)
    console.print(Rule())
    console.print("Select second marker", style="green bold")
    console.print("Start by clicking at the center of the marker, then grow a square around it. The square should enclose the marker.", style="italic")
    ROI2 = cv.selectROI("Images", imref, fromCenter=True, showCrosshair=True)
    utils.draw_roi(imref, ROI2)
    # Display segment between rois
    utils.draw_segment(imref, ROI1, ROI2)
    # Select ROI for clamp displacement measurement
    console.print(Rule())
    console.print("Select a feature on the upper clamp", style="green bold")
    console.print("Start by clicking at the center of the marker, then grow a square around it. The square should enclose the feature.", style="italic")
    ROIcorner = cv.selectROI("Images", imref, fromCenter=True, showCrosshair=True)
    utils.draw_roi(imref, ROIcorner)
    utils.imshow_scaled("Images", imref)
    console.print(Rule())
    console.print("Press ENTER to validate, else press any other key to start again.", style="yellow bold blink")
    key = cv.waitKey(0)
    if key == 13: # if ENTER is pressed, exit loop
        break



# Initial length and clamp ROI center
l0 = utils.compute_length(ROI1, ROI2)
center0 = utils.ROIcenter(ROIcorner)

# Init trackers
tracker1 = dlib.correlation_tracker()
tracker1.start_track(imrefbw, utils.bb_to_rect(ROI1))
tracker2 = dlib.correlation_tracker()
tracker2.start_track(imrefbw, utils.bb_to_rect(ROI2))
tracker3 = dlib.correlation_tracker()
tracker3.start_track(imrefbw, utils.bb_to_rect(ROIcorner))

# Init strain and displacement plot
plt.ion()
fig, ax1 = plt.subplots()
frames = []
strain = []
disp = []
line1, = ax1.plot(frames, strain, "ro")
ax1.set_xlabel("Image")
ax1.set_ylabel("Strain [%]", color="red")
ax1.tick_params(axis="y", labelcolor="red")

ax2 = ax1.twinx()
ax2.set_ylabel("Displacement [px]", color="blue")
line2, = ax2.plot(frames, disp, "b+")
ax2.tick_params(axis="y", labelcolor="blue")
fig.tight_layout()

utils.imshow_scaled("Images", imref)

console.print(Rule())

# Output video
vid = cv.VideoWriter(imdir + '/tracking.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (imref.shape[1], imref.shape[0]))

# Loop for every file
for (nframe, file) in track(enumerate(filenames[1:]), total=len(filenames)-1, description="Tracking Frames: "):
    # Read image and track the rois
    im = cv.imread(file)
    imbw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    tracker1.update(imbw)
    ROI1 = utils.rect_to_bb(tracker1.get_position())
    tracker2.update(imbw)
    ROI2 = utils.rect_to_bb(tracker2.get_position())
    tracker3.update(imbw)
    ROIcorner = utils.rect_to_bb(tracker3.get_position())

    # Current length
    l = utils.compute_length(ROI1, ROI2)
    # Update frames and strain values
    frames.append(nframe)
    strain.append(100*(l-l0)/l0)
    disp.append(utils.compute_disp(utils.ROIcenter(ROIcorner), center0))
    # Update plots and draw rois and line
    utils.update_plot(line1, ax1, line2, ax2, fig, frames, strain, disp)
    utils.draw_roi(im, ROI1)
    utils.draw_roi(im, ROI2)
    utils.draw_roi(im, ROIcorner)
    utils.draw_segment(im, ROI1, ROI2)
    utils.imshow_scaled("Images", im)
    vid.write(im)
    # Wait 1ms for keypress
    key = cv.waitKey(1)
    if key == 27: # if ESC is pressed, exit loop
        break

vid.release()

# Write results
console.print(Rule())
console.print("Select result file", style="green bold")
ft = [("CSV files", "*.csv")]
outfile = filedialog.asksaveasfilename(title = "Result file",  initialfile = "tracking.csv", initialdir = imdir, filetypes = ft, defaultextension = ft)
with open(outfile, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Strain [%]", "Displacement [px]"])
    for (t, e, d) in zip(frames, strain, disp):
        writer.writerow([t, e, d])

console.print(Rule())
console.print("Success! Exit...", style="green bold")

cv.destroyAllWindows()