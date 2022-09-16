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
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import track
import mimetypes
mimetypes.init()
import sys


console = Console()

# Init tkinter for file selection
root = Tk() # pointing root to Tk() to use it as Tk() in program.
root.withdraw() # Hides small tkinter window.
root.attributes("-topmost", True) # Opened windows will be active. above all windows despite of selection.



# Select image directory 
console.print(Rule())
console.print("Select image directory", style="green bold")
console.print("This is the folder containing the video or the images.", style="italic")
imdir = filedialog.askdirectory(title = "Select image directory")


console.print(Rule())
ext = Prompt.ask(Text("Type the image or video extension", style="green"), default="bmp")

# Get files in imdir
filenames = [img for img in glob.glob(imdir + '/*.' + ext)]
filenames.sort()

mimestart = mimetypes.guess_type(filenames[0])[0]

if 'video' in mimestart:
    if len(filenames)>1:
        sys.exit("ERROR: More than one video file has been found!")
    cap = cv.VideoCapture(filenames[0])
    isvideo = True
    nimages = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
elif 'image' in mimestart:
    isvideo = False
    nimages = len(filenames)
else:
    sys.exit("ERROR: No video or image has been found!")


# Read first image
if isvideo:
    ret, imref = cap.read() 
else:
    imref = cv.imread(filenames[0])

imrefbw = cv.cvtColor(imref, cv.COLOR_BGR2GRAY)

# Create window
cv.namedWindow("Images", cv.WINDOW_NORMAL)
utils.imshow_scaled("Images", imref)
cv.waitKey(100)

# Get number of points of interest
console.print(Rule())
npoints = Prompt.ask(Text("How many points do you wish to track?", style="green"), default="2")
npoints = int(npoints)




while True:
    ROIS = []
    for i in range(npoints):
        # Select two ROIs and display them
        console.print(Rule())
        console.print("Select marker #" + str(i+1), style="green bold")
        console.print("Start by clicking at the center of the marker, then grow a square around it. The square should enclose the marker.", style="italic")
        ROI = cv.selectROI("Images", imref, fromCenter=True, showCrosshair=True)
        utils.draw_roi(imref, ROI)
        ROIS.append(ROI)
        
    utils.imshow_scaled("Images", imref)
    console.print(Rule())
    console.print("Press ENTER to validate, else press any other key to start again.", style="yellow bold blink")
    key = cv.waitKey(0)
    if key == 13: # if ENTER is pressed, exit loop
        break




# Init trackers
trackers = []
for r in ROIS:
    t = dlib.correlation_tracker()
    trackers.append(t)
    t.start_track(imrefbw, utils.bb_to_rect(r))


# Output video
vid = cv.VideoWriter(imdir + '/tracking.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (imref.shape[1], imref.shape[0]))


trackinfo = [[utils.ROIcenter(utils.rect_to_bb(t.get_position())) for t in trackers]]
# Loop for every file
for i in track(range(nimages), total=nimages-1, description="Tracking Frames: "):
    if isvideo:
        ret, im = cap.read()
        if not ret:
            break 
    else:
        im = cv.imread(filenames[i])
    # Read image and track the rois
    imbw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    tpos = []
    for (i,t) in enumerate(trackers):
        t.update(imbw)
        ROI = utils.rect_to_bb(t.get_position())
        utils.draw_roi(im, ROI)
        tpos.append(utils.ROIcenter(ROI))
    utils.imshow_scaled("Images", im)
    trackinfo.append(tpos)
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
    writer.writerow(["Frame"] + ["Marker " + str(i+1) + " " + xy +" position [px]" for i in range(len(trackers)) for xy in ["x" "y"]])
    for i in range(len(trackinfo)):
        writer.writerow([i+1] + [tt for t in trackinfo[i] for tt in t])

console.print(Rule())
console.print("Success! Exit...", style="green bold")

cv.destroyAllWindows()