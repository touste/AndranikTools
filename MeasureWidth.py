import numpy as np
import cv2 as cv
from tkinter import Tk, filedialog
import utils
import csv
from rich.console import Console    
from rich.rule import Rule

console = Console()


    
# Init tkinter for file selection
root = Tk() # pointing root to Tk() to use it as Tk() in program.
root.withdraw() # Hides small tkinter window.
root.attributes("-topmost", True) # Opened windows will be active. above all windows despite of selection.



# Select image 
console.print(Rule())
console.print("Select reference image", style="green bold")
console.print("This is the image that will be used for measuring the initial width.", style="italic")
imfile = filedialog.askopenfilename(title = "Select referece image", filetypes=[("BMP files", "*.bmp")])

console.print(Rule())
console.print("Select calibration file", style="green bold")
console.print("This is the file containing the pixel size in mm.", style="italic")
calibfile = filedialog.askopenfilename(title = "Calibration file", filetypes=[("TXT files", "*.txt")])

# Get calibration scale from px to mm
with open(calibfile, "r") as file:
    vstr = file.read()
pxtomm = float(vstr)


# Read image and select ROI
im = cv.imread(imfile)
console.print(Rule())
console.print("Select the sample region", style="green bold")
console.print("You should select the full sample.", style="italic")

cv.namedWindow("Reference", cv.WINDOW_NORMAL)
utils.imshow_scaled("Reference", im)
ROI = cv.selectROI("Reference", im, fromCenter=False, showCrosshair=False)

# Crop image and display
imcrop = im[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
imcrop_disp = imcrop.copy()
utils.imshow_scaled("Reference", imcrop_disp)


# Get sample axis
console.print(Rule())
console.print("Draw a line along the longitudinal direction of the sample", style="green bold")
console.print("Click and drag to draw a line that will be used to correct the vertical axis by rotating the image.", style="italic")
lines = []
cv.setMouseCallback("Reference", utils.lineinput, (lines, imcrop_disp, imcrop_disp.copy(), lambda : None, "Reference"))

# Wait until one line has been selected
while True:
    if len(lines) > 0:
        if len(lines[-1]) == 2:
            cv.setMouseCallback("Reference", lambda *args : None)
            break
    cv.waitKey(10)

# Compute rotation angle
theta = np.arctan2(lines[0][1][1]-lines[0][0][1], lines[0][1][0]-lines[0][0][0])

# Rotate image and display
imrot_disp = utils.rotateImage(imcrop_disp, np.degrees(theta + np.pi/2))
imrot = utils.rotateImage(imcrop, np.degrees(theta + np.pi/2))

utils.imshow_scaled("Reference", imrot_disp)
cv.waitKey(1000)



# Buffer images and masks for filters

imnonfilt = cv.cvtColor(imrot, cv.COLOR_BGR2GRAY) # Unfiltered image in bw
imnonfilt_bgr = imrot.copy() # Unfiltered image in bgr

mask = np.zeros(imnonfilt.shape, np.uint8) # Mask
mask_disp = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # Mask for display, blended with original image

utils.imshow_scaled("Reference", mask_disp)

console.print(Rule())
console.print("Adjust the parameters for thresholding", style="green bold")
console.print("The boundaries of the sample should match the thresholding mask as best as possible.", style="italic")
console.print("[bold] - Block size[/bold] adjusts the area of influence of thresholding.", style="italic")
console.print("[bold] - Threshold[/bold] adjusts the thresholding value.", style="italic")
console.print("[bold] - Smooth[/bold] controls the radius of a median filter.", style="italic")
console.print("Then, press any key to continue...", style="italic")


# Image processing: apply median filter, then adaptive threshold
def process_image(v):
    rad = cv.getTrackbarPos(tb_gaussfilt,"Reference")
    if rad > 0:
        if rad % 2 != 1: rad=rad+1
        cv.medianBlur(imnonfilt, rad, dst=mask)
    else:
        mask[:] = imnonfilt
    thresh = cv.getTrackbarPos(tb_thresh,"Reference")-15
    bs = cv.getTrackbarPos(tb_bs,"Reference")
    if bs % 2 != 1: bs=bs+1
    if bs <= 1: bs=3
    cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, bs, thresh, dst=mask)
    mask_disp[:,:,0] = 0
    mask_disp[:,:,1] = mask
    mask_disp[:,:,2] = 0
    cv.addWeighted(mask_disp, 0.1, imnonfilt_bgr, 1, 0, dst=mask_disp)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(mask_disp, contours, -1, (0,255,0), 1)
    utils.imshow_scaled("Reference", mask_disp)
    


# Create trackbars for tuning thresholding
tb_bs = "Block size"
cv.createTrackbar(tb_bs, "Reference", int(max(mask.shape)/10), max(mask.shape), process_image)
tb_thresh = "Threshold"
cv.createTrackbar(tb_thresh, "Reference", 15, 30, process_image)
tb_gaussfilt = "Smooth"
cv.createTrackbar(tb_gaussfilt, "Reference", 5, 30, process_image)

# Process once then display and wait for user to finish
process_image(0)
utils.imshow_scaled("Reference", mask_disp)
cv.waitKey(0)
cv.destroyWindow("Reference")

# Display final mask
cv.namedWindow("Final", cv.WINDOW_NORMAL)
utils.imshow_scaled("Final", mask_disp)

console.print(Rule())
console.print("Measure the width by drawing a line", style="green bold")
console.print("Click and drag to draw a line crossing the width of the sample. Repeat as many times as you wish. \nThen press any key to continue.", style="italic")

# Process lines: draw them, extract width length, ...
lines = []
widths = []
def drawwidth():
    l = lines[-1]
    row = int((l[0][1]+l[1][1])/2)
    startcol = min(l[0][0], l[1][0])
    endcol = max(l[0][0], l[1][0])
    if mask[row, startcol]>0: mask[:] = 255-mask[:]
    rowvals = mask[row, startcol:endcol+1]
    widthpx = sum(rowvals>0)
    truevals = np.where(rowvals>0)
    x0 = truevals[0][0] + startcol
    x1 = truevals[0][-1] + startcol
    cv.line(mask_disp, (x0, row), (x1, row), (0, 255, 255), 2)
    cv.circle(mask_disp, (x0, row), radius=2, color=(0, 255, 255), thickness=-1)
    cv.circle(mask_disp, (x1, row), radius=2, color=(0, 255, 255), thickness=-1)
    widthmm = widthpx*pxtomm
    widths.append(widthmm)
    widthtext = "{:.3f}".format(widthmm)
    textsize = cv.getTextSize(widthtext, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    textX = int((x0+x1)/2 - textsize[0]/2)
    textY = row-3
    cv.putText(mask_disp, widthtext, (textX, textY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
cv.setMouseCallback("Final", utils.lineinput, (lines, mask_disp, mask_disp.copy(), drawwidth, "Final"))

cv.waitKey(0)



 # Write to file
console.print(Rule())
console.print("Select result file", style="green bold")
ft = [("CSV files", "*.csv")]
outfile = filedialog.asksaveasfilename(title = "Result file",  initialfile = "width.csv", filetypes = ft, defaultextension = ft)
with open(outfile, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Width [mm]"])
    for s in widths:
        writer.writerow([s])
    writer.writerow(["Average [mm]"])
    writer.writerow([np.mean(widths)])
    writer.writerow(["St. dev [mm]"])
    writer.writerow([np.std(widths)])

console.print(Rule())
console.print("Success! Exit...")

cv.destroyAllWindows()