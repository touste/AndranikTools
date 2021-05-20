import numpy as np
import sys
import cv2 as cv
from tkinter import Tk, filedialog
from rich.prompt import Prompt
from rich.console import Console    
from rich.rule import Rule
from rich.text import Text

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


# Select image 
console.print(Rule())
console.print("Select calibration image", style="green bold")
console.print("This is the image with the checkerboard on it.", style="italic")
imfile = filedialog.askopenfilename(title = "Select calibration image", filetypes=[("BMP files", "*.bmp")])


# Grid params
console.print(Rule())
ncols = Prompt.ask(Text("Enter the number of inner corners in the horizontal direction", style="green"), default="9")
ncols = int(ncols)
nrows = Prompt.ask(Text("Enter the number of inner corners in the vertical direction", style="green"), default="7")
nrows = int(nrows)
grid_delta = Prompt.ask(Text("Enter the size of each square of the checkerboard in mm", style="green"), default="2")
grid_delta = float(grid_delta)

# Read image and convert to BW
im = cv.imread(imfile)
imbw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Detect checkerboard
console.print(Rule())
console.print("Detecting checkerboard...", style="yellow bold")
detected, corners = cv.findChessboardCornersSB(imbw, (nrows, ncols))
if not detected:
    sys.exit("ERROR: checkerboard detection failed")
cv.drawChessboardCorners(im, (nrows, ncols), corners, detected)

# Draw checkerboard
cv.namedWindow("Checkerboard", cv.WINDOW_NORMAL)
imshow_scaled("Checkerboard", im)

console.print("Press ENTER to validate and write the calibration data, else press any other key to abort.", style="yellow bold blink")
key = cv.waitKey(0)
cv.destroyAllWindows()

if key == 13: # if ENTER is pressed, continue
    
    # Reshape corners as matrices
    corners = np.squeeze(corners)
    x,y = corners[:,0], corners[:,1]
    x = np.reshape(x, (ncols, nrows))
    y = np.reshape(y, (ncols, nrows))

    # Get distances between corners
    dxx, dxy = np.gradient(x)
    dyx, dyy = np.gradient(y)

    dX = np.sqrt(dxx**2 + dyx**2) 
    dY = np.sqrt(dxy**2 + dyy**2) 
    
    # Take mean value
    delta_px = np.mean(np.vstack([dX, dY]))

    # Get calib scale
    pxtomm = grid_delta/delta_px

    console.print(Rule())
    console.print("The scale is: 1 px =", "{:.4f}".format(pxtomm), "mm", style="yellow bold italic")
    
    # Write to file
    console.print(Rule())
    console.print("Select result file", style="green bold")
    ft = [("TXT files", "*.txt")]
    outfile = filedialog.asksaveasfilename(title = "Result file", initialfile = "calib.txt", filetypes = ft, defaultextension = ft)
    with open(outfile, "w") as file:
        file.write(str(pxtomm))
    
    console.print(Rule())
    console.print("Success! Exit...", style="green bold")
    
else:
    sys.exit("ERROR: checkerboard not valid")

