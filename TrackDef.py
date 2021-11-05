import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import dlib
import csv
from scipy import interpolate

import utils
from rich.console import Console    
from rich.rule import Rule
from rich.progress import track
import csv



root_folder = '/home/pierrat/SeaDrive/My Libraries/C4BIO/Phase 2/'

samplename = 'S3'






# Acquisition rate
framerate = 20

# clamp velocity [mm/s]
clamp_velocity = 1.3

console = Console()



# Select image directory 
imdir = root_folder + 'MarkerTracking/' + samplename


# Get bmp files in imdir
filenames = [img for img in glob.glob(imdir + "/*.bmp")]
filenames.sort()

# Read scale
xmlfile = glob.glob(imdir + "/*.cihx")[0]
pxtomm = utils.read_scale(xmlfile)


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
    ROImachine = cv.selectROI("Images", imref, fromCenter=True, showCrosshair=True)
    utils.draw_roi(imref, ROImachine)
    utils.imshow_scaled("Images", imref)
    
    # Select clamp to clamp distance
    console.print(Rule())
    console.print("Draw a line for clamp to clamp distance measurement", style="green bold")
    
    lines = []
    cv.setMouseCallback("Images", utils.lineinput, (lines, imref, imref.copy(), lambda : None, "Images"))

    # Wait until one line has been selected
    while True:
        if len(lines) > 0:
            if len(lines[-1]) == 2:
                cv.setMouseCallback("Images", lambda *args : None)
                break
        cv.waitKey(10)
    
    l = lines[-1]
    clamtoclamp_t1 = abs(l[0][1]-l[1][1])*pxtomm
    
    console.print(Rule())
    console.print("Press ENTER to validate, else press any other key to start again.", style="yellow bold blink")
    key = cv.waitKey(0)
    if key == 13: # if ENTER is pressed, exit loop
        break



# Initial positions
X0_machine = utils.ROIcenter(ROImachine)

# Init trackers
tracker1 = dlib.correlation_tracker()
tracker1.start_track(imrefbw, utils.bb_to_rect(ROI1))
tracker2 = dlib.correlation_tracker()
tracker2.start_track(imrefbw, utils.bb_to_rect(ROI2))
tracker3 = dlib.correlation_tracker()
tracker3.start_track(imrefbw, utils.bb_to_rect(ROImachine))










utils.imshow_scaled("Images", imref)

console.print(Rule())

# Output video
vid = cv.VideoWriter(imdir + '/tracking.mp4', cv.VideoWriter_fourcc(*'mp4v'), framerate, (imref.shape[1], imref.shape[0]))

# Outputs
times = [0.]
markerdist = [utils.compute_roidistance(ROI1, ROI2)*pxtomm]
machinedisp = [0.]

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
    ROImachine = utils.rect_to_bb(tracker3.get_position())
    
    
    # Update displacements
    markerdist.append(utils.compute_roidistance(ROI2, ROI1)*pxtomm)
    machinedisp.append((X0_machine[1] - utils.ROIcenter(ROImachine)[1])*pxtomm)


    # Update frames
    times.append((nframe+1)/framerate)

    # Update plots and draw rois and line
    utils.draw_roi(im, ROI1)
    utils.draw_roi(im, ROI2)
    utils.draw_roi(im, ROImachine)
    utils.draw_segment(im, ROI1, ROI2)
    utils.imshow_scaled("Images", im)
    vid.write(im)
    # Wait 1ms for keypress
    key = cv.waitKey(1)
    if key == 27: # if ESC is pressed, exit loop
        break

vid.release()

cv.destroyAllWindows()

times = np.array(times)
markerdist = np.array(markerdist)
machinedisp = np.array(machinedisp)



# Plot results

# Init plot
plt.ion()
fig, ax1 = plt.subplots(figsize=(1600/100, 900/100), dpi=100)


sc1 = ax1.scatter(times, markerdist, marker="o", c="r")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Marker Distance [mm]", color="red")
ax1.tick_params(axis="y", labelcolor="red")


ax2 = ax1.twinx()
ax2.set_ylabel("Machine Displacement [mm]", color="blue")
sc2 = ax2.scatter(times, machinedisp, marker="+", c="b")
ax2.tick_params(axis="y", labelcolor="blue")
fig.tight_layout()



input("Looks good? Press Enter to write raw results...")


plt.savefig(imdir + '/figure_tracking.png')


# Write raw results
with open(imdir + '/tracking_raw.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time [s]", "Marker distance [mm]", "Machine displacement [mm]"])
    for (t, d12, d) in zip(times, markerdist, machinedisp):
        writer.writerow([t, d12, d])




# Synchronize preconditionning between Instron and Camera data based on start of cycle #2

console.print(Rule())
console.print("Click on x value at the beginning of cycle #2", style="green bold")

idxapproxstart_camera = int(round(plt.ginput(1, timeout = -1)[0][0]*framerate))

idxstart_camera = np.argmin(machinedisp[idxapproxstart_camera-10:idxapproxstart_camera+10])+idxapproxstart_camera-10

cycle2start_camera = times[idxstart_camera]

clamtoclamp_t0 = clamtoclamp_t1 + machinedisp[idxstart_camera]

ax1.axvline(x=cycle2start_camera)


# Select instron file

instronfile = root_folder + 'InstronData/' + samplename + '/Specimen_RawData_1.csv'

def load_adjusted_instron_precond(instronfile, timestamp_camera, startcycle, times):

    disp_instron = []
    force_instron = []
    times_instron = []
    ncycles = []
    with open(instronfile, newline='', encoding='ISO-8859-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for (i,row) in enumerate(spamreader):
            if i>1:
                disp_instron.append(float(row[0]))
                force_instron.append(float(row[1]))
                times_instron.append(float(row[3]))
                ncycles.append(float(row[4]))
                
    timestamp_instron = times_instron[np.argmax(np.array(ncycles)>=startcycle)]

    delta = timestamp_camera - timestamp_instron

    times_instron = np.array(times_instron) + delta

    fd = interpolate.interp1d(times_instron, disp_instron, bounds_error=False)
    ff = interpolate.interp1d(times_instron, force_instron, bounds_error=False)
    
    disp_instron = fd(times)
    force_instron = ff(times)
    
    return disp_instron, force_instron


disp_instron_precond, force_instron_precond = load_adjusted_instron_precond(instronfile, cycle2start_camera, 1, times)
        
machinedisp = machinedisp + (disp_instron_precond[idxstart_camera] - machinedisp[idxstart_camera])

fig, ax1 = plt.subplots(figsize=(1600/100, 900/100), dpi=100)
ax1.plot(times, disp_instron_precond, "-+", label='From Instron')
ax1.plot(times, machinedisp, "-o", label='From Tracking')

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Machine displacement [mm]")

ax1.legend()


input("Looks good? Press Enter to continue...")

plt.savefig(imdir + '/figure_sync_precond.png')




# Synchronize final ramp between Instron and Camera data based on initial displacement

instronfile = root_folder + 'InstronData/' + samplename + '/Specimen_RawData_2.csv'


def load_adjusted_instron_final(instronfile, machinedisp, times):

    disp_instron = []
    force_instron = []
    times_instron = []
    with open(instronfile, newline='', encoding='ISO-8859-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for (i,row) in enumerate(spamreader):
            if i>1:
                disp_instron.append(float(row[0]))
                force_instron.append(float(row[1]))
                times_instron.append(float(row[3]))
                    

    zerodisp = disp_instron[0]

    idxstart_camera = len(machinedisp) - np.argmax(np.array(machinedisp)[::-1]<=(zerodisp+1)) - 1 - int(round(framerate/clamp_velocity))
    
    timestamp_camera = times[idxstart_camera]

    delta = timestamp_camera

    times_instron = np.array(times_instron) + delta
    
    fd = interpolate.interp1d(times_instron, disp_instron, bounds_error=False)
    ff = interpolate.interp1d(times_instron, force_instron, bounds_error=False)
    
    disp_instron = fd(times)
    disp_instron[idxstart_camera] = zerodisp
    force_instron = ff(times)
    force_instron[idxstart_camera] = 0.1
        
    return disp_instron, force_instron, idxstart_camera



disp_instron_final, force_instron_final, idxstart_final  = load_adjusted_instron_final(instronfile, machinedisp, times)
        
fig, ax1 = plt.subplots(figsize=(1600/100, 900/100), dpi=100)
ax1.plot(times, disp_instron_final, "-+", label = 'From Instron')
ax1.plot(times, machinedisp, "-+", label = 'From Tracking')

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Machine displacement [mm]")

ax1.legend()


input("Looks good? Press Enter to continue...")

plt.savefig(imdir + '/figure_sync_final.png')


stacked = np.array([force_instron_final, force_instron_precond])
force_instron = np.nansum(stacked, axis=0)
force_instron[np.all(np.isnan(stacked), axis=0)] = np.NaN

stacked = np.array([disp_instron_final, disp_instron_precond])
disp_instron = np.nansum(stacked, axis=0)
disp_instron[np.all(np.isnan(stacked), axis=0)] = np.NaN


stretch = markerdist / markerdist[idxstart_final]

fig, ax1 = plt.subplots(figsize=(1600/100, 900/100), dpi=100)
ax1.plot(stretch, force_instron, "-+")

ax1.set_xlabel("Stretch [-]")
ax1.set_ylabel("Force [N]")

input("Looks good? Press Enter to continue...")

plt.savefig(imdir + '/figure_forcevsstretch_all.png')



time_out = times[idxstart_final:]-times[idxstart_final]
u_out = disp_instron_final[idxstart_final:]
f_out = force_instron_final[idxstart_final:]
stretch_out = stretch[idxstart_final:]

idxmax = np.argmax(f_out)
f_ult = f_out[idxmax]
stretch_ult = stretch_out[idxmax]


fig, ax1 = plt.subplots(figsize=(1600/100, 900/100), dpi=100)
ax1.plot(stretch_out, f_out, "-+")

ax1.axvline(x=stretch_ult)
ax1.axhline(y=f_ult)

slopes = []
interc = []
stretchvals = np.arange(1.05, stretch_ult, 0.05)
for l in stretchvals:
    mask = (stretch_out>(l-0.025)) & (stretch_out<(l+0.025))
    x = stretch_out[mask]
    y = f_out[mask]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    slopes.append(m)
    interc.append(c)
    ax1.plot(x, m*x + c)



ax1.set_xlabel("Stretch [-]")
ax1.set_ylabel("Force [N]")

input("Looks good? Press Enter to continue...")

plt.savefig(imdir + '/figure_forcevsstretch_final.png')





# Write processed results
with open(imdir + '/tracking_post.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["F_ult [N]", "λ_ult [-]"])
    writer.writerow([f_ult, stretch_ult])
    writer.writerow([])
    writer.writerow(["Clamp-to-clamp distance at t0 [mm]", "Clamp-to-clamp distance at t1 [mm]"])
    writer.writerow([clamtoclamp_t0, clamtoclamp_t1])
    writer.writerow([])
    writer.writerow(["λ_i [-]", "E_i [N]"])
    for (l, e) in zip(stretchvals, slopes):
        writer.writerow([l, e])
    writer.writerow([])
    writer.writerow(["Time [s]", "u(t) [mm]", "f(t) [N]", "λ_t [-]"])
    for (t, u, f, l) in zip(time_out, u_out, f_out, stretch_out):
        writer.writerow([t, u, f, l])