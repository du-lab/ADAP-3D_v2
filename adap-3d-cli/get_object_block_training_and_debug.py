""" get object block for training and debug

-not part of the workflow
-used to debug and get object blocks to train model on

"""

import numpy as np
import pandas as pd
import params
from get_object_block_imgs import get_image_for_blocks
from mzml_reader import extract_mzvals, extract_timevals, extract_intensities
import matplotlib.pyplot as plt
import matplotlib
import math

plt.rcParams["figure.figsize"] = (3 ,7)
matplotlib.use("Agg")
import os

# Helper function for when attribute values are too rounded in order to locate them in an array
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def find_max_list(list):
    list_len = [len(i) for i in list]
    print(max(list_len))

# Test function to check similarities between CSV files
# No use to workflow, just extra code
def check_similarities_between_v1_v2():
    newdataframe = []
    dataFramePrevEvaluation = pd.read_csv(r"C:\Users\jerry\Desktop\DCSM.csv")
    dataFrameV2Prediction = pd.read_csv(params.results_path + "\ADAP-3D-Predictions.csv")
    predictionsArrTime = dataFramePrevEvaluation['retention_time']
    predictionsArrMz = dataFramePrevEvaluation['mz']
    predictionsArrV2Time = dataFrameV2Prediction['Retention Time']
    predictionsArrV2Mz = dataFrameV2Prediction['M/Z']


    for time in range(len(predictionsArrV2Mz)):
        check = 0
        for time1 in range(len(predictionsArrMz)):
            if(check == 0 and math.isclose(predictionsArrMz[time1], predictionsArrV2Mz[time], rel_tol = 0.005) and math.isclose(predictionsArrTime[time1], predictionsArrV2Time[time], rel_tol = 0.05)):
                newdataframe.append([predictionsArrV2Mz[time], predictionsArrV2Time[time]])
                check = 1

    finaldataframe = pd.DataFrame(newdataframe, columns=['M/Z', 'Retention Time'])
    finaldataframe.to_csv(params.results_path + "\ADAP-3D-Predictions-Similarities.csv")

def get_image_for_blocks2(profile_file_mzML, window_mz=60, window_rt=300, timetoignoreL = 2, timetoignoreR = 20, debugimageversion = False):
    dataFramePrevEvaluation = pd.read_csv(r"C:\Users\jerry\Desktop\DCSM.csv")
    dataFrameV2Prediction = pd.read_csv(params.results_path + "\ADAP-3D-Predictions.csv")
    confirmedpeaks = pd.read_csv(r"C:\Users\jerry\Desktop\confirmed_peaks.csv")
    mzvals = list(map(float, confirmedpeaks['m/z']))
    timevals = confirmedpeaks['rt']


    predictionsArrTime = dataFramePrevEvaluation['retention_time']
    predictionsArrMz = dataFramePrevEvaluation['mz']
    predictionsArrV2Time = dataFrameV2Prediction['Retention Time']
    predictionsArrV2Mz = dataFrameV2Prediction['M/Z']
    #for timev1 in range(len(predictionsArrTime)):
        #predictionsArrTime = round(predictionsArrTime[timev1], 1)
    #for mzv1 in range(len(predictionsArrMz)):
        #predictionsArrMz = round(predictionsArrMz[mzv1], 2)
    #for timev2 in range(len(predictionsV2ArrTime)):
        #predictionsArrTime = round(predictionsArrV2Time[timev1], 1)
    #for mzv2 in range(len(predictionsArrV2Mz)):
        #predictionsArrMz = round(predictionsArrV2Mz[mzv1], 2)


    inputfile = profile_file_mzML


    ## Scanning the raw mzML file into array data ##
    scan_t = extract_timevals(inputfile)
    mz_list = extract_mzvals(inputfile, 0, len(scan_t))
    intensity_list = extract_intensities(inputfile, 0, len(scan_t))
    cnt = 0
    ## Cleaning up data into minutes##
    if (len(scan_t) > 2 and scan_t[1] - scan_t[0] > 0.1):  ## if gap >0.1: second..
        scan_t = [i / 60.0 for i in scan_t]

    ## Extract image for each CWT detected peak, Can adjust to take blocks from entire mzML file if needed ##
    #for time in range(pos_time_val1+window_rt, pos_time_val2-window_rt, window_rt-60):
        #indexLargestMZ = len(max(mz_list, key=len))
    mzarr = [164.0909, 189.0875, 153.0664, 181.0613, 140.0348, 166.0728, 216.0427, 222.9960, 220.1185, 205.0977, 164.0712, 373.0955, 373.0955, 197.0614, 155.0167, 209.0814, 224.1287, 243.0770, 286.1113, 231.1596,  245.1753, 245.1753,475.3900   ]
    timearr = [0.66, 1.17, 1.94, 2.31, 2.65, 2.94, 3.25, 3.90, 4.00, 4.26, 6.08, 6.90, 7.15, 7.97, 8.40, 9.68, 9.88, 10.83, 11.18, 11.54,11.56, 12.28, 14.25 ]

    #Add a check to remove rt before and after for noise
    timetoignoreleft = find_nearest(scan_t, timetoignoreL)
    timetoignoreright = find_nearest(scan_t, timetoignoreR)
    for time in range(len(timevals)):

        # Find index and length of longest mzarr in time range so no mz values are left out.
        # All mzarrs start and end at the same values but some arrays contain more values in between, so we need to include those
        mz0 = mzvals[time]
        rt0 = timevals[time]

        cnt += 1
        # Create indexes of time of entire mzarr
        pos_rt = scan_t.index(find_nearest(scan_t, rt0))

        # Adjust index boundaries to create a 50x130 image that doesn't overlap since we want all mz values based from the largest mzarr to include all the possible values we can
        # (may not be in the center of the image)
        pos_rt1 = pos_rt - window_rt
        pos_rt2 = pos_rt + window_rt

        if pos_rt2 >= len(scan_t):
            pos_rt1 = len(scan_t) - window_rt * 2
            pos_rt2 = len(scan_t)
        elif pos_rt1 < 0:
            pos_rt1 = 0
            pos_rt2 = 2 * window_rt

        allmzintime = []
        allintensityintime = []
        for i in range(pos_rt1, pos_rt2):
            for j in range(len(mz_list[i])):
                allmzintime.append(mz_list[i][j])
                allintensityintime.append(intensity_list[i][j])

        tempmz = []

        for i in np.arange(allmzintime[0], allmzintime[len(allmzintime) - 1], 0.0005):
            tempmz.append(i)

        tempmz[len(tempmz)-1] = tempmz[len(tempmz) - 1] + 0.1
        binnedvals = list(pd.cut(np.array(allmzintime), tempmz, right = False, labels = False))

        newbinnedmz = []
        newbinnedintensity = []
        templistmz = []
        templistintensity = []

        for i in range(1, len(binnedvals)):

            if(binnedvals[i] == binnedvals[i-1]):
                templistmz.append(allmzintime[i])
                templistintensity.append(allintensityintime[i])
            elif len(templistmz) > 0:
                newbinnedmz.append(allmzintime[i])
                newbinnedintensity.append(allintensityintime[i])
            else:
                newbinnedintensity.append(max(templistintensity))
                newbinnedmz.append(templistmz[templistintensity.index(max(templistintensity))])
                templistmz = []
                templistintensity = []

        newallmzintime = newbinnedmz


        # Set index of mz value in the longestmzarr and make windows
        pos_mzorig = newallmzintime.index(find_nearest(newallmzintime, mz0))
        pos_mz1orig = pos_mzorig - window_mz
        pos_mz2orig = pos_mzorig + window_mz
        mzvalarrsplit = []
        intensitysplit = []
        # Go through every m/z value in the longest mzarr that is within 50 values and saving them to an array to be used in image creation
        # Because the m/z values are not always in the same index position, we will use .index and find_nearest()
        # for mzmz in range(window_mz, -1, -1):
        # mzvalarrsplit.append(round(allmzintime[int(pos_rt)][int(pos_mzorig - mzmz)], 6))
        # for mzmz in range(1, window_mz):
        # mzvalarrsplit.append(round(allmzintime[int(pos_rt)][int(pos_mzorig + mzmz)], 6))
        for mzmz in range(pos_mzorig - window_mz, pos_mzorig + window_mz):
            mzvalarrsplit.append(newallmzintime[mzmz])
            intensitysplit.append(allintensityintime[mzmz])
        # Creating the image from calculated ranges
        area = []
        arrofmzrttemp = []
        # Looping through all times in range
        """
        for t in range(pos_rt1, pos_rt2):
            htgrids = intensity_list[t]
            grids_part = []
            for m in range(len(mzvalarrsplit)):
                # Going through every m/z value (50 total) in mzvalarrsplit and getting the intensity of that mzvalue in each different time value, saving it to an array
                # If cannot locate mzvalue in that mzarr at time "t" append intensity of 0
                if(abs(mzvalarrsplit[m] - mz_list[t][mz_list[t].index(find_nearest(mz_list[t], mzvalarrsplit[m]))]) <= 0.000005):
                    pos_mz = mz_list[t].index(find_nearest(mz_list[t], mzvalarrsplit[m]))
                    grids_part.append(np.log10(htgrids[pos_mz]))
                else:
                    grids_part.append(0)
            area.append(grids_part)
            """
        for t in range(pos_rt1, pos_rt2):
            intensitygrid = intensity_list[t]
            mzgrid = mz_list[t]
            mzleft = mzgrid.index(find_nearest(mzgrid, newallmzintime[pos_mzorig - window_mz]))
            mzright = mzgrid.index(find_nearest(mzgrid, newallmzintime[pos_mzorig + window_mz]))
            grids_part = []
            tempvalarrsplit = mzvalarrsplit
            tempvalarrsplit[len(tempvalarrsplit) - 1] = tempvalarrsplit[len(tempvalarrsplit) - 1] + 0.001
            tempvalarrsplit[0] = tempvalarrsplit[0] - 0.001
            binnedvals2 = list(pd.cut(np.array(mzgrid[mzleft: mzright + 1]), tempvalarrsplit, right=False, labels=False))
            finalbins = []

            for i in range(window_mz*2):
                temparr = []
                for j in range(len(binnedvals2)):
                    if(binnedvals2[j] == i):
                        temparr.append(intensitygrid[mzleft + j])
                if(len(temparr) == 0):
                    finalbins.append(0)
                else:
                    finalbins.append(max(temparr))

            area.append(finalbins)
            """
            for m in range(len(finalmzlist)):
                try:
                    pos_mz = mz_list[t].index(finalmzlist[m])
                    grids_part.append(np.log10(intensitygrid[pos_mz]))
                except:
                    grids_part.append(0)
            area.append(grids_part)
            """

        # Render array as an image and save it to local drive path
        plt.imshow(area, cmap='Greys', aspect='auto')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(r'.\mzml-img-blocks-new-test-5\ ' + "Block # " + str(cnt) + '.png')
        plt.close('all')

        """
        Add color bar, etc
        """
        # fig, ax = plt.subplots()
        # fig.set_figheight(1.3)
        # fig.set_figwidth(0.5)
        # img = ax.imshow(area, cmap='Greys', aspect= 'auto', dpi = 3000)
        # plt.imshow(area, cmap = 'Greys', aspect = 'auto')
        ## Label each pixel with Intensity, *uncomment* to use ##
        # for y in range(0, window_rt*2):
        # for x in range(0, window_mz*2):
        # text = ax.text(x, y, round(area[y][x], 3),
        # ha="center", va="center", color="w")

        ## Setting ticks and values according to the window mz and window rt ##
        xticks = []
        xlabels = []
        yticks = []
        ylabels = []
        # for xtick in range(2*window_mz):
        # xticks.append(xtick)
        # for xlabel in range(2*window_mz):
        # xlabels.append(round(mzvalarrsplit[xlabel], 3))
        # for ytick in range(2*window_rt):
        # yticks.append(ytick)
        # for ylabel in range(2*window_rt):
        # ylabels.append(round(scan_t[pos_rt1+ylabel], 3))
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        # hspace=0, wspace=0)
        # plt.margins(0, 0)
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xlabels)
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(ylabels)
        #every_nth = 6
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        # if n % every_nth != 0:
        # label.set_visible(False)
        # for n, label in enumerate(ax.yaxis.get_ticklabels()):
        # if n % every_nth != 0:
        # label.set_visible(False)
        # plt.locator_params(nbins = 3)

        ## Create title, axes labels, and colorbar ##
        # fig.suptitle("Prediction " + str(cnt) + " M/Z: " + str(mz0) + "  RT: " + str(rt0))
        # plt.xlabel('M/Z')
        # plt.ylabel('Retention Time')
        # fig.colorbar(img)
        # plt.savefig(r'.\mzml-img-blocks\ ' + "Block # " + str(cnt) + '.png')
        # plt.savefig(r'.\debug-imgs\ ' + "Block # " + str(cnt) + '.png')
        # plt.close('all')
        # breakpoint()





