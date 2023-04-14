# -*- coding: utf-8 -*-
"""
Analyse a USAF test target image, to determine the image's dimensions.

See: https://en.wikipedia.org/wiki/1951_USAF_resolution_test_chart

(c) Richard Bowman 2017, released under GNU GPL
    modified by Louis Ngo 2022
    modified by Jean-Marie Yazbeck 2022-2023

From Wikipedia, the number of line pairs/mm is 2^(g+(h-1)/6) where g is the
"group number" and h is the "element number".  Each group has 6 elements,
numbered from 1 to 6.  My ThorLabs test target goes down to group 7, meaning
the finest pitch is 2^(7+(6-1)/6)=2^(47/8)=

"""
from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.patches

import numpy as np
import cv2

import scipy.ndimage
import scipy.interpolate
import scipy.stats as st
import os.path
import os
import sys
from skimage.io import imread
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io
from skimage.transform import rotate
from scipy.signal import find_peaks
import math

LP = np.array(
    [0.250, 0.281, 0.315, 0.354, 0.397, 0.445,
     0.500, 0.561, 0.630, 0.707, 0.794, 0.891,
     1.00, 1.12, 1.26, 1.41, 1.59, 1.78,
     2.00, 2.24, 2.52, 2.83, 3.17, 3.56,
     4.00, 4.49, 5.04, 5.66, 6.35, 7.13,
     8.00, 8.98, 10.08, 11.31, 12.70, 14.25])


def template(n):
    """Generate a template image of three horizontal bars, n x n pixels

    NB the bars occupy the central square of the template, with a margin
    equal to one bar all around.  There are 3 bars and 2 spaces between,
    so bars are m=n/7 wide.

    returns: n x n numpy array, uint8
    """
    n = int(n)
    template = np.ones((n, n), dtype=np.uint8)
    template *= 255

    for i in range(3):
        template[n//7:-n//7,  (1 + 2*i) * n//7:(2 + 2*i) * n//7] = 0
    return template


def find_elements(image,
                  template_fn=template,
                  scale_increment=1.015,
                  n_scales=300,
                  return_all=True):
    """Use a multi-scale template match to find groups of 3 bars in the image.

    We return a list of tuples, (score, (x,y), size) for each match.

    image: a 2D uint8 numpy array in which to search
    template_fn: a function to generate a square uint8 array, which takes one
        argument n that specifies the side length of the square.
    scale_increment: the factor by which the template size is increased each
        iteration.  Should be a floating point number a little bigger than 1.0
    n_scales: the number of sizes searched.  The largest size is half the image
        size.

    """
    matches = []
    start = np.log(image.shape[0] / 2) / np.log(scale_increment) - n_scales
    print("Searching for targets", end='')
    for nf in np.logspace(start, start + n_scales, base=scale_increment):
        if nf < 24:  # There's no point bothering with tiny boxes...
            continue
        templ = template(nf)  # NB n is rounded down from nf
        # slides a window over the image and compares it to the template
        res = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        matches.append((max_val, max_loc, templ.shape[0]))
        print('.', end='')
    print("done")

    print("len matches: ", len(matches))

    # Take the matches at different scales and filter out the good ones
    scores = np.array([m[0] for m in matches])
    threshold_score = (scores.max() + scores.min()) / 2
    filtered_matches = [m for m in matches if m[0] > threshold_score]

    # Group overlapping matches together, and pick the best one
    def overlap1d(x1, n1, x2, n2):
        """Return the overlapping length of two 1d regions

        Draw four positions, indicating the edges of the regions (i.e. x1,
        c1+n1, x2, x2+n2).  The smallest distance between a starting edge (x1
        or x2) and a stopping edge (x+n) gives the overlap.  This will be
        one of the four values in the min().  The overlap can't be <0, so if
        the minimum is negative, return zero.
        """
        return max(min(x1 + n1 - x2, x2 + n2 - x1, n1, n2), 0)

    unique_matches = []
    while len(filtered_matches) > 0:
        current_group = []
        new_matches = [filtered_matches.pop()]
        while len(new_matches) > 0:
            current_group += new_matches
            new_matches = []
            for m1 in filtered_matches:
                for m2 in current_group:
                    s1, (x1, y1), n1 = m1
                    s2, (x2, y2), n2 = m2
                    overlap = (overlap1d(x1, n1, x2, n2) *
                               overlap1d(y1, n1, y2, n2))
                    if overlap > 0.5 * min(n1, n2) ** 2:
                        new_matches.append(m1)
                        filtered_matches.remove(m1)
                        break
        # Now we should have current_group full of overlapping matches.  Pick
        # the best one.
        best_score_index = np.argmax([m[0] for m in current_group])
        unique_matches.append(current_group[best_score_index])

    elements = unique_matches
    if return_all:
        return elements, matches
    else:
        return elements


def plot_matches(image, elements, elements_T=[], pdf=None):
    """Plot the image, then highlight the groups."""
    f, ax = plt.subplots(1, 1)
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    elif len(image.shape) == 3 and image.shape[2] == 3 or image.shape[2] == 4:
        ax.imshow(image)
    else:
        raise ValueError("The image must be 2D or 3D")
    for score, (x, y), n in elements:
        ax.add_patch(matplotlib.patches.Rectangle((x, y), n, n,
                                                  fc='none', ec='red'))
    for score, (y, x), n in elements_T:
        ax.add_patch(matplotlib.patches.Rectangle((x, y), n, n,
                                                  fc='none', ec='blue'))

    pdf.savefig(f)
    plt.close(f)


def approx_contrast(image, pdf=None, ax=None):
    """
    Compute the approximate contrast of an image.

    Args:
        image (ndarray): The input grayscale image.
        pdf (PdfPages or None): A PDF object to save the plot to (optional).
        ax (Axes or None): The Matplotlib axes to plot on (optional).

    Returns:
        float: The approximate contrast of the image.
    """
    # Get index of middle row
    mid_row = image.shape[0] // 2
    # Set the range of rows to use for averaging
    pixel_row_range = mid_row // 3

    # Compute the contrast of the center row
    contrast_center, first_plot, last_plot = approx_contrast_by_row(
        image, mid_row)

    # Plot the masked intensity values of the center row
    y = image[mid_row]
    x = np.arange(len(y))

    # Mask out the left side of the brightest region
    y[:first_plot+1] = 0
    # Mask out the right side of the brightest region
    y[last_plot:] = 0
    # Mask the values that were set to 0
    y_masked = np.ma.masked_equal(y, 0)
    # Mask the left side of the brightest region in the masked array
    y_masked[:first_plot+1] = np.ma.masked
    # Mask the right side of the brightest region in the masked array
    y_masked[last_plot:] = np.ma.masked
    # Plot the masked intensity values of the center row
    ax.plot(x, y_masked, color='black', linewidth=1,
            alpha=0.5, label='Center Row intensity')

    # Set the plot limits and labels
    ax.set_xlim([0, len(y)])
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')
    ax.set_ylim([0, 260])
    y_major = [0, 50, 100, 150, 200, 255]
    y_minor = np.arange(0, 260, 5)
    ax.set_yticks(y_major, major=True)
    ax.set_yticks(y_minor, minor=True)
    ax.set_xticks(np.arange(0, len(x), len(x)//10))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.xaxis.grid(which='both', alpha=0.2)
    ax.grid(which='both', alpha=0.2)

    # Compute the average intensity values over the range of rows
    contrast_mean = [contrast_center]
    row_range = range(mid_row - pixel_row_range,
                      mid_row + pixel_row_range + 1, 1)
    image_rows = [image[row] for row in row_range]
    y_stacked = np.vstack(image_rows)
    x = np.arange(y_stacked.shape[1])

    # Mask out the left side of the brightest region for each row
    y_stacked[:, :first_plot+1] = 0
    # Mask out the right side of the brightest region for each row
    y_stacked[:, last_plot:] = 0

    """
    Note: We still take the same range of pixels as the centered column, 
    but it is important to note that the range could change for each row 
    since the function approx_contrast_by_row() returns the first and last pixel 
    index of the brightest region. In the end, we would need to truncate the array 
    to the same size for each row to do the average. This way the result remains the same.
    """

    # Mask the brightest first and last regions for each row in the masked array
    y_masked = np.ma.masked_equal(y_stacked, 0)
    y_masked[:, :first_plot+1] = np.ma.masked
    y_masked[:, last_plot:] = np.ma.masked
    # Compute the mean intensity values over the range of rows
    y = np.mean(y_masked, axis=0)
    # Plot the average intensity values over the range of rows
    ax.plot(x, y, color='red', linewidth=0.7, alpha=0.4,
            label='Average intensity over rows')
    ax.legend(loc='lower right', fontsize=4)

    # Compute the contrast of rows above and below the center row
    for i in range(1, pixel_row_range + 1):
        # Compute the contrast of the row below the center row
        contrast, _, _ = approx_contrast_by_row(image, mid_row + i)
        # Append the contrast value to the list of mean contrast values
        contrast_mean.append(contrast)
        # Compute the contrast of the row above the center row
        contrast, _, _ = approx_contrast_by_row(image, mid_row - i)
        # Append the contrast value to the list of mean contrast values
        contrast_mean.append(contrast)

    # Compute the mean contrast over all rows
    num_rows = len(contrast_mean)
    contrast_mean = np.mean(contrast_mean)

    # Add the contrast value to the PDF object (if provided)
    if pdf is not None:
        pdf.attach_note(f'Contrast ({num_rows} rows): {contrast_mean}')

    # Save the plot to the PDF object (if provided)
    if pdf is not None:
        pdf.savefig()

    # Close the plot
    plt.close()

    # Return the mean contrast over all rows
    return contrast_mean


def approx_contrast_by_row(image, row):
    """
    Calculates the approximate contrast of a row in an image.

    Args:
        image (numpy.ndarray): The image to calculate the contrast of.
        row (int): The row number to calculate the contrast of.

    Returns:
        tuple: A tuple containing the contrast value and the first and last plot points
        to consider for the plotting of the intensity.

    NB: The first and last plot points are the first and last points of the darkest areas.
    NB 2: We are also considering this range of points to calculate the contrast. The
    first and last brightest regions are not considered.
    """
    # Get the pixel values for the specified row
    y = np.array(image[row])

    # Calculate the middle point of the row
    middle_pt = np.mean(y)

    # Identify the peak intensity values above and below the middle point
    max_peaks = y[y > middle_pt]
    max_peaks_arg = np.argwhere(y > middle_pt)[:, 0]
    min_peaks = y[y < middle_pt]
    min_peaks_arg = np.argwhere(y < middle_pt)[:, 0]

    # Check if the first peak is a maximum or minimum, and adjust accordingly
    first_plot = 0
    if max_peaks_arg[0] < min_peaks_arg[0]:
        # Find the first maximum peak after the first minimum peak
        first_max = np.where(max_peaks_arg > min_peaks_arg[0])[0][0]
        max_peaks = max_peaks[first_max:]
        first_plot = min_peaks_arg[0]

    # Check if the last peak is a maximum or minimum, and adjust accordingly
    last_plot = len(y)
    if max_peaks_arg[-1] > min_peaks_arg[-1]:
        # Find the last maximum peak before the last minimum peak
        last_max = np.where(max_peaks_arg < min_peaks_arg[-1])[0][-1]
        max_peaks = max_peaks[:last_max+1]
        last_plot = min_peaks_arg[-1]

    # Find the mode (most common value) of the peak intensities
    max_peak_intensity, _ = st.mode(max_peaks, keepdims=False)
    min_peak_intensity, _ = st.mode(min_peaks, keepdims=False)

    # Convert the peak intensities to 64-bit integers and calculate the contrast to avoid integer overflow
    I_max = np.int64(max_peak_intensity)
    I_min = np.int64(min_peak_intensity)
    contrast = (I_max - I_min) / (I_max + I_min)

    # Return the contrast value and the first and last plot points
    return contrast, first_plot, last_plot


def compute_mtf_curve(image, elements, pdf=None, horizontal=0):
    """
    Computes the modulation transfer function (MTF) curve for an image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        elements (list): A list of tuples representing the subregions of the image to analyze.
            Each tuple should contain a contrast score, a (x, y) position, and a size n.
        pdf (str): The filename of a PDF file to save the MTF curves to (optional).

    Returns:
        list: A list of the contrast scores for each element in the input list.
    """
    # Initialize variables
    _, axes = None, [[None] * len(elements)] * 2
    contrasts = []
    LP_count = 0  # Set a LP counter to be able to title the plots

    # Loop through each element in the input list
    for (score, (x, y), n), ax0, ax1 in zip(elements, axes[0], axes[1]):
        # Create a new figure with three subplots
        f, (ax1, ax_empty, ax2) = plt.subplots(
            1, 3, gridspec_kw={'width_ratios': [1, 0.2, 1]})
        # Remove ticks and labels from the empty subplot
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        ax_empty.set_frame_on(False)
        # Extract the gray-scale region of interest from the input image
        gray_roi = image[y:y + n, x:x + n]
        # Display the gray-scale image in the first subplot
        ax1.imshow(gray_roi, vmin=0, vmax=255)
        # Set the major x-ticks to 10% of the width of the region of interest
        ax1.set_xticks(np.arange(0, n, n//10))
        # Reduce the font size of the x-tick labels
        ax1.xaxis.set_tick_params(labelsize=5, rotation=90)
        # Add a grid to the x-axis
        ax1.xaxis.grid(which='both', alpha=0.2)
        # Calculate the row range used to compute the contrast
        mid_row = gray_roi.shape[0]//2
        pixel_row_range = mid_row // 3
        y_major = [mid_row - pixel_row_range, mid_row + pixel_row_range]
        # Set the major y-ticks to the row range used to compute contrast
        ax1.set_yticks(y_major, major=True)
        # Make the y-tick labels invisible
        ax1.yaxis.set_tick_params(labelsize=0, labelcolor='white')
        # Add a grid to the y-axis
        ax1.grid(which='major', axis='y', alpha=0.1, color='red')
        # Add a label for the y-axis
        ax1.set_ylabel('Range of rows used to calculate contrast', fontsize=5)
        # Add a label for the x-axis
        ax1.set_xlabel('Pixel')
        # Add a text indicating how many line pairs per mm the user is looking at and indicate if the bars are vertical or not
        if (horizontal == 0):
            ax1.annotate("no. line pairs per mm for vertical pairs: "+str(LP[LP_count]), xy=(0.5, 1.05), xytext=(0, 10),
                         xycoords='axes fraction', textcoords='offset points',
                         ha='center', va='bottom', fontsize=5)
        else:
            ax1.annotate("no. line pairs per mm for horizontal pairs: "+str(LP[LP_count]), xy=(0.5, 1.05), xytext=(0, 10),
                         xycoords='axes fraction', textcoords='offset points',
                         ha='center', va='bottom', fontsize=5)
        # Add a colorbar to the first subplot
        # This helps to compare the contrast of the different groups and see where the lighting affects results
        f.colorbar(ax1.get_images()[0], ax=ax1, pad=0.05)
        # Compute the contrast of the gray-scale region of interest
        contrasts.append(approx_contrast(gray_roi, pdf, ax2))
        # Increment the line pair counter
        LP_count += 1

    # Set the x-tick locations to the input line pairs
    xticks = LP[:len(contrasts)]
    # Plot the MTF curve
    plt.plot(xticks, contrasts, 'o-', markersize=2, linewidth=0.7)
    # Add a label for the x-axis
    plt.xlabel("no. line pairs per mm")
    # Add a label for the y-axis
    plt.ylabel("contrast")
    # Add a title for the plot
    if (horizontal == 0):
        plt.title("MTF curve for vertical pairs")
    else:
        plt.title("MTF curve for horizontal pairs")
    # Add grid lines
    minor_yticks = np.arange(0, 1.05, 0.01)
    major_yticks = np.arange(0, 1.05, 0.05)
    plt.gca().set_yticks(major_yticks, major=True)
    plt.gca().set_yticks(minor_yticks, minor=True)
    plt.gca().set_xticks(xticks)

    # Hide every 3rd xtick label for the first 1/3 of the xticks
    # Set the visible ticks as major ticks and the hidden ones as minor ticks
    minor_ticks_location = []
    major_ticks_location = []

    for i in range(0, len(plt.gca().get_xticklabels())):
        if i < len(xticks)//3:
            if i % 3 != 0:
                minor_ticks_location.append(plt.gca().get_xticks()[i])
            else:
                major_ticks_location.append(plt.gca().get_xticks()[i])
        else:
            major_ticks_location.append(plt.gca().get_xticks()[i])
    plt.gca().set_xticks(minor_ticks_location, minor=True)
    plt.gca().set_xticks(major_ticks_location, major=True, visible=True)
    plt.tick_params(axis='x', which='major', length=5, color='blue',
                    labelrotation=90, labelsize=4.5, labelcolor='blue')
    plt.tick_params(axis='x', which='minor', labelsize=0)
    plt.tick_params(axis='y', labelsize=6.5)
    plt.grid(which='minor', alpha=0.15)
    plt.grid(which='major', alpha=0.2, color='blue')

    # If a PDF output file is provided, save the figure to the PDF file
    if pdf:
        pdf.savefig()
    # Close the figure
    plt.close()


def analyse_image(gray_image, pdf=None):
    """
    Find USAF groups in the image and plot their contrast as a MTF curve.

    This is the top-level function that you should call to analyse an image.
    The image should be a 2D numpy array.  If the optional "pdf" argument is
    supplied, several graphs will be plotted into the given PdfPages object.

    """
    elementsx, matchesx = find_elements(gray_image, return_all=True)
    elementsy, matchesy = find_elements(gray_image.T, return_all=True)

    plot_matches(gray_image, elementsx, elementsy, pdf)
    compute_mtf_curve(gray_image, elementsx, pdf)
    compute_mtf_curve(gray_image.T, elementsy, pdf, horizontal=1)


def analyse_file(filename, generate_pdf=True):
    """Analyse the image file specified by the given filename"""
    delim = None
    if '/' in filename:
        delim = '/'
    elif '\\' in filename:
        delim = '\\'

    if delim is not None:
        savename = filename.split(delim)
        savename[len(savename) - 1] = 'rotated_' + savename[len(savename) - 1]
        new_fn = delim.join(savename)
    else:
        new_fn = 'rotated_' + filename

    gray_image = imread(filename, as_gray=1)

#     # Save the rotated image
#     io.imsave(new_fn, rotated_img)

    with PdfPages(new_fn + "_analysis.pdf") as pdf:
        analyse_image(gray_image, pdf)


def analyse_folders(datasets):
    """Analyse a folder hierarchy containing a number of calibration images.

    Given a folder that contains a number of other folders (one per microscope usually),
    find all the USAF images (<datasets>/*/usaf_*.jpg) and analyse them.  It also generates
    a summary file in CSV format, and a PDF with all the images and the detected elements.
    """
    files = []
    # if d.startswith('6led')]:
    for dir in [os.path.join(datasets, d) for d in os.listdir(datasets)]:
        files += [os.path.join(dir, f) for f in os.listdir(dir)
                  if f.startswith("usaf_") and f.endswith(".jpg")]

    with PdfPages("usaf_calibration.pdf") as pdf:
        for filename in files:
            print("\nAnalysing file {}".format(filename))
            analyse_file(filename)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: {} <file_or_folder> [<file2> ...]".format(sys.argv[0]))
        print(
            "If a file is specified, we produce <file>_analysis.pdf and <file>_analysis.txt")
        print("If a folder is specified, we produce usaf_calibration.pdf and usaf_calibration_summary.csv")
        print("as well as the single-file analysis for <folder>/*/usaf_*.jpg")
        print("Multiple files may be specified, using wildcards if your OS supports it - e.g. myfolder/calib*.jpg")
        exit(-1)
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        analyse_folders(sys.argv[1])
    else:
        for filename in sys.argv[1:]:
            analyse_file(filename)
