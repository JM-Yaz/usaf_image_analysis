# -*- coding: utf-8 -*-
"""
Analyse a USAF test target image, to determine the image's dimensions.

See: https://en.wikipedia.org/wiki/1951_USAF_resolution_test_chart

(c) Richard Bowman 2017, released under GNU GPL
    modified by Louis Ngo 2022


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
    [0.250, 0.281, 0.315, 0.354, 0.397, 0.445,\
     0.500, 0.561, 0.630, 0.707, 0.794, 0.891,\
     1.00, 1.12, 1.26, 1.41, 1.59, 1.78,\
     2.00, 2.24, 2.52, 2.83, 3.17, 3.56,\
     4.00, 4.49, 5.04, 5.66, 6.35, 7.13,\
     8.00, 8.98, 10.08, 11.31, 12.70, 14.25])


#################### Rotate the image so the bars are X/Y aligned #############
# def find_image_orientation(gray_image, fuzziness=5):
#     """Determine the angle we need to rotate a USAF target image by.

#     This is a two-step "straightener", first looking at the gradient directions
#     in the image, then fine-tuning it to get a more accurate answer.  Rotating
#     the image by this amount should cause the bars to align with the x/y axes.
#     It may be beneficial to do this iteratively, i.e. run it, then rotate, then
#     run it again, then tweak - this should help avoid any issues with linearity
#     in the calculation of gradient angles.

#     fuzziness sets the size of the Gaussian kernel used for gradient estimation

#     returns: the angle of the bars, as a floating-point number.
#     """
#     image = gray_image / gray_image.max()
#     # Calculate local gradient directions (to find angle)
#     Gx = scipy.ndimage.gaussian_filter1d(image, axis=0, order=1, sigma=fuzziness)
#     Gy = scipy.ndimage.gaussian_filter1d(image, axis=1, order=1, sigma=fuzziness)
#     angles = np.arctan2(Gx, Gy)
#     weights = Gx ** 2 + Gy ** 2

#     # First, find the histogram and pick the maximum
#     h, e = np.histogram(angles % (np.pi), bins=100, weights=weights)
#     # plt.plot((e[1:] + e[:-1])/2, h, 'o-')
#     rough_angle = ((e[1:] + e[:-1]) / 2)[np.argmax(h)]  # pick the peak
#     bin_width = np.mean(np.diff(e))

#     # Recalculate the histogram about the peak (avoids wrapping issues)
#     h, e = np.histogram(((angles - rough_angle + np.pi / 4) % (np.pi / 2)) - np.pi / 4,
#                         bins=50, range=(-0.1, 0.1), weights=weights)
#     h *= bin_width / np.mean(np.diff(e))
#     h /= 2
#     # plt.plot((e[1:] + e[:-1])/2+rough_angle, h)

#     # Use spline interpolation to fit the peak and find the rotation angle
#     spline = scipy.interpolate.UnivariateSpline(e[1:-1], np.diff(h))
#     root = spline.roots()[np.argmin(spline.roots() ** 2)]  # pick root nearest zero
#     # plt.plot(e+rough_angle, spline(e))

#     fine_angle = root + rough_angle

#     # TODO: do we want to rotate the image here and repeat the above analysis?

#     return fine_angle

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
        res = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED) #slides a window over the image and compares it to the template
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
    # get index of middle row
    mid_row = image.shape[0] // 2
    # print(image.shape[0])
    # print(image.shape)
    # width is 3
    contrast_center, first_plot, last_plot = approx_contrast_by_row(image, mid_row)
    y = image[mid_row]
    x = [x for x in range(len(y))]
    y[:first_plot+1] = 0
    y[last_plot:] = 0
    # ax.plot(max_peaks, y[max_peaks], "x", color='red')
    # ax.plot(min_peaks, y[min_peaks], "x", color='green')
    y_masked = np.ma.masked_equal(y, 0)    
    y_masked[:first_plot+1] = np.ma.masked
    y_masked[last_plot:] = np.ma.masked
    ax.plot(x, y_masked)
    ax.set_xlim([0, len(y)])
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')
    ax.set_ylim([0, 260])
    ax.set_yticks([0, 50, 100, 150, 200, 255])
    ax.set_yticklabels(['0', '50', '100', '150', '200', '255'])
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1*len(x))) # minor ticks every 0.1*pixel_amount units
    # Show grid lines
    ax.grid(which='both', alpha=0.2)
    contrast_mean = [contrast_center]

    for i in range(1, mid_row//4):
        contrast, _ , _ = approx_contrast_by_row(image, mid_row + i)
        contrast_mean.append(contrast)
        contrast, _ , _ = approx_contrast_by_row(image, mid_row - i)
        contrast_mean.append(contrast)
    num_rows = len(contrast_mean)
    contrast_mean = np.mean(contrast_mean)

    pdf.attach_note(f'Contrast ({num_rows} rows): {contrast_mean}')
    pdf.savefig()
    plt.close()
    return contrast_mean

def approx_contrast_by_row(image, row):
    y = np.array(image[row])
    middle_pt = np.mean(y)

    max_peaks = y[y > middle_pt]
    max_peaks_arg = np.argwhere(y > middle_pt)[:, 0]
    min_peaks = y[y < middle_pt]
    min_peaks_arg = np.argwhere(y < middle_pt)[:, 0]
    first_plot = 0
    last_plot = len(y)
    if max_peaks_arg[0] < min_peaks_arg[0]:
        first_max = np.where(max_peaks_arg > min_peaks_arg[0])[0][0]
        max_peaks = max_peaks[first_max:]
        first_plot = min_peaks_arg[0]
    if max_peaks_arg[-1] > min_peaks_arg[-1]:
        last_max = np.where(max_peaks_arg < min_peaks_arg[-1])[0][-1]
        max_peaks = max_peaks[:last_max+1]
        last_plot = min_peaks_arg[-1]

    max_peaks, _ = st.mode(max_peaks, keepdims=False)
    min_peaks, _ = st.mode(min_peaks, keepdims=False)

    
    I_max = np.int64(max_peaks)
    I_min = np.int64(min_peaks)
    contrast = (I_max - I_min) / (I_max + I_min)
    return contrast, first_plot, last_plot

def compute_mtf_curve(image, elements, pdf=None):
    _, axes = None, [[None] * len(elements)] * 2
    contrasts = []

    for (score, (x, y), n), ax0, ax1 in zip(elements, axes[0], axes[1]):
        f, (ax1, ax_empty, ax2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 0.175, 1]})
        # Remove ticks and labels from the empty subplot
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        ax_empty.set_frame_on(False)
        gray_roi = image[y:y + n, x:x + n]
        ax1.imshow(gray_roi, vmin=0, vmax=255)
        ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1*n)) # minor ticks every 5 units
        ax1.xaxis.grid(which='minor', alpha=0.2)
        # Show the colorbar of ax1, values are in the range of 0 to 255 and no longer normalized
        # This helps to compare the contrast of the different groups and see where the lighting affects results
        f.colorbar(ax1.get_images()[0], ax=ax1, pad=0.05)
        contrasts.append(approx_contrast(gray_roi, pdf, ax2))

    plt.plot(LP[:len(contrasts)], contrasts, 'o-')
    plt.xlabel("no. line pairs per mm")
    # scale the y axis from 0 to 1
    plt.ylim((0, 1.0))
    plt.ylabel("contrast")
    plt.title("MTF curve")
    # Show girid lines
    plt.grid(which='both')
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    pdf.savefig()
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
    compute_mtf_curve(gray_image.T, elementsy, pdf)


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
    # ref_image = imread('rotated_unrotated.bmp', as_gray=1).astype(np.uint8)

#     # Use the ORB algorithm to detect keypoints and compute descriptors
#     orb = cv2.ORB_create()
#     real_kp, real_desc = orb.detectAndCompute(gray_image, None)
#     ideal_kp, ideal_desc = orb.detectAndCompute(ref_image, None)

#    # Match the keypoints using the Brute Force matcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(ideal_desc, real_desc)

#     # Find rotation and translation
#     src_pts = np.float32([ideal_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([real_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#     # Use the homography matrix to rotate the image
#     rows, cols = gray_image.shape
#     rotated_img = cv2.warpPerspective(gray_image, M, (cols, rows))


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
    for dir in [os.path.join(datasets, d) for d in os.listdir(datasets)]:  # if d.startswith('6led')]:
        files += [os.path.join(dir, f) for f in os.listdir(dir) if f.startswith("usaf_") and f.endswith(".jpg")]

    with PdfPages("usaf_calibration.pdf") as pdf:
        for filename in files:
            print("\nAnalysing file {}".format(filename))
            analyse_file(filename)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: {} <file_or_folder> [<file2> ...]".format(sys.argv[0]))
        print("If a file is specified, we produce <file>_analysis.pdf and <file>_analysis.txt")
        print("If a folder is specified, we produce usaf_calibration.pdf and usaf_calibration_summary.csv")
        print("as well as the single-file analysis for <folder>/*/usaf_*.jpg")
        print("Multiple files may be specified, using wildcards if your OS supports it - e.g. myfolder/calib*.jpg")
        exit(-1)
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        analyse_folders(sys.argv[1])
    else:
        for filename in sys.argv[1:]:
            analyse_file(filename)
