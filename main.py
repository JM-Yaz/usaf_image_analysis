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
def find_image_orientation(gray_image, fuzziness=5):
    """Determine the angle we need to rotate a USAF target image by.

    This is a two-step "straightener", first looking at the gradient directions
    in the image, then fine-tuning it to get a more accurate answer.  Rotating
    the image by this amount should cause the bars to align with the x/y axes.
    It may be beneficial to do this iteratively, i.e. run it, then rotate, then
    run it again, then tweak - this should help avoid any issues with linearity
    in the calculation of gradient angles.

    fuzziness sets the size of the Gaussian kernel used for gradient estimation

    returns: the angle of the bars, as a floating-point number.
    """
    image = gray_image / gray_image.max()
    # Calculate local gradient directions (to find angle)
    Gx = scipy.ndimage.gaussian_filter1d(image, axis=0, order=1, sigma=fuzziness)
    Gy = scipy.ndimage.gaussian_filter1d(image, axis=1, order=1, sigma=fuzziness)
    angles = np.arctan2(Gx, Gy)
    weights = Gx ** 2 + Gy ** 2

    # First, find the histogram and pick the maximum
    h, e = np.histogram(angles % (np.pi), bins=100, weights=weights)
    # plt.plot((e[1:] + e[:-1])/2, h, 'o-')
    rough_angle = ((e[1:] + e[:-1]) / 2)[np.argmax(h)]  # pick the peak
    bin_width = np.mean(np.diff(e))

    # Recalculate the histogram about the peak (avoids wrapping issues)
    h, e = np.histogram(((angles - rough_angle + np.pi / 4) % (np.pi / 2)) - np.pi / 4,
                        bins=50, range=(-0.1, 0.1), weights=weights)
    h *= bin_width / np.mean(np.diff(e))
    h /= 2
    # plt.plot((e[1:] + e[:-1])/2+rough_angle, h)

    # Use spline interpolation to fit the peak and find the rotation angle
    spline = scipy.interpolate.UnivariateSpline(e[1:-1], np.diff(h))
    root = spline.roots()[np.argmin(spline.roots() ** 2)]  # pick root nearest zero
    # plt.plot(e+rough_angle, spline(e))

    fine_angle = root + rough_angle

    # TODO: do we want to rotate the image here and repeat the above analysis?

    return fine_angle


################### Find the bars ############################

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
        template[n // 7:-n // 7, (1 + 2 * i) * n // 7:(2 + 2 * i) * n // 7] = 0
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
        if nf < 20:  # There's no point bothering with tiny boxes...
            continue
        templ = template(nf)  # NB n is rounded down from nf
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
    # get index of middle row
    mid_row = image.shape[0] // 2
    # width is 3
    map_center, mip_center, min_peaks, max_peaks = approx_contrast_by_row(image, mid_row)
    y = image[mid_row]
    x = [x for x in range(len(y))]
    ax.plot(max_peaks, y[max_peaks], "x", color='red')
    ax.plot(min_peaks, y[min_peaks], "x", color='red')
    ax.plot(x, y)

    map_bot, mip_bot, _, _ = approx_contrast_by_row(image, mid_row + 1)
    map_top, mip_top, _, _ = approx_contrast_by_row(image, mid_row - 1)
    map_mean = (map_center + map_bot + map_top -
                mip_center - mip_bot - mip_top) / (map_center +
                                                   map_bot + map_top +
                                                   mip_center + mip_bot + mip_top)
    pdf.attach_note(f'Contrast (3 rows): {map_mean}')
    pdf.savefig()
    plt.close()
    return map_mean


def approx_contrast_by_row(image, row):
    y = image[row]

    # adapt if no. of local minima/maxima is unequal 3
    min_dist = len(y) // 4
    max_peaks, _ = find_peaks(y, height=0, distance=min_dist)
    # discard 4th local maxima
    if len(max_peaks) == 4:
        max_peaks = max_peaks[:-1]

    min_peaks, _ = find_peaks(-y, height=0, distance=min_dist)
    if len(min_peaks) == 4:
        min_peaks = min_peaks[:-1]

    sum_map = sum([y[x] for x in max_peaks])
    sum_mip = sum([y[x] for x in min_peaks])
    return sum_map, sum_mip, max_peaks, min_peaks


def compute_mtf_curve(image, elements, pdf=None):
    _, axes = None, [[None] * len(elements)] * 2
    contrasts = []

    for (score, (x, y), n), ax0, ax1 in zip(elements, axes[0], axes[1]):
        f, (ax1, ax2) = plt.subplots(1, 2)
        gray_roi = image[y:y + n, x:x + n]
        ax1.imshow(gray_roi)
        # ax1.plot()
        contrasts.append(approx_contrast(gray_roi, pdf, ax2))

    plt.plot(LP[:len(contrasts)], contrasts, 'o-')
    plt.xlabel("no. line pairs per mm")
    plt.ylabel("contrast (3 rows)")
    pdf.savefig()
    plt.close()


def analyse_image(gray_image, pdf=None):
    """Find USAF groups in the image and plot their contrast as a MTF curve.

    This is the top-level function that you should call to analyse an image.
    The image should be a 2D numpy array.  If the optional "pdf" argument is
    supplied, several graphs will be plotted into the given PdfPages object.

    returns

    fig is a matplotlib figure object plotting to the image with each
    element highlighted in a box."""
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

    gray_image = imread('unrotated.bmp', as_gray=1).astype(np.uint8)
    ref_image = imread('lab.bmp', as_gray=1).astype(np.uint8)

    # Detect and match features between the images using the ORB algorithm
    orb = cv2.ORB_create()
    keypoints_1, descriptors_1 = orb.detectAndCompute(gray_image, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(ref_image, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_1, descriptors_2)

    # Sort the matches by distance and take the 10 first values
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:10]

    # Extract the coordinates of the matched keypoints
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use the homography matrix to calculate the rotation angle
    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    rotation_matrix, _ = cv2.Rodrigues(h[:, :3])
    angle = np.rad2deg(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

    # Rotate the target image using the rotation angle
    rows, cols = ref_image.shape
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(ref_image, M, (cols, rows))

    io.imsave(new_fn, rotated_img)
    gray_image = imread(new_fn).astype(np.uint8)

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
