## Overview
The following document explains the code that analyzes a USAF test target image to determine its dimensions. The USAF test target is a common resolution test pattern used to determine the resolving power of optical systems such as microscopes and cameras.
The script was initially developed by Richard Bowman in 2017 and later modified by Louis Ngo and Jean-Marie Yazbeck in 2022-2023.

### Dependencies

This script requires the following Python libraries:

- matplotlib
- numpy
- opencv-python (cv2)
- scipy
- scikit-image (skimage)

You can install the required libraries using the following command in the terminal:
`pip install -r requirements.txt`

## Usage

To use the script, run the following command in the terminal:
`python main.py path/to/your/image/file.bmp`

The script will generate a PDF file containing the analysis results. You can disable the PDF generation by setting the `generate_pdf` parameter to False in the code.

## Code Description

### Constants

A numpy array, `LP`, is defined with the number of line pairs per millimeter for each group and element of the USAF test target. This array needs to be edited for different charts.

The provided code can be used as a starting point for further analysis and enhancements. For example, you can extend the code to support different test target patterns or incorporate additional image processing techniques to improve the accuracy of the MTF curve calculation.

### Functions

The following is a description of each function in the code:

`template(n)`:
This function generates a template of size n x n with alternating black and white bars. The input, n, must be an integer.

`find_elements(image, template_fn=template, scale_increment=1.015, n_scales=300, return_all=True)`:
Detects the groups and elements in the input image using the template_fn with the specified scale_increment and n_scales. Returns the detected elements and matches, or just the elements if return_all is set to False.

- image: The grayscale input image.
- template_fn: A function that generates the template.
- scale_increment: The scale increment between templates.
- n_scales: The number of scales to search.
- return_all: If true, returns all elements and matches; otherwise, returns only unique elements.

`plot_matches(image, elements, elements_T=[], pdf=None)`:
Plots the input image with the found elements highlighted. If the elements_T argument is provided, it also plots the transposed image with the found elements highlighted.

- image: The input image.
- elements: The found elements.
- elements_T: The elements of the transposed image. (For the horizontal bars in this case)
- pdf: The PDF file to save the plot to.

`approx_contrast(image, pdf=None, ax=None)`:
This function computes an approximate contrast value for the image. It takes the following inputs:

- image: The input image.
- pdf: The PDF file to save the plot to.
- ax: The axes to plot on.

`approx_contrast_by_row(image, row)`:
This function computes an approximate contrast value for a specific row of the image. It takes the following inputs:

- image: The input image.
- row: The row index.

`compute_mtf_curve(image, elements, pdf=None, horizontal=0)`:
This function computes the MTF curve for the image using the found elements. It takes the following inputs:

- image: The input image.
- elements: The found elements.
- pdf: The PDF file to save the plot to.
- horizontal: This is only a flag to mention in the PDF that we are computing the MTF curve for horizontal pairs. That's when it is set to 1.

`add_execution_info(pdf: PdfPages)`:
This function adds a page to the PDF with information about the execution. It takes the following inputs:
- pdf (PdfPages): A PdfPages object representing the PDF file.

`analyse_image(gray_image, pdf=None)`:
This function performs the analysis on the input grayscale image. It takes the following inputs:

- gray_image: The grayscale input image.
- pdf: The PDF file to save the analysis results to.

`analyse_file(filename, generate_pdf=True)`:
This function analyzes the image file specified by the given filename. It takes the following inputs:

- filename: The filename of the image file to analyze.
- generate_pdf: If true, generates a PDF file with the analysis results.