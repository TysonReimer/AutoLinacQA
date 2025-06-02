"""
Tyson Reimer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import zipfile
import tempfile
import io
from matplotlib.backends.backend_pdf import PdfPages


###############################################################################


def _find_center_of_img(img):
    """Find center of the jaw-defined field

    Parameters
    ----------
    img : array_like
        2D array of pixel values

    Returns
    -------
    c_pix :
        Tuple of indices of center pixel
    """

    # Normalize image
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Find the center pixel
    c_pix = np.argwhere(
        img_normalized >= 0.5 * np.max(img_normalized)
    ).mean(axis=0)

    return c_pix

def _find_CAX(ds_1, ds_2):
    """Find the MV CAX using two dicom files of jaw-defined field images

    Parameters
    ----------
    ds_1 :
        dcmread() of dicom file of jaw-defined field
    ds_2 :
        dcmread() of dicom file of jaw-defined field, with collimator
        rotated 180deg with respect to the position for ds_1
    """

    img1 = ds_1.pixel_array
    img2 = ds_2.pixel_array

    c_pix1 = _find_center_of_img(img=img1)
    c_pix2 = _find_center_of_img(img=img2)

    # MV CAX defined as average of the two centers
    # (geometrically corresponds to the CAX, assuming negligible
    # size of collimator isocenter)
    beam_CAX = 0.5 * (c_pix1 + c_pix2)

    # Swap coordinates to maintain consistency of row/col and x/y
    beam_CAX = beam_CAX[::-1]

    return beam_CAX



def _idx_to_mm(ds, do_x, image, edge_pixel_position, beam_CAX):
    """Convert EPID image array index to absolute position in [mm]

    Parameters
    ----------
    ds :
        Dicom file
    do_x : bool
        If True, do conversion for x-axis. Else do y-axis
    image : array_like
        2D EPID image arr
    edge_pixel_position :
        The interpolated index of the edge pixel
    beam_CAX : tuple
        Beam CAX (x,y) indices
    """

    origin = 0
    if not (beam_CAX is None):
        center_pixel = beam_CAX
    else:
        # Get origin of EPID from dicom
        origin = ds.XRayImageReceptorTranslation[:-1]

        # Define the image center as (0 mm, 0 mm); the center in pixel coordinates is:
        center_pixel = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])

    # Get pixel spacing from DICOM (assumed format: [row_spacing, col_spacing])
    pixel_spacing = np.array(ds.ImagePlanePixelSpacing, dtype=float)

    if do_x:
        # For an x-edge, convert the x coordinate relative to center:
        physical_edge = ((edge_pixel_position - center_pixel[0])
                         * pixel_spacing[1])
        if beam_CAX is None:
            physical_edge += origin[0]

    else:  # If doing y
        # For a y-edge, convert the y coordinate relative to center:
        physical_edge = ((edge_pixel_position - center_pixel[1])
                         * pixel_spacing[0])
        if beam_CAX is None:
            physical_edge += origin[1]

    # Correct for EPID SID (should be very close to 1)
    physical_edge = physical_edge * (ds.RTImageSID / 1000)

    return physical_edge


def _find_edge_in_slice(image, slice_index, from_negative, along_x, img_max):
    """Find the jaw edge, defined as 50% of max intensity, in img slice

    Parameters
    ----------
    image : array_like
        2D EPID image arr
    slice_index : int
        Index of the slice to consider (row or col)
    from_negative : bool
        If True, starts from the negative axis side. Else from the
        positive sde.
    along_x : bool
        If True, looks for an X-jaw position. Else looks for Y-jaw.
    img_max : float
        The maximum intensity in the image

    Returns
    -------
    edge_pos : float
        The edge position interpolated index
    """

    if along_x:  # If looking for X-jaws
        slice = image[slice_index, :]
    else:  # If looking for Y-jaws
        slice = image[:, slice_index]

    # Check if this slice contains the jaw-defined field, or if it's
    # outside the region of interest
    if not (np.max(slice) >= 0.7 * img_max):
        return

    # Get the threshold used to define an 'edge' for this slice
    threshold = (np.min(slice) + np.max(slice)) / 2.0

    edge_pos = None  # Start with no edge position

    # Find first index where intensity reaches/exceeds the threshold
    if from_negative:
        indices = np.where(slice >= threshold)[0]
    else:  # If going from the positive side of the axis
        indices = np.where(np.flip(slice) >= threshold)[0]

    if indices.size != 0:  # If some indices were found

        idx = indices[0]

        # If the index is not at one of the extremes
        if (idx > 5) and (idx < len(slice) - 5):

            if from_negative:
                # Linear interpolation between pixel idx-1 and idx:
                frac = (threshold - slice[idx - 1]) / (
                        slice[idx] - slice[idx - 1])
                edge_pos = (idx - 1) + frac

            else:  # If from positive axis

                flip_slice = np.flip(slice)

                # Linear interpolation between pixel idx-1 and idx:
                frac = (threshold - flip_slice[idx - 1]) / (
                        flip_slice[idx] - flip_slice[idx - 1])
                edge_pos = (idx - 1) + frac

                # Subtract 1 b/c 0-based indexing
                edge_pos = len(slice) - edge_pos - 1

    return edge_pos


def _find_edge(image, ds, img_max, along_x=True, from_negative=True,
               beam_CAX=None):
    """Find the edge of the jaw-defined field in an EPID image

    Parameters
    ----------
    image : array_like
        2D EPID image
    ds :
        Dicom file of the image
    along_x : bool
        If True, will find the edge of one of the X-jaws. Else will
        find the Y-jaw edge.
    from_negative : bool
        If True, will find the edge of the jaw on the negative axis side

    Returns
    -------
    edge : int
        The index of the edge for the image array
    edge_mm : float
        The position in [mm] of the edge along the axis perpendicular
        to the edge
    """

    edges = []  # Init list for storing edges

    if along_x:  # If doing X-jaw
        n_slices = image.shape[0] - 1  # Get number of slices to look at
    else:  # If along_y
        n_slices = image.shape[1] - 1

    # Define search region to avoid primary collimator shadow
    search_ini_idx = int(0.2 * n_slices)
    search_fin_idx = int(0.8 * n_slices)

    # For each slice (excluding the edge cases)
    for slice_index in range(search_ini_idx, search_fin_idx):

        # Get the position of the edge in this slice
        edge_pos = _find_edge_in_slice(
            image,
            slice_index=slice_index,
            from_negative=from_negative,
            along_x=along_x,
            img_max=img_max,
        )

        if not (edge_pos is None):  # If an edge position was found
            edges.append(edge_pos)  # Store it for later

    edge = np.median(edges)  # Define the jaw edge by the median

    # Convert to absolute position in [mm]
    edge_mm = _idx_to_mm(
        ds=ds,
        do_x=along_x,
        image=image,
        edge_pixel_position=edge,
        beam_CAX=beam_CAX,
    )

    return edge, edge_mm


def _find_all_edges(image, ds, beam_CAX, img_max):
    """Find jaw edges in image

    Parameters
    ----------
    image : array_like
        2D EPID image of jaw-defined field
    ds :
        Dicom file

    Returns
    -------
    x1_edge : int
        The x1-jaw edge position (arr index)
    x1_edge_mm : float
        The x1-jaw edge position (mm)
    x2_edge : int
        The x2-jaw edge position (arr index)
    x2_edge_mm : float
        The x2-jaw edge position (mm)
    y1_edge : int
        The y1-jaw edge position (arr index)
    y1_edge_mm : float
        The y1-jaw edge position (mm)
    y2_edge : int
        The y1-jaw edge position (arr index)
    y2_edge_mm : float
        The y2-jaw edge position (mm)
    """

    x1_edge, x1_edge_mm = _find_edge(
        image=image,
        ds=ds,
        along_x=True,
        from_negative=True,
        beam_CAX=beam_CAX,
        img_max=img_max,
    )
    x2_edge, x2_edge_mm = _find_edge(
        image=image,
        ds=ds,
        along_x=True,
        from_negative=False,
        beam_CAX=beam_CAX,
        img_max=img_max,
    )
    y1_edge, y1_edge_mm = _find_edge(
        image=image,
        ds=ds,
        along_x=False,
        from_negative=True,
        beam_CAX=beam_CAX,
        img_max=img_max,
    )
    y2_edge, y2_edge_mm = _find_edge(
        image=image,
        ds=ds,
        along_x=False,
        from_negative=False,
        beam_CAX=beam_CAX,
        img_max=img_max,
    )

    return (x1_edge, x1_edge_mm,
            x2_edge, x2_edge_mm,
            y1_edge, y1_edge_mm,
            y2_edge, y2_edge_mm)


def _get_dcm_jaw_pos(ds):
    """Get the jaw positions recorded in the .dcm file

    Parameters
    ----------
    ds :
        Dicom of the relevant jaw-defined field image

    Returns
    -------
    x1 :
        Position of x1-jaw in [mm]
    x2 :
        Position of x2-jaw in [mm]
    y1 :
        Position of y1-jaw in [mm]
    y1 :
        Position of y2-jaw in [mm]
    """

    # Get x/y jaw positions
    x_jaws = ds.ExposureSequence[0].BeamLimitingDeviceSequence[
        0].LeafJawPositions
    y_jaws = ds.ExposureSequence[0].BeamLimitingDeviceSequence[
        1].LeafJawPositions

    return x_jaws[0], x_jaws[1], y_jaws[0], y_jaws[1]


def _make_qat_dict_entry(jaw_dcm_pos, jaw_calc_pos, qat_dict, jaw_str):


    if "1" in jaw_str:
        cor_fac = -1
    else:
        cor_fac = 1

    this_str = "%s %d" % (jaw_str, round(jaw_dcm_pos * cor_fac / 10, 0))


    if this_str in qat_dict.keys():

        # If a list
        if isinstance(qat_dict[this_str], list):

            qat_dict[this_str] = (
                    qat_dict[this_str] + [jaw_calc_pos * cor_fac]
            )

        else:  # make it a list
            qat_dict[this_str] = (
                    [qat_dict[this_str]] + [jaw_calc_pos * cor_fac]
            )
    else:
        qat_dict[this_str] = [jaw_calc_pos * cor_fac]

    return qat_dict


###############################################################################

pdf_report_path = io.BytesIO()

zip_path = FILE.name  # For use in QA Track with file upload

tmp_dir = tempfile.mkdtemp()

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

__D_DIR = tmp_dir

# Find the MV CAX
beam_CAX = _find_CAX(
    ds_1=dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[0])),
    ds_2=dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[1])),
)

cc = 0  # Init counter
jaw_pos_list = []  # List of jaw positions

img_fig_list = []  # List of images, for plotting later

# For each image not used to find the MV CAX
for ff in sorted(os.listdir(__D_DIR))[2:]:

    ds = dcmread(os.path.join(__D_DIR, ff))  # Get dicom

    # Get image, normalize, and store max value
    image = ds.pixel_array
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    img_max = np.max(image)

    #  Transpose image if at Col-270 to correctly ID jaws
    if round(ds.BeamLimitingDeviceAngle, 0) == 270:
        image = image.T
        this_CAX = beam_CAX[::-1]  # Transpose CAX too
    else:
        this_CAX = beam_CAX

    # Find the jaw edges in the image
    (x1_edge, x1_edge_mm,
     x2_edge, x2_edge_mm,
     y1_edge, y1_edge_mm,
     y2_edge, y2_edge_mm) = _find_all_edges(image=image, ds=ds,
                                            beam_CAX=this_CAX,
                                            img_max=img_max)

    # Get jaw positions from plan
    x1, x2, y1, y2 = _get_dcm_jaw_pos(ds=ds)

    # Store the jaw positions here
    jaw_pos_list.append(
        (
            (x1, x2, y1, y2),
            (x1_edge_mm, x2_edge_mm, y1_edge_mm, y2_edge_mm)
        )
    )

    # Get pixel spacing from DICOM (assumed format: [row_spacing, col_spacing])
    pixel_spacing = np.array(ds.ImagePlanePixelSpacing, dtype=float)

    # Plot the image and overlay the detected edge line.
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Image {cc+1}", fontsize=16, y=0.96)

    plt.imshow(image, cmap='gray')

    center_pixel = np.array([image.shape[0] / 2.0, image.shape[1] / 2.0])

    # Draw a vertical red line at the detected x-edge position.
    plt.axvline(x=x1_edge, color='y', linestyle='-',
                linewidth=2, label='Detected X1-Edge')
    plt.text(x1_edge + 50, float(center_pixel[0]),
             f"Meas: {x1_edge_mm:.2f} mm\nPlan: {x1:.2f} mm\nDiff: "
             f"{(x1_edge_mm - x1):.2f} mm",
             color='red',
             fontsize=9,
             bbox=dict(
                 facecolor='white',
                 alpha=0.9,
                 edgecolor='black',
                 boxstyle='round',
             )
             )
    # Draw a vertical red line at the detected x-edge position.
    plt.axvline(x=x2_edge, color='y', linestyle='--',
                linewidth=2, label='Detected X2-Edge')
    plt.text(x2_edge + 50, float(center_pixel[0]),
             f"Meas: {x2_edge_mm:.2f} mm\nPlan: {x2:.2f} mm\nDiff: "
             f"{(x2_edge_mm - x2):.2f} mm",
             color='red',
             fontsize=9,
             bbox=dict(
                 facecolor='white',
                 alpha=0.9,
                 edgecolor='black',
                 boxstyle='round',
             )
             )
    # Draw a horizontal red line at the detected y-edge position.
    plt.axhline(y=y1_edge, color='red', linestyle='-',
                linewidth=2, label='Detected Y1-Edge')
    plt.text(float(center_pixel[1]), y1_edge + 50,
             f"Meas: {y1_edge_mm:.2f} mm\nPlan: {y1:.2f} mm\nDiff: "
             f"{(y1_edge_mm - y1):.2f} mm",
             color='red',
             fontsize=9,
             bbox=dict(
                 facecolor='white',
                 alpha=0.9,
                 edgecolor='black',
                 boxstyle='round',
             )
             )
    # Draw a horizontal red line at the detected y-edge position.
    plt.axhline(y=y2_edge, color='red', linestyle='--',
                linewidth=2, label='Detected Y2-Edge')
    plt.text(float(center_pixel[1]), y2_edge + 50,
             f"Meas: {y2_edge_mm:.2f} mm\nPlan: {y2:.2f} mm\nDiff: "
             f"{(y2_edge_mm - y2):.2f} mm",
             color='red',
             fontsize=9,
             bbox=dict(
                 facecolor='white',
                 alpha=0.9,
                 edgecolor='black',
                 boxstyle='round',
             )
             )

    cc += 1
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.legend(loc='lower right')
    plt.tight_layout()
    img_fig_list.append(fig)
    # plt.close()

# --- Create the summary page ---

# Make summary_data for plt.table() and dict for QA Track reporting
summary_data = []
qat_dict = dict()
for ii in range(len(jaw_pos_list)):
    (
        (x1, x2, y1, y2),
        (x1_edge_mm, x2_edge_mm, y1_edge_mm, y2_edge_mm)
    ) = jaw_pos_list[ii]

    summary_data.append(
        [
            "%d" % (ii + 1),
            "%.2f" % (x1_edge_mm - x1),
            "%.2f" % (x2_edge_mm - x2),
            "%.2f" % (y1_edge_mm - y1),
            "%.2f" % (y2_edge_mm - y2),
         ]
    )

    qat_dict = _make_qat_dict_entry(
        jaw_dcm_pos=x1,
        jaw_calc_pos=x1_edge_mm,
        qat_dict=qat_dict,
        jaw_str="X1"
    )
    qat_dict = _make_qat_dict_entry(
        jaw_dcm_pos=x2,
        jaw_calc_pos=x2_edge_mm,
        qat_dict=qat_dict,
        jaw_str="X2"
    )
    qat_dict = _make_qat_dict_entry(
        jaw_dcm_pos=y1,
        jaw_calc_pos=y1_edge_mm,
        qat_dict=qat_dict,
        jaw_str="Y1"
    )
    qat_dict = _make_qat_dict_entry(
        jaw_dcm_pos=y2,
        jaw_calc_pos=y2_edge_mm,
        qat_dict=qat_dict,
        jaw_str="Y2"
    )

for ii in list(qat_dict.keys()):
    if "20" in ii:
        qat_dict[ii] = qat_dict[ii][
            np.argmax(np.abs(np.array(qat_dict[ii]) - 20))] / 10
    elif "-20" in ii:
        qat_dict[ii] = qat_dict[ii][
            np.argmax(np.abs(np.array(qat_dict[ii]) + 20))] / 10
    else:
        qat_dict[ii] = qat_dict[ii][0] / 10

# Define column headers for the summary table.
col_labels = [
    "Image Num",
    "X1 Diff (mm)",
    "X2 Diff (mm)",
    "Y1 Diff (mm)",
    "Y2 Diff (mm)"
]
# Create a new figure for the summary.
fig_summary = plt.figure(figsize=(8, 11))
plt.axis('off')
plt.title("Jaw Position Differences Summary", fontsize=14, pad=20)

max_diff_x1 = np.max([abs(float(row[1])) for row in summary_data])
max_diff_x2 = np.max([abs(float(row[2])) for row in summary_data])
max_diff_y1 = np.max([abs(float(row[3])) for row in summary_data])
max_diff_y2 = np.max([abs(float(row[4])) for row in summary_data])

# Prepare the extra text and add it above the table:
extra_text = (
    f"Largest Absolute Deviations:\n"
    f"X1: {max_diff_x1:.2f} mm, X2: {max_diff_x2:.2f} mm, "
    f"Y1: {max_diff_y1:.2f} mm, Y2: {max_diff_y2:.2f} mm"
)
plt.figtext(0.5, 0.85, extra_text, wrap=True, horizontalalignment='center',
            fontsize=12)

# Create the table inside the figure.
table = plt.table(cellText=summary_data,
                  colLabels=col_labels,
                  loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)  # Adjust the row heights

# --- Write PDF with the summary page first, then image pages ---
with PdfPages(pdf_report_path) as pdf:
    pdf.savefig(fig_summary)  # Save summary page as first page.
    plt.close(fig_summary)
    # Now add each image page.
    for fig in img_fig_list:
        pdf.savefig(fig)
        plt.close(fig)

# In QA Track, write the .pdf file to be visible after submission
UTILS.write_file("JawReport.pdf", pdf_report_path)
