"""
Tyson Reimer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from matplotlib.backends.backend_pdf import PdfPages

import zipfile
import tempfile
import io

###############################################################################


def _make_qat_dict_entry(mlc_A_calc_pos, mlc_B_calc_pos,
                         mlc_A_plan_pos, mlc_B_plan_pos,
                         qat_dict):
    """Make entry into dict used for QA Track reporting

    Parameters
    ----------
    mlc_A_calc_pos : array_like
        MLC calculated / measured bank A leaf positions for an image
    mlc_B_calc_pos : array_like
        MLC calculated / measured bank B leaf positions for an image
    mlc_A_plan_pos : array_like
        MLC planned bank A leaf positions for an image
    mlc_B_plan_pos : array_like
        MLC planned bank B leaf positions for an image
    qat_dict : dict
        Dictionary used for QA Track integration

    Returns
    -------
    qat_dict : dict
        Dictionary used for QA Track integration, after adding new entry
    """

    # A and B planned positions
    A_plan_pos = np.median(mlc_A_plan_pos)
    B_plan_pos = np.median(mlc_B_plan_pos)

    # Str format of bank A plan pos and blank B plan pos, in [cm]
    # (Flip sign on B pos to comply with QA Track)
    this_str = "%d %d" % (A_plan_pos, -1 * B_plan_pos)

    A_max_error_idx = np.argmax(np.abs(mlc_A_calc_pos - mlc_A_plan_pos))
    B_max_error_idx = np.argmax(np.abs(mlc_B_calc_pos - mlc_B_plan_pos))

    A_max_err = (mlc_A_calc_pos[A_max_error_idx]
                 - mlc_A_plan_pos[A_max_error_idx])
    B_max_err = (mlc_B_calc_pos[B_max_error_idx]
                 - mlc_B_plan_pos[B_max_error_idx])

    qat_dict[this_str] = A_max_err, B_max_err

    return qat_dict


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

    Returns
    -------
    beam_CAX :
        Indices of MV CAX in image space
    """

    img1 = ds_1.pixel_array
    img2 = ds_2.pixel_array

    c_pix1 = _find_center_of_img(img=img1)
    c_pix2 = _find_center_of_img(img=img2)

    # MV CAX defined as average of the two centers
    # (geometrically corresponds to the CAX, assuming negligible
    # size of collimator isocenter)
    beam_CAX = 0.5 * (c_pix1 + c_pix2)

    beam_CAX = beam_CAX[::-1]

    return beam_CAX


def _idx_to_mm(ds, do_x, image, edge_pixel_position, beam_CAX=None):
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
    """
    origin = 0
    if beam_CAX is not None:
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
        physical_edge = (edge_pixel_position - center_pixel[0]) * \
                        pixel_spacing[1] #+ origin[0]
        if beam_CAX is None:
            physical_edge += origin[0]

    else:  # If doing y
        # For a y-edge, convert the y coordinate relative to center:
        physical_edge = (edge_pixel_position - center_pixel[1]) * \
                        pixel_spacing[0] #+ origin[1]
        if beam_CAX is None:
            physical_edge += origin[1]

    # Correct for EPID SID (should be very close to 1)
    physical_edge = physical_edge * (ds.RTImageSID / 1000)

    return physical_edge


def _find_edge_in_slice(image, slice_index, from_negative, along_x, img_max):
    """Find the jaw edge, defined as 50% of max intensity, in img img_slice

    Parameters
    ----------
    image : array_like
        2D EPID image arr
    slice_index : int
        Index of the img_slice to consider (row or col)
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
        img_slice = image[slice_index, :]
    else:  # If looking for Y-jaws
        img_slice = image[:, slice_index]

    # Check if this img_slice contains the jaw-defined field, or if it's
    # outside the region of interest
    if not (np.max(img_slice) >= 0.7 * img_max):
        return

    # Get the threshold used to define an 'edge' for this img_slice
    threshold = (np.min(img_slice) + np.max(img_slice)) / 2.0

    edge_pos = None  # Start with no edge position

    # Find first index where intensity reaches/exceeds the threshold
    if from_negative:
        indices = np.where(img_slice >= threshold)[0]
    else:  # If going from the positive side of the axis
        indices = np.where(np.flip(img_slice) >= threshold)[0]

    if indices.size != 0:  # If some indices were found

        idx = indices[0]

        # If the index is not at one of the extremes
        if (idx > 5) and (idx < len(img_slice) - 5):

            if from_negative:
                # Linear interpolation between pixel idx-1 and idx:
                frac = (threshold - img_slice[idx - 1]) / (
                        img_slice[idx] - img_slice[idx - 1])
                edge_pos = (idx - 1) + frac

            else:  # If from positive axis

                flip_slice = np.flip(img_slice)

                # Linear interpolation between pixel idx-1 and idx:
                frac = (threshold - flip_slice[idx - 1]) / (
                        flip_slice[idx] - flip_slice[idx - 1])
                edge_pos = (idx - 1) + frac

                # Subtract 1 because 0-based indexing
                edge_pos = len(img_slice) - edge_pos - 1

    return edge_pos


def _find_edge(image, ds, along_x=True, from_negative=True,
               lims=None, beam_CAX=None):
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
    lims :
        If None, no limits used. If a tuple, defines the (min, max)
        indices used to find the edge along the axis perpendicular
        to the slice.

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

    if lims is None:
        ini_slice_idx = 1
        fin_slice_idx = n_slices
    else:
        ini_slice_idx = lims[0]
        fin_slice_idx = lims[1]

    # For each slice (excluding the edge cases)
    for slice_index in range(ini_slice_idx, fin_slice_idx):

        # Get the position of the edge in this slice
        edge_pos = _find_edge_in_slice(
            image,
            slice_index=slice_index,
            from_negative=from_negative,
            along_x=along_x,
            img_max=np.max(image),
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


def _find_leaf_indices_from_comb(comb_img, comb_ds):
    """Extract indices of each leaf from a comb image

    Note: Only extracts indices for the extended leaves (half of total)

    Parameters
    ----------
    comb_img : array_like
        2D pixel array of MLC-defined comb image

    Returns
    -------
    leaf_positions:
        Array of the indices of the leaf positions
    leaf_axis:
        Value in [0, 1]. Defines which axis is perpendicular to leaf
        motion.
    bank_A_leaf_positions:
        Positions (indices) of the leaves in bank A
    bank_B_leaf_positions:
        Positions (indices) of the leaves in bank B
    bank_A_idx:
        The central index used to search for leaves in Bank A. Used for
        calculation of the rotation angle.
    bank_B_idx:
        The central index used to search for leaves in Bank B. Used for
        calculation of the rotation angle.
    """

    # If using HDMLC, more robust to first invert the image
    if '2301' in comb_ds.StationName:  # If RTUS
        img = np.max(comb_img) - comb_img
    else:

        # If not using HDMLC, more robust to not invert the image

        img = comb_img

    # Decide the leaf axis by looking at the mean intensity profiles
    # of the row/s cols. The leaf axis should have higher intensity
    # (because of open-field gap in middle)
    row_means = np.zeros(img.shape[0])
    for ii in range(img.shape[0]):
        row_means[ii] = np.mean(img[ii, :])
    col_means = np.zeros(img.shape[1])
    for jj in range(img.shape[1]):
        col_means[jj] = np.mean(img[:, jj])

    # Set the leaf axis to the higher-intensity axis
    if np.max(row_means) > np.max(col_means):
        leaf_axis = 0
    else:
        leaf_axis = 1

    if leaf_axis == 0:

        # First, auto-identify which region of the image occupied
        # by the MLC leaves in the comb pattern
        peaks = find_peaks(np.abs(np.diff(row_means)), height=50, distance=50)

        peak_indices = peaks[0]

        # HDMLC characteristically has 6 peaks here
        if len(peak_indices) == 6:

            # Hard-code indices based on observed patterns:
            # A-bank is between 1st and 2nd peaks (0-based indexing)
            # B-bank is between 3rd and 4th peaks
            start_idx_B = 3
            end_idx_B = 4
            start_idx_A = 1
            end_idx_A = 2

        else:  # 120MLC characteristically does NOT have 6 peaks

            # Hard-code indices based on observed patterns:
            # A-bank is between 0th and 1st peaks (0-based indexing)
            # B-bank is between 2nd and 3rd peaks
            start_idx_A = 0
            end_idx_A = 1
            start_idx_B = 2
            end_idx_B = 3

        # Identify the part of the image corresponding to bank A/B
        # and average over axis perpendicular to leaf travel
        bank_A_slice = np.mean(
            img[peak_indices[start_idx_A]: peak_indices[end_idx_A], :],
            axis=0
        )
        bank_B_slice = np.mean(
            img[peak_indices[start_idx_B]:peak_indices[end_idx_B], :],
            axis=0
        )

    else:  # For the other leaf axis

        # First, auto-identify which region of the image occupied
        # by the MLC leaves in the comb pattern
        peaks = find_peaks(np.abs(np.diff(col_means)), height=50, distance=50)

        peak_indices = peaks[0]
        if len(peak_indices) == 6:
            # Hard-code indices based on observed patterns:
            # A-bank is between 1st and 2nd peaks (0-based indexing)
            # B-bank is between 3rd and 4th peaks
            start_idx_B = 3
            end_idx_B = 4
            start_idx_A = 1
            end_idx_A = 2
        else:
            # Hard-code indices based on observed patterns:
            # A-bank is between 0th and 1st peaks (0-based indexing)
            # B-bank is between 2nd and 3rd peaks
            start_idx_A = 0
            end_idx_A = 1
            start_idx_B = 2
            end_idx_B = 3

        # Identify the part of the image corresponding to bank A/B
        # and average over axis perpendicular to leaf travel
        bank_A_slice = np.mean(
            img[:, peak_indices[start_idx_A]:peak_indices[end_idx_A]],
            axis=1
        )
        bank_B_slice = np.mean(
            img[:, peak_indices[start_idx_B]:peak_indices[end_idx_B]],
            axis=1
        )


    if len(peak_indices) == 6:  # If using HDMLC

        # Further refine the definition of the A/B slices because
        # there is open-field on either side of the MLC banks
        bank_A_valleys = find_peaks(-bank_A_slice, height=-3000,
                                    distance=10)[0]
        bank_A_slice = bank_A_slice[bank_A_valleys[0]:bank_A_valleys[-1]]
        bank_B_valleys = find_peaks(-bank_B_slice, height=-3000,
                                    distance=10)[0]
        bank_B_slice = bank_B_slice[bank_B_valleys[0]:bank_B_valleys[-1]]

    else:  # If not using HDMLC, then set these to dummy zeroes
        bank_A_valleys = (0, 0)
        bank_B_valleys = (0, 0)

    # Find the leaf peaks within the bank A/B slices
    bank_A_leaf_peaks = find_peaks(
        bank_A_slice,
        height=0.5 * np.max(bank_A_slice),
        distance=10,
    )
    bank_B_leaf_peaks = find_peaks(
        bank_B_slice,
        height=0.5 * np.max(bank_B_slice),
        distance=10,
    )

    # Store the indices used to define the 'bank A' and 'bank B' leaves
    #   Note: Only used ot calculate the EPID-MLC angle
    bank_A_idx = np.mean([peak_indices[start_idx_A],
                         peak_indices[end_idx_A]])
    bank_B_idx = np.mean([peak_indices[start_idx_B],
                          peak_indices[end_idx_B]])

    # --- Section below for plotting extracted leaf positions. Useful
    # for troubleshooting
    # plt_xs_A = np.arange(0, len(bank_A_slice))
    # plt_xs_B = np.arange(0, len(bank_B_slice))
    # plt.figure()
    # plt.rc('font', family='Times New Roman')
    # plt.plot(plt_xs_A, bank_A_slice, color='k', label='Line Profile')
    # plt.scatter(plt_xs_A[bank_A_leaf_peaks[0]],
    #             bank_A_slice[bank_A_leaf_peaks[0]],
    #             marker='x', color='r', label='Leaf Positions')
    # plt.legend()
    # plt.title("Bank A")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(plt_xs_B, bank_B_slice, color='k', label='Line Profile')
    # plt.rc('font', family='Times New Roman')
    # plt.title("Bank B")
    # plt.scatter(plt_xs_B[bank_B_leaf_peaks[0]],
    #             bank_B_slice[bank_B_leaf_peaks[0]],
    #             marker='x', color='r', label='Leaf Positions')
    # plt.legend()
    # plt.show()

    # Add-back the valley index to restore original indexing
    bank_A_leaf_positions = np.array(
        bank_A_leaf_peaks[0]
        + bank_A_valleys[0]
    )
    bank_B_leaf_positions = np.array(
        bank_B_leaf_peaks[0]
        + bank_B_valleys[0]
    )

    # Sort and store the leaf positions
    leaf_positions = np.sort(
        np.concatenate((bank_A_leaf_positions, bank_B_leaf_positions))
    )

    return (leaf_positions, leaf_axis,
            bank_A_leaf_positions, bank_B_leaf_positions,
            bank_A_idx, bank_B_idx)


def _get_bank_A_leaf_positions(img, leaf_indices, leaf_axis, beam_CAX, ds):
    """Get the positions of the leaves in bank A from an MLC-defined img

    Parameters
    ----------
    img : array_like
        2D image of MLC-defined field
    leaf_indices : array_like
        Array of leaf-indices, used to identify individual leaves in
        image
    leaf_axis :
        In [0, 1], defines axis perpendicular to leaf travel
    beam_CAX :
        The coordinates of the MV CAX in the image space
    ds :
        Loaded dicom containing the img image

    Returns
    -------
    leaf_edges :
        Indices of the leaf edges
    leaf_edges_mm :
        Position of the leaf edges, in [mm]
    """

    # Find the leaf edges (in [indices])
    leaf_edges = np.zeros(np.shape(leaf_indices))
    for ii in range(len(leaf_edges)):
        leaf_edges[ii] = _find_edge_in_slice(
            img,
            slice_index=leaf_indices[ii],
            from_negative=False,
            along_x=leaf_axis == 1,
            img_max=np.max(img)
        )

    # Find the leaf edges in [mm]
    leaf_edges_mm = np.zeros_like(leaf_edges)
    for ii in range(len(leaf_edges_mm)):
        leaf_edges_mm[ii] = _idx_to_mm(
            ds=ds,
            do_x=leaf_axis == 1,
            image=img,
            edge_pixel_position=leaf_edges[ii],
            beam_CAX=beam_CAX,
        )

    return leaf_edges, leaf_edges_mm


def _get_bank_B_leaf_positions(img, leaf_indices, leaf_axis, beam_CAX, ds):
    """Get the positions of the leaves in bank B from an MLC-defined img

    Parameters
    ----------
    img : array_like
        2D image of MLC-defined field
    leaf_indices : array_like
        Array of leaf-indices, used to identify individual leaves in
        image
    leaf_axis :
        In [0, 1], defines axis perpendicular to leaf travel
    beam_CAX :
        The coordinates of the MV CAX in the image space
    ds :
        Loaded dicom containing the img image

    Returns
    -------
    leaf_edges :
        Indices of the leaf edges
    leaf_edges_mm :
        Position of the leaf edges, in [mm]
    """

    # Find the leaf edges (in [indices])
    leaf_edges = np.zeros(np.shape(leaf_indices))
    for ii in range(len(leaf_edges)):
        leaf_edges[ii] = _find_edge_in_slice(
            img,
            slice_index=leaf_indices[ii],
            from_negative=True,
            along_x=leaf_axis == 1,
            img_max=np.max(img)
        )

    # Find the leaf edges in [mm]
    leaf_edges_mm = np.zeros_like(leaf_edges)
    for ii in range(len(leaf_edges_mm)):
        leaf_edges_mm[ii] = _idx_to_mm(
            ds=ds,
            do_x=leaf_axis == 1,
            image=img,
            edge_pixel_position=leaf_edges[ii],
            beam_CAX=beam_CAX,
        )

    return leaf_edges, leaf_edges_mm


def find_leaf_positions(img, ds, leaf_indices, leaf_axis, beam_CAX=None):
    """Get the bank A and bank B leaf positions from img

    Parameters
    ----------
    img : array_like
        2D image of MLC-defined field
    ds :
        Loaded dicom containing the img image
    leaf_indices : array_like
        Array of leaf-indices, used to identify individual leaves in
        image
    leaf_axis :
        In [0, 1], defines axis perpendicular to leaf travel
    beam_CAX :
        The coordinates of the MV CAX in the image space
    """

    leaf_edges_A, leaf_edges_A_mm = _get_bank_A_leaf_positions(
        img=img,
        leaf_indices=leaf_indices,
        leaf_axis=leaf_axis,
        beam_CAX=beam_CAX,
        ds=ds,
    )

    leaf_edges_B, leaf_edges_B_mm = _get_bank_B_leaf_positions(
        img=img,
        leaf_indices=leaf_indices,
        leaf_axis=leaf_axis,
        beam_CAX=beam_CAX,
        ds=ds,
    )

    return leaf_edges_A, leaf_edges_A_mm, leaf_edges_B, leaf_edges_B_mm



def _get_rot_ang(comb_img, comb_ds):
    """Get the angle of rotation between EPID and MLCs

    Parameters
    ----------
    comb_img : array_like
        Image of MLC-defined comb
    comb_ds :
        Loaded .dcm of the comb_img

    Returns
    -------
    rot_ang :
        Angle of rotation between the EPID and MLCs in [degrees]
    """

    # Get the leaf positions
    (leaf_positions, leaf_axis, bank_A_leaf_pos, bank_B_leaf_pos,
     bank_A_idx, bank_B_idx) = (
        _find_leaf_indices_from_comb(comb_img=comb_img, comb_ds=comb_ds)
    )

    # Denominator for tangent equation
    denom = np.abs(bank_A_idx - bank_B_idx)

    # Define the argument for the tangent function, i.e., opposite
    # over adjacent sides of our triangle
    tan_arg = (np.mean(bank_A_leaf_pos - bank_B_leaf_pos)) / (denom)

    # Convert to deg
    rot_ang = np.rad2deg(np.arctan(tan_arg))

    return rot_ang


def extract_leaf_positions(comb_ds1, comb_ds2, rot_ang):
    """Extract the positions of all 60 MLC leafs using 2 comb images

    Use 2 comb images for better identification of leafs. Challenges
    when using peaks and valleys in same image observed during
    implementation.

    Parameters
    ----------
    comb_ds1 :
        Loaded dicom of MLC-defined comb
    comb_ds2 :
        Loaded dicom of MLC-defined comb, with extended/retracted
        leafs switched vs comb_ds1
    rot_ang :
        The angle of rotation between EPID and MLC

    Returns
    -------
    leaf_positions :
        Indices of the leaf positions, used to identify individual
        leaves in other images
    leaf_axis :
        The axis perpendicular to the direction of leaf motion
    """

    img = comb_ds1.pixel_array
    img = rotate(img, rot_ang, reshape=False)

    leaf_positions, leaf_axis, _, _, _, _ = (
        _find_leaf_indices_from_comb(comb_img=img, comb_ds=comb_ds1)
    )

    # Keep every 2nd position (ignore one of the banks
    leaf_positions = leaf_positions[::2]

    img2 = comb_ds2.pixel_array
    img2 = rotate(img2, rot_ang, reshape=False)
    leaf_positions2, _, _, _, _, _ = (
        _find_leaf_indices_from_comb(comb_img=img2, comb_ds=comb_ds2)
    )

    # Concatenate leaf positions from first comb image with those
    # from second
    leaf_positions = np.concatenate((leaf_positions, leaf_positions2[::2]))

    return leaf_positions, leaf_axis


###############################################################################

# Create path for generating pdf report
pdf_report_path = io.BytesIO()

zip_path = FILE.name  # Get path to uploaded .zip file

tmp_dir = tempfile.mkdtemp()  # Make temp dir for lodaing files in zip

# Open the .zip folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

__D_DIR = tmp_dir  # Define dir for the data


# Find the CAX of the MV beam as the average of the positions
# of the middle of two jaw-defined fields, one at Col-90
# and one at Col-270
beam_CAX = _find_CAX(
    ds_1=dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[0])),
    ds_2=dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[1])),
)

# Load the two comb images first
comb_ds1 = dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[2]))
comb_ds2 = dcmread(os.path.join(__D_DIR, sorted(os.listdir(__D_DIR))[3]))

# Get the rotation between EPID and MLC
rot_ang1 = _get_rot_ang(comb_img=comb_ds1.pixel_array, comb_ds=comb_ds1)
rot_ang2 = _get_rot_ang(comb_img=comb_ds2.pixel_array, comb_ds=comb_ds2)
rot_ang = float(np.mean([rot_ang1, rot_ang2]))

# Find the leaf positions from the two comb images
leaf_positions, leaf_axis = extract_leaf_positions(
    comb_ds1=comb_ds1,
    comb_ds2=comb_ds2,
    rot_ang=rot_ang,
)
# Sort to idx leaves correctly
leaf_positions = np.sort(leaf_positions)

cc = 0  # Init counter

# Hard-code expected MLC positions.
if '2301' in comb_ds1.StationName:  # If RTUS
    MLC_pos = (
        np.tile(
            np.array([-100, 0, 100, -190, -190, -190]
                     ).reshape(6, 1), (1, 60)
        ),
        np.tile(
            np.array([190, 190, 190, 100, 0, -100]
                     ).reshape(6, 1), (1, 60)
        ),
    )
else:  # If using non-RTUS (i.e., 120MLC)
    MLC_leaves_row = np.concatenate(
        (np.array([-150, -160, -170, -180]),
         np.full(60 - 8, -190),
         np.array([-180, -170, -160, -150]))
    )

    MLC_pos = (
        np.vstack(
            [np.full(60, v) for v in [-100, 0, 100]]
            + [MLC_leaves_row] * 3
        ),
        np.flip(np.vstack(
            [np.full(60, v) for v in [-100, 0, 100]]
            + [-1 * MLC_leaves_row] * 3
        ))
        )

# Init lists for creating .pdf report
img_fig_list = []
all_errors = []
alerts = []  # tuples (level, bank, leaf_idx, err, img_num)
worst_A = []  # tuples (img_num, leaf_idx, err)
worst_B = []
image_number = 1  # Image counter for pdf report

# Hard-code tol levels
tol_level = 2  # Level 1 tolerance, 2 mm
action_level = 5  # Level 2 tolerance, 5 mm

mlc_pos_list = []  # Init list for QA Track integration

# For every image (excluding the CAX and comb images)
for ff in sorted(os.listdir(__D_DIR))[4:]:

    ds = dcmread(os.path.join(__D_DIR, ff))  # Get dicom

    image = ds.pixel_array

    # Apply rotation
    image = rotate(image, angle=rot_ang, reshape=False)

    # Normalize image
    image = (image- np.min(image)) / (np.max(image) - np.min(image))
    img_max = np.max(image)

    # Extract the positions in indices and [mm] of each leaf
    (A_leafs, A_leafs_mm,
     B_leafs, B_leafs_mm) = find_leaf_positions(
        img=image,
        ds=ds,
        leaf_indices=leaf_positions,
        leaf_axis=leaf_axis,
        beam_CAX=beam_CAX,
    )

    # Define the edges of the A-bank and B-bank
    A_edge = int(np.median(A_leafs))
    A_edge_mm = np.median(A_leafs_mm)
    B_edge = int(np.median(B_leafs))
    B_edge_mm = np.median(B_leafs_mm)

    # Get the position errors, in [mm], of the individual leaves
    A_errs = A_leafs_mm - MLC_pos[1][cc, :]
    B_errs = B_leafs_mm - MLC_pos[0][cc, :]

    mlc_pos_list.append(
        (A_leafs_mm, B_leafs_mm, MLC_pos[1][cc, :], MLC_pos[0][cc, :])
    )

    # Get the expected positions of the MLC banks here
    bank_B_pos = np.median(MLC_pos[0][cc, :])
    bank_A_pos = np.median(MLC_pos[1][cc, :])

    # collect every error and generate alerts
    for idx, err in enumerate(A_errs):
        ae = abs(err)
        all_errors.append(ae)
        if ae > action_level:
            alerts.append(("ACTION ALERT", "A", idx, err, image_number))
        elif ae > tol_level:
            alerts.append(("ALERT", "A", idx, err, image_number))
    for idx, err in enumerate(B_errs):
        ae = abs(err)
        all_errors.append(ae)
        if ae > action_level:
            alerts.append(("ACTION ALERT", "B", idx, err, image_number))
        elif ae > tol_level:
            alerts.append(("ALERT", "B", idx, err, image_number))

    # find worst per bank
    wA_idx = int(np.argmax(np.abs(A_errs)))
    wA_err = A_errs[wA_idx]
    worst_A.append((image_number, wA_idx, wA_err))
    wB_idx = int(np.argmax(np.abs(B_errs)))
    wB_err = B_errs[wB_idx]
    worst_B.append((image_number, wB_idx, wB_err))

    # planned bank positions (for overlay text)
    bank_A_pos = np.median(MLC_pos[1][cc, :])
    bank_B_pos = np.median(MLC_pos[0][cc, :])

    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(2, 2,
                          figure=fig,
                          width_ratios=[3, 1],
                          height_ratios=[1, 1],
                          wspace=0.3, hspace=0.4)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_Bbar = fig.add_subplot(gs[0, 1])
    ax_Abar = fig.add_subplot(gs[1, 1])

    fig.suptitle("Image %d" % (image_number), fontsize=16, y=0.96)
    fig.text(0.5, 0.88,
             "Bank A: Worst leaf was %d with position error %.2f mm\n"
         "Bank B: Worst leaf was %d with position error %.2f mm"
             % (wA_idx + 1, wA_err, wB_idx + 1, wB_err),
             ha="center")

    ax_img.imshow(image, cmap="gray")
    ax_img.axhline(A_edge, color="y", linestyle="-", linewidth=2,
                   label="Detected A-bank Edge")
    ax_img.axhline(B_edge, color="y", linestyle="--", linewidth=2,
                   label="Detected B-bank Edge")
    cx = image.shape[1] / 2.0

    ax_img.text(cx, A_edge + 150,
                "Meas: %.2f mm\nPlan: %.2f mm"
                "\nMedian Diff: %.2f mm" % (
                    A_edge_mm, bank_A_pos, A_edge_mm - bank_A_pos),
                fontsize=9, color="red",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black",
                          boxstyle="round"))
    ax_img.text(cx, B_edge + 150,
                "Meas: %.2f mm\nPlan: %.2f mm"
                "\nMedian Diff: %.2f mm" % (
                    B_edge_mm, bank_B_pos, B_edge_mm-bank_B_pos),
                fontsize=9, color="red",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black",
                          boxstyle="round"))

    ax_img.set_xlabel("Pixel Column")
    ax_img.set_ylabel("Pixel Row")
    ax_img.legend(loc="upper right")

    x = np.arange(len(B_errs))
    B_colors = ["b"] * len(B_errs)
    B_colors[wB_idx] = "r"
    ax_Bbar.bar(x, B_errs, width=0.75, color=B_colors, edgecolor="k",
                linewidth=0.5)
    ax_Bbar.set_title("Bank B errors")
    ax_Bbar.set_xlabel("Leaf Number")
    ax_Bbar.set_ylabel("Position Error (mm)")

    x = np.arange(len(A_errs))
    A_colors = ["b"] * len(A_errs)
    A_colors[wA_idx] = "r"
    ax_Abar.bar(x, A_errs, width=0.75, color=A_colors, edgecolor="k",
                linewidth=0.5)
    ax_Abar.set_title("Bank A errors")
    ax_Abar.set_xlabel("Leaf Number")
    ax_Abar.set_ylabel("Position Error (mm)")

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    img_fig_list.append(fig)
    plt.close(fig)

    cc += 1
    image_number += 1

# Make summary_data for plt.table() and dict for QA Track reporting
summary_data = []
qat_dict = dict()

for ii in range(len(mlc_pos_list)):  # For each target image

    # Get leaf measured/planned positions
    A_leafs, B_leafs, A_leafs_plan, B_leafs_plan = mlc_pos_list[ii]

    # Make entry for QA Track, reporting result in [cm] units
    qat_dict = _make_qat_dict_entry(
        mlc_A_calc_pos=A_leafs / 10,
        mlc_B_calc_pos=B_leafs / 10,
        mlc_A_plan_pos=A_leafs_plan / 10,
        mlc_B_plan_pos=B_leafs_plan / 10,
        qat_dict=qat_dict,
    )

# Create summary page for the pdf report
fig_summary = plt.figure(figsize=(8, 11))
fig_summary.patch.set_facecolor("white")

fig_summary.text(0.5, 0.9, "MLC Leaf Position Deviations Summary",
                 ha="center", fontsize=18, weight="bold")

y = 0.84
dy = 0.035

# Report any alerts identified
for level, bank, leaf, err, img_num in alerts:
    if level == "ACTION ALERT":
        color = "red"
        triggered_level = action_level
    else:  # Level == "ALERT"
        color = "darkgoldenrod"
        triggered_level = tol_level

    fig_summary.text(0.1, y,
                     f"{level}: Bank {bank} Leaf {leaf + 1} in "
                     f"image number {img_num} "
                     f"had position error {err:.2f} mm\n    above "
                     f"Level {2 if level.startswith('ACTION') else 1} "
                     f"tolerance of {triggered_level:.2f} mm.",
                     color=color, fontsize=10)
    y -= dy

# overall stats
max_err = max(all_errors) if all_errors else 0.0
avg_err = np.mean(all_errors) if all_errors else 0.0
fig_summary.text(0.1, y,
                 f"Max leaf position deviation: {max_err:.2f} mm",
                 fontsize=12)
y -= dy
fig_summary.text(0.1, y,
                 f"Average leaf position deviation: {avg_err:.2f} mm",
                 fontsize=12)

# Create pdf report
with PdfPages(pdf_report_path) as pdf:
    pdf.savefig(fig_summary)
    plt.close(fig_summary)
    for fig in img_fig_list:
        pdf.savefig(fig)
        plt.close(fig)

# Make the .pdf file
UTILS.write_file("MLCreport.pdf", pdf_report_path)
