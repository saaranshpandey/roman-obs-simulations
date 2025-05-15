import numpy as np
import pandas as pd
import healpy as hp
import matplotlib
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.wcs import WCS
from collections import Counter
import yaml
from dustmaps.config import config
from dustmaps.sfd import fetch, SFDQuery
import os
import random

NSIDE = 4096
def get_vertices(df, index):
    '''
    Function to convert ra and dec values into vectors compatible with healpy
    Arguments: The dataframe and the index value of the row corresponding to the SCA that will be plotted.
    Returns: An array of vertices in the form of vectors for each corner of the SCA.
    '''
    ra1 = df['RA1'][index]
    ra2 = df['RA2'][index]
    ra3 = df['RA3'][index]
    ra4 = df['RA4'][index]
    dec1 = df['DEC1'][index]
    dec2 = df['DEC2'][index]
    dec3 = df['DEC3'][index]
    dec4 = df['DEC4'][index]
    def ra_dec_to_theta_phi(ra, dec):
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        theta = np.pi / 2 - dec_rad
        phi = ra_rad
        return theta, phi
    theta1, phi1 = ra_dec_to_theta_phi(ra1, dec1)
    theta2, phi2 = ra_dec_to_theta_phi(ra2, dec2)
    theta3, phi3 = ra_dec_to_theta_phi(ra3, dec3)
    theta4, phi4 = ra_dec_to_theta_phi(ra4, dec4)
    vec1 = hp.ang2vec(theta1, phi1)
    vec2 = hp.ang2vec(theta2, phi2)
    vec3 = hp.ang2vec(theta3, phi3)
    vec4 = hp.ang2vec(theta4, phi4)
    vertices = np.array([vec1, vec2, vec3, vec4])
    return vertices


def translate_squares(df, shift=0.01, upward_repetitions=1, left_translation=True, downward_repetitions=1):
    """
    Function to create the tiling of the footprint, following the snake pattern outlined in (Wang et al., 2023). It works left to right if visualized using healpy gnomview().
    Arguments:
        df: Must be a dataframe that contains only one pointing (only 18 rows). Contains columns RA{i} and DEC{i} for the 4 corner coordinates of every SCA.
        shift: change this value to alter the spacing between rows of pointings. (Increase it and the space between decreases and vice versa.)
        upward_repetitions: One less than the number of desired rows
        left translation: True if another column is desired.
        downward_repetitions: One less than the number of desired rows
    Output: Original dataframe with the rows corresponding to the new pointings appended to the end.

    """
    # Step 1: Identify the bounding box
    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()


    # Step 2: Calculate the upward translation distance
    translation_distance_up = max_dec - min_dec - shift

    # Initialize the result dataframe with the original data
    result_df = df.copy()

    # Step 3: Apply the upward translations
    for i in range(upward_repetitions):
        # Translate each corner's Dec coordinates upward
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += (i + 1) * translation_distance_up

        # Append the translated squares to the result dataframe
        result_df = pd.concat([result_df, translated_df], ignore_index=True)

    if left_translation:
        # Step 4: Extract the last set of translated squares
        last_set = result_df.tail(18)

        # Step 5: Calculate the left translation distance
        min_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].min().min()
        max_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].max().max()
        translation_distance_left = max_ra - min_ra - shift

        # Step 6: Apply the left translation to the last set
        last_set_left_translated = last_set.copy()
        last_set_left_translated[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left

        # Append the left-translated squares to the result dataframe
        result_df = pd.concat([result_df, last_set_left_translated], ignore_index=True)

        # Step 7: Apply the downward translations to the left-translated set
        for i in range(downward_repetitions):
            downward_translated_df = last_set_left_translated.copy()
            downward_translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= (i + 1) * translation_distance_up
            result_df = pd.concat([result_df, downward_translated_df], ignore_index=True)

    return result_df

def rotate_point(x, y, angle_rad):
    """Rotate a point clockwise by a given angle around the origin (0, 0)."""
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    x_new = cos_angle * x + -sin_angle * y
    y_new = sin_angle * x + cos_angle * y

    return x_new, y_new



def rotate_squares_healpy(df, foot_angle_deg):
    '''
    Function to rotate the footprint clockwise once as viewed in the gnomview projection.
    Commented out, there is code to also rotate individual pointings if desired. The angle of rotation and a list of lists of indices corresponding to the groups of SCAs in a pointing would need to be added as arguments in the function.
    The previous names for these arguments were point_angle_deg and subset_indices_list.
    Inputs:
        Dataframe that contains the rows for the footprint.
        foot_angle_deg is the rotation angle in degrees that you want to rotate the footprint by.
    Output: A dataframe of corner ra and dec coordinates for the rotated footprint. (If angle is 0 then it is the ra and dec coordinates for the original footprint).
    '''
    # Convert angles from degrees to radians
    angle_rad = np.radians(foot_angle_deg)
    #subset_angle_rad = np.radians(point_angle_deg)

    # Calculate the center of the entire set
    ra_center = df[[f'RA{i}' for i in range(1, 5)]].values.mean()
    dec_center = df[[f'DEC{i}' for i in range(1, 5)]].values.mean()

    # Initialize an empty DataFrame to store rotated squares
    rotated_data = {f'RA{i}': [] for i in range(1, 5)}
    rotated_data.update({f'DEC{i}': [] for i in range(1, 5)})

    # Apply the initial rotation to the entire set
    for index, row in (df.tail(648)).iterrows():
        for i in range(1, 5):
            ra, dec = row[f'RA{i}'], row[f'DEC{i}']

            # Translate points to origin (center of the entire set)
            ra_translated = ra - ra_center
            dec_translated = dec - dec_center

            # Rotate the points
            ra_rot, dec_rot = rotate_point(ra_translated, dec_translated, angle_rad)

            # Translate points back to the original position
            rotated_data[f'RA{i}'].append(ra_rot + ra_center)
            rotated_data[f'DEC{i}'].append(dec_rot + dec_center)

    # Create a DataFrame for the initially rotated squares
    rotated_df = pd.DataFrame(rotated_data)

    # Apply the additional rotation to each subset
    #for subset_indices in subset_indices_list:
     #   subset_df = rotated_df.iloc[subset_indices]
      #  ra_center_subset = subset_df[[f'RA{i}' for i in range(1, 5)]].values.mean()
       # dec_center_subset = subset_df[[f'DEC{i}' for i in range(1, 5)]].values.mean()

        #for index in subset_indices:
         #   row = rotated_df.loc[index]
          #  for i in range(1, 5):
           #     ra, dec = row[f'RA{i}'], row[f'DEC{i}']

                # Translate points to origin (center of the subset)
              #  ra_translated = ra - ra_center_subset
               # dec_translated = dec - dec_center_subset

                # Rotate the points
                #ra_rot, dec_rot = rotate_point(ra_translated, dec_translated, subset_angle_rad)

                # Translate points back to the original position
                #rotated_df.at[index, f'RA{i}'] = ra_rot + ra_center_subset
                #rotated_df.at[index, f'DEC{i}'] = dec_rot + dec_center_subset

    # Concatenate the original and rotated DataFrames
    #result_df = pd.concat([df, rotated_df], ignore_index=True)
    result_df = rotated_df

    return result_df


def rotate_squares_custom_healpy(df, foot_angle_deg, ra_axis, dec_axis):
    '''
    Function to rotate the sky map clockwise around a user-specified axis.
    '''
    angle_rad = np.radians(foot_angle_deg)

    rotated_data = {f'RA{i}': [] for i in range(1, 5)}
    rotated_data.update({f'DEC{i}': [] for i in range(1, 5)})

    for _, row in df.iterrows():
        for i in range(1, 5):
            ra, dec = row[f'RA{i}'], row[f'DEC{i}']

            ra_translated = ra - ra_axis
            # ra_translated = (ra - ra_axis + 360) % 360
            # print(ra_translated)
            dec_translated = dec - dec_axis
            ra_rot, dec_rot = rotate_point(ra_translated, dec_translated, angle_rad)

            rotated_data[f'RA{i}'].append(ra_rot + ra_axis)
            rotated_data[f'DEC{i}'].append(dec_rot + dec_axis)

    rotated_df = pd.DataFrame(rotated_data)
    return rotated_df

def rotate_squares_custom_astropy(df, foot_angle_deg, ra_axis, dec_axis):
    """
    Rotates the sky map properly in spherical space around a custom axis using
    a local tangent-plane (SkyOffsetFrame).
    """
    # 1) Define the pivot/center as a SkyCoord in ICRS
    center_coord = SkyCoord(ra=ra_axis*u.deg, dec=dec_axis*u.deg, frame='icrs')

    # 2) Create a local tangent-plane frame with that center as (0,0)
    offset_frame = center_coord.skyoffset_frame()

    # Convert rotation angle to radians
    angle_rad = np.radians(foot_angle_deg)
    cosA = np.cos(angle_rad)
    sinA = np.sin(angle_rad)

    # Prepare output columns
    rotated_data = {f'RA{i}': [] for i in range(1, 5)}
    rotated_data.update({f'DEC{i}': [] for i in range(1, 5)})
    rotated_data.update({'cell': []})
    rotated_data.update({'imaging_tier': []})
    rotated_data.update({'SCA': []})
    
    # 3) For each corner, convert -> tangent plane -> 2D rotate -> convert back
    for _, row in df.iterrows():
        rotated_data['cell'].append(row['cell'])
        rotated_data['imaging_tier'].append(row['imaging_tier'])
        rotated_data['SCA'].append(row['SCA'])
        for i in range(1, 5):
            orig_ra = row[f'RA{i}']
            orig_dec = row[f'DEC{i}']

            # Original corner in ICRS
            corner_icrs = SkyCoord(ra=orig_ra*u.deg, dec=orig_dec*u.deg, frame='icrs')

            # Transform to offset (tangent) frame => x=lon, y=lat in deg
            corner_offset = corner_icrs.transform_to(offset_frame)
            x = corner_offset.lon.deg
            y = corner_offset.lat.deg

            # 2D rotation in the local tangent plane
            x_new = cosA*x - sinA*y
            y_new = sinA*x + cosA*y

            # Convert back to ICRS
            corner_offset_new = SkyCoord(lon=x_new*u.deg, lat=y_new*u.deg,
                                         frame=offset_frame)
            corner_icrs_new = corner_offset_new.transform_to('icrs')

            # Store results
            rotated_data[f'RA{i}'].append(corner_icrs_new.ra.deg)
            rotated_data[f'DEC{i}'].append(corner_icrs_new.dec.deg)
            

    # 4) Build and return a new DataFrame
    rotated_df = pd.DataFrame(rotated_data)
    return rotated_df

###OLD - CAN IGNORE###
def calculate_coverage(df, total_degrees=365):
    """
    Calculates the effectiveness of sky coverage after each degree rotation and overall.

    Returns:
        coverage_per_rotation: List of effectiveness (fraction of original area covered) at each degree.
        overall_coverage: Average effectiveness over the year.
    """
    #Fetch pixels for the original region
    original_pixels = set()
    for index in df.index:
        vertices = get_vertices(df, index)  # Get the vertices for each region at t=0
        original_pixels.update(hp.query_polygon(NSIDE, vertices, inclusive=False))

    total_original_pixels = len(original_pixels)

    #Rotate the region and calculate coverage
    coverage_per_rotation = []
    for degree in range(total_degrees + 1):  # Include t=0
        # Rotate the region
        rotated_df = rotate_squares(df, degree)

        #Fetch pixels for rotated region
        rotated_pixels = set()
        for index in rotated_df.index:
            vertices = get_vertices(rotated_df, index)
            rotated_pixels.update(hp.query_polygon(NSIDE, vertices, inclusive=False))

        #Overlap calculation
        overlap_pixels = original_pixels & rotated_pixels  # Intersection of original and rotated
        coverage_fraction = len(overlap_pixels) / total_original_pixels


        coverage_per_rotation.append(coverage_fraction)

    #Overall coverage (mean) calculation
    overall_coverage = np.mean(coverage_per_rotation)

    return coverage_per_rotation, overall_coverage


def visualize_effectiveness(effectiveness_per_rotation, overall_effectiveness):
    """
    For visualizing the efficiency (overlap) of sky coverage at various values of degree rotation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(effectiveness_per_rotation)), effectiveness_per_rotation, label="Effectiveness per Rotation")
    plt.xlabel("Rotation (Degrees)")
    plt.ylabel("Effectiveness")
    plt.title("Effectiveness of Sky Coverage with Each Rotation")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Overall Effectiveness After 1 Year: {overall_effectiveness * 100:.2f}%")


def create_trans_sq(og_df, data):
    """
    Utilised the code from earlier notebook to create the tiling strategy 8*5 = 40 exposures
    """
    up1, up2, up3, up4, up5 = data['up1'], data['up2'], data['up3'], data['up4'], data['up5']
    dw1, dw2, dw3, dw4, dw5 = data['dw1'], data['dw2'], data['dw3'], data['dw4'] ,data['up5']
    shift1, shift2, shift3, shift4, shift5 = data['shift1'], data['shift2'], data['shift3'], data['shift4'], data['shift5']
    left_tr1, left_tr2, left_tr3, left_tr4, left_tr5 = data['left_tr1'], data['left_tr2'], data['left_tr3'], data['left_tr4'], data['left_tr5']
    pointing_num = data['pointing_num']

    df = og_df[(og_df['POINTING'] == pointing_num)]

    #Finishes the first column and makes the second column
    trans = translate_squares(df, shift=shift1, upward_repetitions=up1, left_translation=left_tr1, downward_repetitions=dw1)
    #Starts the third column
    trans2 = translate_squares(trans, shift=shift2, upward_repetitions=up2, left_translation=left_tr2, downward_repetitions=dw2)
    #Finishes the third column and makes the fourth
    trans3 = translate_squares(trans2, shift=shift3, upward_repetitions=up3, left_translation=left_tr3, downward_repetitions=dw3)
    #Starts the fifth column
    trans4 = translate_squares(trans3, shift=shift4, upward_repetitions=up4, left_translation=left_tr4, downward_repetitions=dw4)
    #Finishes the fifth column
    new_df = translate_squares(trans4, shift=shift5, upward_repetitions=up5, left_translation=left_tr5, downward_repetitions=dw5)


    return new_df

###OLD - CAN IGNORE###
def create_trans_sq_customTest(og_df, data):
    """
    Utilised for testing purpose on a smaller region.
    """
    up1, up2, up3, up4, up5 = data['up1'], data['up2'], data['up3'], data['up4'], data['up5']
    dw1, dw2, dw3, dw4, dw5 = data['dw1'], data['dw2'], data['dw3'], data['dw4'] ,data['up5']
    shift1, shift2, shift3, shift4, shift5 = data['shift1'], data['shift2'], data['shift3'], data['shift4'], data['shift5']
    left_tr1, left_tr2, left_tr3, left_tr4, left_tr5 = data['left_tr1'], data['left_tr2'], data['left_tr3'], data['left_tr4'], data['left_tr5']
    pointing_num = data['pointing_num']

    df = og_df[(og_df['POINTING'] == pointing_num)]

    #Finishes the first column and makes the second column
    trans = translate_squares(df, shift=shift1, upward_repetitions=up1, left_translation=left_tr1, downward_repetitions=dw1)
    #Starts the third column
    # trans2 = translate_squares(trans, shift=shift2, upward_repetitions=up2, left_translation=left_tr2, downward_repetitions=dw2)
    # #Finishes the third column and makes the fourth
    # trans3 = translate_squares(trans2, shift=shift3, upward_repetitions=up3, left_translation=left_tr3, downward_repetitions=dw3)
    # #Starts the fifth column
    # trans4 = translate_squares(trans3, shift=shift4, upward_repetitions=up4, left_translation=left_tr4, downward_repetitions=dw4)
    # #Finishes the fifth column
    # new_df = translate_squares(trans4, shift=shift5, upward_repetitions=up5, left_translation=left_tr5, downward_repetitions=dw5)


    return trans


def remove_edges(og_df, new_df, filter_val='K213', filter_pointing_num='K'):
    """
    Removing the corners to get the required 36 exposures.
    """
    filter_dict = {
        'R': [0, 7, 39, 32],
        'Z': [55, 62, 94, 87],
        'J': [165, 172, 204, 197],
        'H': [220, 227, 259, 252],
        'F': [275, 282, 314, 307],
        'K': [330, 337, 369, 362]
    }

    filter_vals = filter_dict[filter_pointing_num]

    filter_df = og_df[(og_df['FILTER'] == filter_val)]
    sorted_df = filter_df.sort_values(by='POINTING')

    ra1_array = new_df['RA1'].values
    ra2_array = new_df['RA2'].values
    ra3_array = new_df['RA3'].values
    ra4_array = new_df['RA4'].values
    dec1_array = new_df['DEC1'].values
    dec2_array = new_df['DEC2'].values
    dec3_array = new_df['DEC3'].values
    dec4_array = new_df['DEC1'].values

    SCA_array = new_df['SCA'].values
    filter_array = new_df['FILTER'].values
    exptime_array = new_df['EXPTIME'].values

    pointing_array = sorted_df['POINTING'].values
    mjd_array = sorted_df['MJD'].values
    dateobs_array = sorted_df['DATE-OBS'].values

    shape = SCA_array.shape[0]
    data = {
        'POINTING': pointing_array[:shape], #720
        'SCA': SCA_array,
        'FILTER': filter_array,
        'MJD': mjd_array[:shape],
        'EXPTIME': exptime_array,
        'DATE-OBS': dateobs_array[:shape],
        'RA1': ra1_array,
        'DEC1':dec1_array,
        'RA2': ra2_array,
        'DEC2': dec2_array,
        'RA3': ra3_array,
        'DEC3': dec3_array,
        'RA4': ra4_array,
        'DEC4': dec4_array
    }

    nuevo_df = pd.DataFrame(data)
    indices_to_drop = nuevo_df[nuevo_df['POINTING'] == filter_vals[0]].index
    nuevo_df.drop(indices_to_drop, inplace=True)


    indices_to_drop = nuevo_df[nuevo_df['POINTING'] == filter_vals[1]].index
    nuevo_df.drop(indices_to_drop, inplace=True)

    indices_to_drop = nuevo_df[nuevo_df['POINTING'] == filter_vals[2]].index
    nuevo_df.drop(indices_to_drop, inplace=True)

    indices_to_drop = nuevo_df[nuevo_df['POINTING'] == filter_vals[3]].index
    nuevo_df.drop(indices_to_drop, inplace=True)

    nuevo_df = nuevo_df.reset_index(drop=True)

    return nuevo_df


def gen_tiling(og_df, data):
    '''
    Utilised the code from earlier notebook to create the tiling strategy 8*5 with corners removed = 36 exposures
    '''
    new_df = create_trans_sq(og_df, data)
    # new_df = create_trans_sq_customTest(og_df, data)
    final_df = remove_edges(og_df, new_df)

    return final_df


def gen_tiling_customTest(og_df, data):
    '''
    Utilised the code from earlier notebook to create the custom tiling strategy
    '''
    # new_df = create_trans_sq(og_df, data)
    new_df = create_trans_sq_customTest(og_df, data)
    final_df = remove_edges(og_df, new_df)

    return final_df


def visualize_healpy(new_df, mean_ra, mean_dec):
    '''
    Visualzing the region via healpy
    '''
    obs_rows = new_df.index # new_df
    #iterates through the list of indices for all of the rows that correspond to the SCAs in the new footprint and gets the list of pixels that the SCAs touch.
    ipix_box_list = []
    for index in obs_rows:
        vertices = get_vertices(new_df, index)
        ipix_box_single = hp.query_polygon(nside=NSIDE, vertices=vertices, inclusive=False)
        ipix_box_list.append(ipix_box_single)
    ipix_box = np.concatenate(ipix_box_list)
    # rot = [9.7, -44] 
    # mean_ra = new_df[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    # mean_dec = new_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()
    rot = [mean_ra, mean_dec]
    m = np.zeros(hp.nside2npix(NSIDE)) #these 3 rows create the map that gets fed into healpy
    counts = np.bincount(ipix_box, minlength=len(m))
    m[:len(counts)] = counts
    hp.gnomview(m, rot=rot,title="Sky Location of Roman Pictures-Z087", reso = 1.6, xsize = 200) #plots the map centered on "rot". reso is the number of arcminutes per pixel and xsize is the number of pixels.
    plt.show()

def visualize_astropy_gnomonic(df, mean_ra, mean_dec,
                               npix=200, reso_arcmin=1.6):
    """
    Visualize the corner polygons in a gnomonic (TAN) projection
    centered at (mean_ra, mean_dec). This mimics the style of 'gnomview'
    but uses Astropy WCS + Matplotlib instead of healpy.
    """

    # 1) Construct a gnomonic (TAN) WCS centered on (mean_ra, mean_dec).
    #    'cdelt' is degrees per pixel; reso_arcmin is arcminutes per pixel.
    deg_per_pix = reso_arcmin / 60.0

    w = WCS(naxis=2)
    w.wcs.crpix = [npix/2, npix/2]               # Reference pixel = image center
    w.wcs.cdelt = [-deg_per_pix, deg_per_pix]    # Pixel scale in deg/pix
    w.wcs.crval = [mean_ra, mean_dec]            # Reference RA/Dec in degrees
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]       # Gnomonic projection

    # 2) Create a Matplotlib figure with that WCS as the projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=w)

    # We set plotting limits to the full W x H in pixel coords
    ax.set_xlim(0, npix)
    ax.set_ylim(0, npix)

    # For each footprint (row), plot the corner polygon
    for idx, row in df.iterrows():
        # Extract the 4 corners for this row
        ra_corners = [row[f'RA{i}'] for i in range(1,5)]
        dec_corners = [row[f'DEC{i}'] for i in range(1,5)]

        # Close the polygon by repeating the first corner
        ra_corners.append(ra_corners[0])
        dec_corners.append(dec_corners[0])

        # Convert RA/Dec -> pixel coords using our WCS
        # The second argument to wcs_world2pix is "origin"; set to 1 means
        # 1-based pixel coordinates. Commonly we do 0, but let's do 1 for consistency with FITS.
        pix_coords = w.wcs_world2pix(np.column_stack((ra_corners, dec_corners)), 1)

        # Plot the polygon outline
        ax.plot(pix_coords[:,0], pix_coords[:,1], '-')

    # Final labeling
    ax.set_title(f"Sky Polygons (Gnomonic): center=({mean_ra:.2f}, {mean_dec:.2f})")
    ax.set_xlabel("X (pix)")
    ax.set_ylabel("Y (pix)")

    plt.show()


def visualize_astropy_plotcoord(df, mean_ra, mean_dec,
                                npix=200, reso_arcmin=1.6):
    """
    Using plot_coord with SkyCoord objects
    to visualize footprints in a gnomonic projection.
    """

    deg_per_pix = reso_arcmin / 60.0

    w = WCS(naxis=2)
    w.wcs.crpix = [npix/2, npix/2]
    w.wcs.cdelt = [-deg_per_pix, deg_per_pix]
    w.wcs.crval = [mean_ra, mean_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=w)
    ax.set_xlim(0, npix)
    ax.set_ylim(0, npix)

    for idx, row in df.iterrows():
        ra_corners = [row[f'RA{i}'] for i in range(1,5)]
        dec_corners = [row[f'DEC{i}'] for i in range(1,5)]

        # close the polygon
        ra_corners.append(ra_corners[0])
        dec_corners.append(dec_corners[0])

        # Make a SkyCoord for the polygon corners
        corners_coord = SkyCoord(ra=ra_corners*u.deg, dec=dec_corners*u.deg, frame='icrs')

        # ax.plot_coord can connect them with lines
        ax.plot_coord(corners_coord, '-')

    ax.set_title(f"Gnomonic: center=({mean_ra:.2f}, {mean_dec:.2f})")
    ax.set_xlabel("X (pix)")
    ax.set_ylabel("Y (pix)")
    plt.show()


def create_heat_map(df, ra_axis, dec_axis, total_degrees=365, step=5, rot="astropy"):
    """
    Creates a heat map showing pixel retention over a time period (rotations every step degrees).
    """
    total_steps = total_degrees // step + 1
    heat_map = np.zeros(hp.nside2npix(NSIDE))
    deg = (degree*360) / 365
    for degree in range(0, total_degrees + 1, step):
        # rotated_df = rotate_squares(df, degree)
        if rot == "astropy":
            
            rotated_df = rotate_squares_custom_astropy(df, deg, ra_axis, dec_axis)

        else:
            rotated_df = rotate_squares_custom_healpy(df, degree, ra_axis, dec_axis)

        for index in rotated_df.index:
            vertices = get_vertices(rotated_df, index)
            pixels = hp.query_polygon(NSIDE, vertices, inclusive=False)
            heat_map[pixels] += 1

    heat_map /= heat_map.max()  # Normalize to 0–1 range

    # Display the heat map using gnomview
    mean_ra = df[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    mean_dec = df[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()
    rot = [mean_ra, mean_dec]
    hp.gnomview(heat_map, rot=rot, title="Heat Map of Sky Coverage Over 1 Year", reso=1.6, xsize=400, cmap='viridis')
    plt.show()

    return heat_map


def calculate_pixel_efficiency(df, ra_axis, dec_axis, total_degrees=365, step=5, rot="astropy"):
    """
    Calculates pixel efficiency and records the degrees where each pixel appears.
    """
    total_steps = total_degrees // step + 1
    pixel_data = {}
    deg = (degree*360) / 365
    imaging_data = {}

    for degree in range(0, total_degrees + 1, step):
        if rot == "astropy":
            rotated_df = rotate_squares_custom_astropy(df, deg, ra_axis, dec_axis)

        else:
            rotated_df = rotate_squares_custom_healpy(df, degree, ra_axis, dec_axis)

        for index in rotated_df.index:
            cell = rotated_df.loc[index, "cell"]
            im_tier = rotated_df.loc[index, "imaging_tier"]
            if im_tier not in imaging_data:
                imaging_data[im_tier] = {}

            if degree not in imaging_data[im_tier]:
                imaging_data[im_tier][degree] = []

            vertices = get_vertices(rotated_df, index)
            pixels = hp.query_polygon(NSIDE, vertices, inclusive=False)

            for pixel in pixels:
                if pixel not in pixel_data:
                    pixel_data[pixel] = {"degrees": [], "degree_details": []}
                # pixel_data[pixel]["appearances"] += 1
                pixel_data[pixel]["degrees"].append(degree)
                
                dummy_dict = {}
                dummy_dict["cell"] = cell
                dummy_dict["degree"] = degree
                pixel_data[pixel]["degree_details"].append(dummy_dict)

                dummy_dict_2 = {}
                dummy_dict_2["pixel"] = pixel
                dummy_dict_2["cell"] = cell
                
                imaging_data[im_tier][degree].append(dummy_dict_2)

    # final efficiency calc
    for pixel, data in pixel_data.items():
        data["efficiency"] = len(data["degree_details"]) / total_steps
        data["appearances"] = len(data["degree_details"])

    return pixel_data, imaging_data


def hist_plot(pixel_details):

    efficiency_vals = [pix["efficiency"] for _, pix in pixel_details.items()]

    plt.hist(efficiency_vals, bins=10, edgecolor='black')
    plt.title('Efficiency Distribution')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')

    plt.show()


def cdf_calc(pixel_details):

    pixel_freqs = []
    for _, pix in pixel_details.items():
        pixel_freqs.append(pix['appearances'])

    d = {}

    d = Counter(pixel_freqs)
    sorted_d = dict(sorted(d.items(), key=lambda x: -x[0]))
    area_per_pixel = (0.014)**2
    y = []
    x = []
    tot = 0
    for k, v in sorted_d.items():
        tot += v*area_per_pixel
        y.append(tot)
        x.append(k)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o', s=20)  # s controls dot size
    plt.title("Cumulative Covered Area")
    plt.xlabel("Observations / year")
    plt.ylabel("Cumulative covered area (deg^2)")
    plt.gca().invert_xaxis()  # Reverse X-axis to match earlier plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def cdf_calc_unique(pixel_details):

    pixel_freqs = []
    for _, pix in pixel_details.items():
        pixel_freqs.append(len(set(pix['degrees'])))

    d = {}

    d = Counter(pixel_freqs)
    sorted_d = dict(sorted(d.items(), key=lambda x: -x[0]))
    area_per_pixel = (0.014)**2
    y = []
    x = []
    tot = 0
    for k, v in sorted_d.items():
        tot += v*area_per_pixel
        y.append(tot)
        x.append(k)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o', s=20)  # s controls dot size
    plt.title("Cumulative Covered Area")
    plt.xlabel("Observations / year")
    plt.ylabel("Cumulative covered area (deg^2)")
    plt.gca().invert_xaxis()  # Reverse X-axis to match earlier plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def translate_squares_custom_2x1(df, shift=0.01):

    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    ## Upward Tile (current pattern: 2x1)
    translation_distance_up = max_dec - min_dec - shift

    result_df = df.copy()

    translated_df = single_pointing.copy()
    translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up

    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    return result_df

def translate_squares_custom_3x2(df, shift=0.01):
    """
    Function to create the tiling of the footprint (3x2), following the snake pattern outlined in (Wang et al., 2023).
    Output: Original dataframe with the rows corresponding to the new pointings appended to the end.

    """

    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    ## Upward Tile (current pattern: 2x1)
    translation_distance_up = max_dec - min_dec - shift

    result_df = df.copy()

    translated_df = single_pointing.copy()
    translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up

    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    ## Upward Middle Tile (current pattern: 3x1)    
    last_set = result_df.tail(18)

    mean_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    max_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].max().max()
    
    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()
    mean_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()

    translation_distance_left = max_ra - mean_ra - shift
    translation_distance_up = max_dec - mean_dec  + 15*shift

    last_set_translated = last_set.copy()
    last_set_translated[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up

    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)

    ##Downward Tile (current pattern: 3x2 with 1 row remaining)
    last_set = result_df.tail(18)

    mean_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    max_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].max().max()
    
    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()
    mean_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()

    translation_distance_left = max_ra - mean_ra - shift
    translation_distance_down = max_dec - mean_dec + 15*shift

    last_set_translated = last_set.copy()
    last_set_translated[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_down

    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)

    ##Downward Tile (current pattern: 3x2 complete)
    last_set = result_df.tail(18)
    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    translation_distance_down = max_dec - min_dec - shift
    last_set_translated = last_set.copy()
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_down
    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)
    
    return result_df


def translate_squares_custom_2x1(df, shift=0.01):

    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    cell = 0
    ## Upward Tile (current pattern: 2x1)
    translation_distance_up = max_dec - min_dec - 0.09

    result_df = df.copy()
    result_df["cell"] = cell
    cell += 1
    translated_df = single_pointing.copy()
    translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
    translated_df["cell"] = cell
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    return result_df

def translate_squares_custom_7(df, shift=0.09):
    """
    Function to create the tiling of the footprint (3x2), following the snake pattern outlined in (Wang et al., 2023).
    Output: Original dataframe with the rows corresponding to the new pointings appended to the end.

    """
    cell = 0
    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    ## Upward Tile (current pattern: 2x1)
    translation_distance_up = max_dec - min_dec - shift

    result_df = df.copy()
    result_df["cell"] = cell
    cell += 1
    
    translated_df = single_pointing.copy()
    translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
    
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    ## Upward Tile (current pattern: 2x1)
    single_pointing = result_df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()
    translation_distance_up = max_dec - min_dec - shift

    # result_df = df.copy()

    translated_df = single_pointing.copy()
    translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
    
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    ## Upward Middle Tile (current pattern: 3x1)
    last_set = result_df.tail(18)

    mean_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    max_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].max().max()

    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()
    mean_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()

    translation_distance_left = max_ra - mean_ra
    translation_distance_up = max_dec - mean_dec  + 1*shift

    last_set_translated = last_set.copy()
    last_set_translated[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
    
    last_set_translated["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)

    ##Downward Tile (current pattern: 3x2 with 1 row remaining)
    last_set = result_df.tail(18)

    mean_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].mean().mean()
    max_ra = last_set[['RA1', 'RA2', 'RA3', 'RA4']].max().max()

    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()
    mean_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()

    translation_distance_left = max_ra - mean_ra
    translation_distance_down = max_dec - mean_dec + 1*shift

    last_set_translated = last_set.copy()
    last_set_translated[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_down
    
    last_set_translated["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)

    ##Downward Tile (current pattern: 3x2 complete)
    last_set = result_df.tail(18)
    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    translation_distance_down = max_dec - min_dec - shift
    last_set_translated = last_set.copy()
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_down
    
    last_set_translated["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)

    ##Downward Tile (current pattern: 3x2 complete)
    last_set = result_df.tail(18)
    min_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = last_set[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    translation_distance_down = max_dec - min_dec - shift
    last_set_translated = last_set.copy()
    last_set_translated[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_down
    
    last_set_translated["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, last_set_translated], ignore_index=True)
    result_df["imaging_tier"] = "deep"
    return result_df


def translate_squares_custom_9x5_38(df, shift=0.01, cell=7):

    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    min_ra = single_pointing[['RA1', 'RA2', 'RA3', 'RA4']].min().min()
    max_ra = single_pointing[['RA1', 'RA2', 'RA3', 'RA4']].max().max()

    cell = 7 
    mean_dec = df[['DEC1', 'DEC2', 'DEC3', 'DEC4']].mean().mean()

    translation_distance_up = max_dec - min_dec - 0.09
    translation_distance_left = max_ra - min_ra + 0.01

    result_df = single_pointing.copy()
    result_df["cell"] = cell
    cell += 1
    result_df["remove"] = 1

    single_pointing = result_df.tail(18)
    for i in range(8):
        if i == 7:
            single_pointing["remove"] = 1
        else:
            single_pointing["remove"] = 0
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
        translated_df["cell"] = cell
        cell += 1
        result_df = pd.concat([result_df, translated_df], ignore_index=True)
        single_pointing = result_df.tail(18)

    single_pointing = result_df.tail(18)
    single_pointing["remove"] = 0
    translated_df = single_pointing.copy()
    translated_df[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    single_pointing = result_df.tail(18)
    for i in range(8):
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_up
        translated_df["cell"] = cell
        cell += 1
        result_df = pd.concat([result_df, translated_df], ignore_index=True)
        single_pointing = result_df.tail(18)

    single_pointing = result_df.tail(18)
    single_pointing["remove"] = 0
    translated_df = single_pointing.copy()
    translated_df[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    single_pointing = result_df.tail(18)
    for i in range(8):
        if i in [2,3,4]:
            single_pointing["remove"] = 1
        else:
            single_pointing["remove"] = 0
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
        translated_df["cell"] = cell
        cell += 1
        result_df = pd.concat([result_df, translated_df], ignore_index=True)
        single_pointing = result_df.tail(18)

    single_pointing = result_df.tail(18)
    single_pointing["remove"] = 0
    translated_df = single_pointing.copy()
    translated_df[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    single_pointing = result_df.tail(18)
    for i in range(8):
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] -= translation_distance_up
        translated_df["cell"] = cell
        cell += 1
        result_df = pd.concat([result_df, translated_df], ignore_index=True)
        single_pointing = result_df.tail(18)

    single_pointing = result_df.tail(18)
    single_pointing["remove"] = 1
    translated_df = single_pointing.copy()
    translated_df[['RA1', 'RA2', 'RA3', 'RA4']] += translation_distance_left
    translated_df["cell"] = cell
    cell += 1
    result_df = pd.concat([result_df, translated_df], ignore_index=True)

    single_pointing = result_df.tail(18)
    for i in range(8):
        if i == 7:
            single_pointing["remove"] = 1
        else:
            single_pointing["remove"] = 0
        translated_df = single_pointing.copy()
        translated_df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += translation_distance_up
        translated_df["cell"] = cell
        cell += 1
        result_df = pd.concat([result_df, translated_df], ignore_index=True)
        single_pointing = result_df.tail(18)

    result_df = result_df[result_df['remove'] == 0]
    result_df["imaging_tier"] = "wide"
    return result_df


def shift_centers(df, r1, d1):

    single_pointing = df.tail(18)
    min_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].min().min()
    max_dec = single_pointing[['DEC1', 'DEC2', 'DEC3', 'DEC4']].max().max()

    min_ra = single_pointing[['RA1', 'RA2', 'RA3', 'RA4']].min().min()
    max_ra = single_pointing[['RA1', 'RA2', 'RA3', 'RA4']].max().max()

    ra_cen = df[[f'RA{i}' for i in range(1, 5)]].mean().mean()
    dec_cen = df[[f'DEC{i}' for i in range(1, 5)]].mean().mean()

    translation_distance_up = max_dec - min_dec - 0.09
    translation_distance_left = max_ra - min_ra + 0.01

    df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += (d1 - dec_cen) #2.0*translation_distance_up
    df[['RA1', 'RA2', 'RA3', 'RA4']] += (r1 - ra_cen)#1.5*translation_distance_left
    # df['cell'] = 0
    df['remove'] = 0

    return df

###OLD - CAN IGNORE###
def extract_sim_data_old(exposures):

    exp_sorted = dict(sorted(exposures.items(), key=lambda x: x[0]))

    lib_count = 0
    final_res = ""

    header = "MJD\tID\tFLT"
    for name, d in exp_sorted.items():

        final_res += f"LIBGEN: {lib_count}; name: {name}\n"
        final_res += f"{header}\n"
        final_res += d['text'] + "\n\n"
        lib_count += 1

    return final_res

###OLD - CAN IGNORE###
def create_sim_file_old(df, ra_axis, dec_axis, mjd_st, total_degrees=365, step=5, rot="astropy"):
    """
    Calculates pixel efficiency and records the degrees where each pixel appears.
    """
    total_steps = total_degrees // step + 1
    exposures = {}
    mjd = mjd_st
    for degree in range(0, total_degrees + 1, step):
        mjd += degree

        if rot == "astropy":
            rotated_df = rotate_squares_custom_astropy(df, degree, ra_axis, dec_axis)

        else:
            rotated_df = rotate_squares_custom_healpy(df, degree, ra_axis, dec_axis)

        for index in rotated_df.index:
            cell = rotated_df.loc[index, "cell"]
            im_tier = rotated_df.loc[index, "imaging_tier"]
            sca = rotated_df.loc[index, "SCA"]

            name = f"{im_tier}_{cell}_{sca}"
            if name not in exposures:
                exposures[name] = {'text': '', 'counter': 1}

            ##0-wide: RZJ - repeat 10degs
            ##5-wide: RYH - repeat 10degs
            ##0-deep: ZYH - repeat 10degs
            ##5-deep: ZJF - repeat 10degs

            if im_tier == 'wide':
                
                if degree % 10 == 0:
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tR\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tZ\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tJ\n"
                    exposures[name]['counter'] += 1

                else:
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tR\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tY\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tH\n"
                    exposures[name]['counter'] += 1

            else:
                
                if degree % 10 == 0:
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tZ\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tY\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tH\n"
                    exposures[name]['counter'] += 1

                else:
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tZ\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tJ\n"
                    exposures[name]['counter'] += 1
                    exposures[name]['text'] += f"{mjd}\t{exposures[name]['counter']}\tF\n"
                    exposures[name]['counter'] += 1

    res = extract_sim_data_old(exposures)
            

    return res


def get_ra_dec(pixel_index):
    theta, phi = hp.pix2ang(NSIDE, pixel_index)
    ra_deg = np.degrees(phi)          # RA = longitude in degrees
    dec_deg = 90.0 - np.degrees(theta)  # Dec = 90° - colatitude in degrees
    return ra_deg, dec_deg

def get_roman_hltd_noise_data(f_path='roman_hltds_noise.yml'):

    with open(f_path) as f:
        config = yaml.safe_load(f)

    custom_dict = {}
    filt_map = {"F062": "R", "F087": "Z", "F106": "Y", "F129": "J", "F158": "H", "F184": "F"}
    for filt, props in config['noise'].items():
        exp = props.get('exp', {})

        # img_dict = {}
        details_dict = {}
        details_dict['nea'] = props['nea']
        details_dict['zodi'] = props['zodi']
        details_dict['thermal'] = props['thermal']
        for img, dct in exp.items():
            img_dict = {}
            img_dict['time'] = dct['time']
            img_dict['read_noise'] = dct['read_noise']
            img_dict['sky_sigma'] = dct['sky_sigma']
            img_dict['snana_zp'] = dct['snana_zp']
            details_dict[img] = img_dict

        custom_dict[filt_map[filt]] = details_dict

    return custom_dict

def initialize_dustmaps(custom_dir='./dustmaps_data'):

    config['data_dir'] = custom_dir
    sfd_folder = os.path.join(config['data_dir'], 'sfd')
    if not os.path.isdir(sfd_folder):
        print(f"Downloading SFD maps into {sfd_folder} …")
        fetch()
    else:
        print(f"SFD maps already present in {sfd_folder}, skipping fetch.")

    sfd = SFDQuery()
    return sfd

def format_row(values, widths, cols):
    return "  ".join(f"{str(v):<{widths[col]}}" 
                     for col, v in zip(cols, values)) + "\n"

def documentation_text():

    doc = "DOCUMENTATION:\n"
    doc += "     PURPOSE: CCS SIMULATION\n"
    doc += "     USAGE_KEY: SIMLIB_FILE\n"
    doc += "     USAGE_CODE: snlc_sim.exe\n"
    doc += "DOCUMENTATION_END:\n"
    ##can add more details here

    return doc

def create_sim_file(pixel_data):
    
    # SURVEY:  ROMAN
    # FILTERS: RZYJHFK
    # PIXSIZE: 0.11 # arcsec
    # PSF_UNIT: NEA_PIXEL # Noise-equiv-Area instead of PSF
    # SOLID_ANGLE: 0.0077  # (sr) sum of all tiers
    # TEXPOSE(WIDE):   72  84  305  95 1280   0   0
    # TEXPOSE(DEEP):   0  72  231  366  266 1333   0
    # BEGIN LIBGEN
    lib_count = 0
    final_res = documentation_text()
    roman_yml_data = get_roman_hltd_noise_data()
    sfd = initialize_dustmaps()
    
    solid_angle = hp.nside2pixarea(NSIDE, degrees = True)*len(pixel_data)
    filt_ord = "RZYJHF"
    final_res += f"SURVEY: ROMAN\nFILTERS: RZYJHF\nPIXSIZE: 0.109\nPSF_UNIT: NEA_PIXEL\nSOLID_ANGLE: {solid_angle:.3f}\n"

    text_time_wide = "TEXPOSE(WIDE): "
    text_time_deep = "TEXPOSE(DEEP): "
    text_zodi = "ZODI: "
    text_thermal = "THERMAL: "

    arr_tw = []
    arr_td = []
    arr_z = []
    arr_th = []
    for filt in filt_ord:
        time_wide = roman_yml_data[filt]['wide']['time'] if 'wide' in roman_yml_data[filt] else 0
        time_deep = roman_yml_data[filt]['deep']['time'] if 'deep' in roman_yml_data[filt] else 0
        zodi = roman_yml_data[filt]['zodi']
        thermal = roman_yml_data[filt]['thermal']

        arr_tw.append(time_wide)
        arr_td.append(time_deep)
        arr_z.append(zodi)
        arr_th.append(thermal)

    text_time_wide += "  ".join(str(x) for x in arr_tw)
    text_time_deep += "  ".join(str(x) for x in arr_td)
    text_zodi += "  ".join(str(x) for x in arr_z) 
    text_thermal += "  ".join(str(x) for x in arr_th)
   
    final_res += text_time_wide + "\n" + text_time_deep + "\n" + text_zodi + "\n" + text_thermal + "\n"
    final_res += "BEGIN LIBGEN\n"
    # COL_WIDTH = 10
    cols = ["#", "MJD", "IDEXPT", "BAND", "GAIN", "NOISE", "SKYSIG", "NEA", "ZPTAVG", "ZPTERR","MAG"]
    widths = {c: len(c) for c in cols}     
    widths["#"] = 2
    widths["MJD"] = 5
    widths["NEA"] = 6
    widths["NOISE"] = 6
    # header = "".join(f"{c:<{COL_WIDTH}}" for c in cols) + "\n"
    header = format_row(cols, widths, cols)
    
    lst = list(range(1, len(pixel_data)+1))
    rnd = random.Random(123)
    rnd.shuffle(lst)

    for pix, dct in pixel_data.items():
        ra, dec = dct['ra'], dct['dec']
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        ebv_val = sfd(coord)
        final_res += f"\nLIBID: {lst[lib_count]}\n"
        # emp_sp = "     "
        # final_res += f"RA: {ra:.3f}{emp_sp}DECL: {dec:.3f}{emp_sp}NOBS: {3*len(dct['data'])}{emp_sp}MWEBV: {ebv_val:.3f}{emp_sp}PIXSIZE: 0.109\n"
        # final_res += (
        #         f"RA:    {ra:7.3f}  "
        #         f"DECL:  {dec:8.3f}  "
        #     )
        final_res += f"RA: {ra:.3f} DEC: {dec:.3f}\n"
        final_res += f"NOBS: {3*len(dct['data'])}\n"
        final_res += f"HEALPIX: {pix} MWEBV: {ebv_val:6.3f}\n"
        final_res += f"{header}"

        for d in dct['data']:
            filters = d['filter']
            im_tier = d['imaging_tier']
            
            for filt in filters:
                gain = 1
                noise = roman_yml_data[filt][im_tier]['read_noise']
                skysig = roman_yml_data[filt][im_tier]['sky_sigma'] 
                zptavg = roman_yml_data[filt][im_tier]['snana_zp']
                zperr = 0.005
                mag = 99
                nea = roman_yml_data[filt]['nea']
    
                row_vals = ["S:", d['mjd'], d['id'], filt, gain, noise, skysig, nea, zptavg, zperr, mag]
                
                # row = "".join(f"{str(v):<{COL_WIDTH}}" for v in row_vals) + "\n"
                # final_res += row
                final_res += format_row(row_vals, widths, cols)

        final_res += f"END_LIBID: {lst[lib_count]}\n"
        # break
        lib_count += 1

    final_res += "END LIBGEN\n"
    return final_res



def sim_file_calc(df, ra_axis, dec_axis, current_mjd, total_degrees=365, step=5, rot="astropy"):
    """
    Calculates pixel efficiency and records the degrees where each pixel appears.
    """
    ##0-wide: RZJ - repeat 10degs
    ##5-wide: RYH - repeat 10degs
    ##0-deep: ZYH - repeat 10degs
    ##5-deep: ZJF - repeat 10degs

    total_steps = total_degrees // step + 1
    exposures = {}
    mjd = int(current_mjd)
    ids = {}
    pixel_data = {}
    
    for degree in range(0, total_degrees + 1, step):
        
        deg = ((degree % 365)*360) / 365
        if rot == "astropy":
            rotated_df = rotate_squares_custom_astropy(df, deg, ra_axis, dec_axis)

        else:
            rotated_df = rotate_squares_custom_healpy(df, degree, ra_axis, dec_axis)


        for index in rotated_df.index:
            cell = rotated_df.loc[index, "cell"]
            _id = f"{degree}_{cell}"

            if _id not in ids:
                ids[_id] = len(ids) + 1

            id_val = ids[_id]
            im_tier = rotated_df.loc[index, "imaging_tier"]
            
            vertices = get_vertices(rotated_df, index)
            pixels = hp.query_polygon(NSIDE, vertices, inclusive=False)

            for pix in pixels:
                if pix not in pixel_data:
                    pixel_data[pix] = {'data': []}
                    ra, dec = get_ra_dec(pix)
                    pixel_data[pix]['ra'] = ra
                    pixel_data[pix]['dec'] = dec
                    
                dummy_dict = {}
                dummy_dict['id'] = id_val
                dummy_dict['mjd'] = mjd
                dummy_dict['imaging_tier'] = im_tier

                if degree % 10 == 0 and im_tier == 'wide':
                    dummy_dict['filter'] = 'RZJ'

                elif degree % 10 == 0 and im_tier == 'deep':
                    dummy_dict['filter'] = 'ZYH'

                elif degree % 10 == 5 and im_tier == "wide":
                    dummy_dict['filter'] = 'RYH'

                elif degree % 10 == 5 and im_tier == "deep":
                    dummy_dict['filter'] = 'ZJF'

                pixel_data[pix]['data'].append(dummy_dict)

        mjd += step

    sim_file = create_sim_file(pixel_data)
    return sim_file, pixel_data


def recenter_exposure(df, r1 = 242.504, d1 = 54.510):

    ra_cen = df[[f'RA{i}' for i in range(1, 5)]].mean().mean()
    dec_cen = df[[f'DEC{i}' for i in range(1, 5)]].mean().mean()

    df[['DEC1', 'DEC2', 'DEC3', 'DEC4']] += (d1 - dec_cen) #2.0*translation_distance_up
    df[['RA1', 'RA2', 'RA3', 'RA4']] += (r1 - ra_cen)#1.5*translation_distance_left

    return df
