import cv2 as cv
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from nd2reader import ND2Reader
from scipy.ndimage.measurements import histogram
from skimage.filters import threshold_otsu
from skimage import measure, io
from skimage.measure import label
from skimage.segmentation import watershed, flood_fill
from aicsimageio import AICSImage


def find_cells(im, thresh):
    """
    Given a phase image of bacterial cells, finds where the cells are.
    Args:
        im (2-D numpy array): Phase image of cells
        thresh (float, optional): how much darker the cells are than the background 
        (theoretically exposure invariant)
    Returns:
        mask (2-D numpy array): labelled mask of the cells 
    """
    # Background subtracts and inverts the image so that darker regions (i.e. the cells
    # are brighter). Then finds where the cells are by finding the brightest peaks
    back_sub = -1 * im + np.median(im)
    foreground = back_sub > thresh * 255
    background = (back_sub < thresh * 255)

    background = background.astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    background = cv.erode(background, kernel)
    cells = measure.label(foreground)

    # removes anything that is smaller than 125 pixels
    hist_cells = histogram(cells, 0, np.max(cells), np.max(cells))
    small_cells = np.where(hist_cells < 125)[0]
    i_cells = np.isin(cells, small_cells)
    cells[i_cells] = 0
    markers = foreground * (cells + 1) + background

    # Finds the edges using a Scharr filter and smooths with a Gaussian filter
    edges_x = cv.Scharr(im, cv.CV_64F, 1, 0)
    edges_y = cv.Scharr(im, cv.CV_64F, 0, 1)
    edges = np.uint16(np.sqrt(np.square(edges_x) + np.square(edges_y)))
    edges = cv.GaussianBlur(edges, (3, 3), 0)

    markers = watershed(edges, markers)
    mask = markers != 1

    hist_mask = histogram(markers, 0.5, np.max(markers) + 0.5, np.max(markers))
    small_cells = np.where(hist_mask < 125)[0]
    if np.size(small_cells) > 0:
        i_cells = np.isin(markers, small_cells + 1)
        mask[i_cells] = 0

    return mask


def save_tiff_stack(images, path, filename, binary=True):
    """
    Given an image stack, will save as a tiff stack
    parameters:
        images (x, y, t) ndarray: array of images to be saved as a tiff stack
        path (string): path where tiff stack will be saved
        filename (string):filename
        binary (bool): whether the image sure be saved as a binary image
    returns:
        empty
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    im_list = []
    if binary:
        images = images.astype(np.bool)
        for im in images.T:
            size = im.shape[::-1]
            data_bytes = np.packbits(im, axis=1)
            im_list.append(Image.frombytes(mode='1', size=size, data=data_bytes))

    else:
        images = images.astype(np.uint8)
        for im in images.T:
            im_list.append(Image.fromarray(im))
    im_list[0].save(f'{path}/{filename}', save_all=True,
                    append_images=im_list[1:])


class Movie:
    def __init__(self, filename, ref_name, bkg_name):
        self.filename = filename
        self.dir = '/'.join(filename.split('/')[:-1]) + '/'
        self.ref_path = ref_name
        self.bkg_path = bkg_name
        if '.nd2' in filename:
            images = AICSImage(filename)
            try:
                p_ms = images.metadata['experiment'][0].parameters.periodMs
            except AttributeError:
                p_ms = images.metadata['experiment'][0].parameters.periods[0].periodMs

            self.t_delta = int(np.round(p_ms / (1000 * 60),
                                        decimals=0))
            self.FOV = len(images.scenes)
            self.t_total = images.dims.T
            self.height = images.dims.Y
            self.width = images.dims.X

    def generate_masks(self, thresh=0.05, sigma=1):
        """
        Given an ND2Reader, will create and save masks for all FOVs and
        frames and save the masks as tiff stacks in a given directory. 
        """
        # makes a dir to hold all masks
        if not os.path.isdir(self.dir + 'masks/'):
            os.makedirs(self.dir + 'masks/')

        with ND2Reader(self.filename) as images:
            if len(images.sizes) == 5 or len(images.sizes) == 6:
                images.iter_axes = 'vtz'
            elif self.FOV != 1:
                images.iter_axes = 'vt'
            else:
                images.iter_axes = 't'

            # Iterates over the different fields of view
            for f in range(self.FOV):
                print(self.dir + 'masks/FOV_%d' % (f + 1))

                # initializes a mask for the given field of view
                mask = np.zeros((self.height, self.width, self.t_total))

                # iterates over the different times
                for t in range(self.t_total):
                    print(t, " ", end="")
                    if len(images.sizes) == 4:
                        im = images[self.t_total * f + t].astype(np.uint16)
                    elif len(images.sizes) == 5 or len(images.sizes) == 6:
                        im = images[self.FOV * t + f].astype(np.uint16)
                    # smooths the image with a Gaussian Blur
                    smooth_im = cv.GaussianBlur(im, (sigma, sigma), 0)
                    cv.normalize(smooth_im, smooth_im, 0, 255, cv.NORM_MINMAX)

                    # calculates a binary mask of the cells and saves it
                    cells = find_cells(smooth_im, thresh=thresh).astype(np.uint8)
                    mask[:, :, t] = cells
                save_tiff_stack(mask, self.dir + 'masks/', 'FOV_%d.tif' % (f + 1))
        param = {"thresh": thresh,
                 "sigma": sigma}
        filename = self.dir + 'segmentation_parameters'
        file_object = open(filename, 'wb')
        pickle.dump(param, file_object)

        # TODO: save the threshold value and sigma in a human readable file format

    def get_area(self, mask, fov):
        """
        For a mask, generates a pandas dataframe of the area of each object (in pixels) over time
        parameters:
            mask (x, y, t) ndarray: boolean mask of cells
            fov (int): which fov the masks correspond to, used in labels in the resulting dataframe and pathname of
            labeled masks
        returns:
            cell_size pandas dataframe: dataframe where each column represents an object in the mask and each row
            represents the size of that object (in pixels) over time
        """

        mask = mask.astype(np.uint16)

        # labels the cells in the last mask and then removes anything that is on the edge
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        cells_dilated = cv.dilate(mask, kernel)
        cells_labeled = np.multiply(label(cells_dilated[:, :, -1]), cells_dilated[:, :, -1])

        # finds the labels which are on the edge
        label_edge = np.unique(np.concatenate((cells_labeled[:, :1],
                                               cells_labeled[:, -2:],
                                               cells_labeled[:1, :],
                                               cells_labeled[-2:, :]),
                                              axis=None))
        # if there are cells on the edge, remove them from the labeled cells
        if np.size(label_edge > 1):
            i_edge = np.isin(cells_labeled, label_edge)
            cells_labeled[i_edge] = 0

        t_total = np.shape(mask)[2]
        cell_id = np.unique(cells_labeled)[1:]

        if np.size(cell_id) > 0:
            mask_labeled = np.zeros(np.shape(mask) + (np.max(cell_id),))
        else:
            mask_labeled = np.zeros(np.shape(mask) + (1,))

        # loops through every unique object in the last frame and propagates those identifiers
        # to every other frame based on overlap
        for c in cell_id:
            cell = np.where(cells_labeled == c)
            seed_point = (cell[0][0], cell[1][0], t_total - 1)
            cells_dilated = flood_fill(cells_dilated, seed_point, c + 1)

        # reduces indices of cells by 1 (otherwise lowest possible theoretical index is 2)
        cells_dilated = cells_dilated - cv.dilate(mask, kernel)

        # converts 3-D array where each cell has unique label into 4-D array where each cell has unique index in 4th dim
        for c in cell_id:
            mask_labeled[:, :, :, c - 1] = cells_dilated == c
        cell_id = np.unique(cells_dilated)[1:]

        # because we labeled a dilated version of the mask, applies the labels to the initial pre-dilated mask
        mask_labeled = np.multiply(mask[:, :, :, np.newaxis], mask_labeled)

        # sums up total number of labeled pixels for each cell per time point and stores it in dataframe
        idx = pd.Index(np.arange(self.t_total) * self.t_delta)
        clm = pd.MultiIndex.from_product([[f'FOV_{fov}'], cell_id],
                                         names=['FOV', 'cell_id'])
        # removes unused cell_id in case cells ended up merging
        cell_size = pd.DataFrame(data=np.sum(mask_labeled[:, :, :, cell_id - 1], axis=(0, 1)),
                                 index=idx,
                                 columns=clm)
        cell_size.index.name = 'Time (min)'
        cell_size.columns.name = 'Cell number'

        # removes cells that are not found in more than 20% of frames
        cell_size = cell_size.replace(0, np.nan)
        cell_size = cell_size.dropna(axis=1, thresh=cell_size.shape[0] * 0.2)

        # removes any object that is not always larger than 50 pixels
        cell_size = cell_size[cell_size > 50].dropna(axis=1)

        # finds the index of the cells that haven't been filtered out and saves the masks of those cells
        true_cells = cell_size.columns.get_level_values(1).values
        mask_labeled = mask_labeled[:, :, :, true_cells - 1] * true_cells
        mask_labeled = np.sum(mask_labeled, axis=3)
        save_tiff_stack(mask_labeled, f'{self.dir}/labeled_masks', f'FOV_{fov}.tif', binary=False)
        # print('labeled mask saved!')
        return cell_size

    def save_area(self):
        """
        For every field of view, finds the area of each object for every time point, and saves the data as a dataframe
        """
        idx = pd.Index(np.arange(self.t_total) * self.t_delta)
        area_all = pd.DataFrame(index=idx)
        for i in range(self.FOV):
            print(f'Loading FOV_{i + 1}.tif')
            if not os.path.exists(f'{self.dir}/masks/FOV_{i + 1}.tif'):
                self.generate_masks()
            mask = io.imread(f'{self.dir}/masks/FOV_{i + 1}.tif').T

            area_fov = self.get_area(mask, i + 1)
            print(f'Area for FOV_{i + 1} calculated')
            area_all = pd.concat([area_all, area_fov], axis=1)
        area_all.columns = pd.MultiIndex.from_tuples(area_all.columns.values)
        area_all.to_pickle(self.dir + 'cell_areas.pkl')

    def calc_gr(self, lag=15):
        """
        Calculates the growth rate of every object in the movie, assuming exponential growth, ignoring the first few
        frames. Saves the data as a dataframe
        :param lag (int): the number of initial frames to ignore when calculating growth rate
        :return: None
        """
        cell_area = pd.read_pickle(self.dir + 'cell_areas.pkl').dropna()
        growth_rate = []
        # growth_rate = pd.DataFrame(columns = ['td', 'r', 'FOV', 'object'])
        for c in cell_area.columns:
            area = cell_area[c]
            time = area.index / 60
            log_area = np.log2(area.values)
            fit = np.polyfit(x=time[lag:],
                             y=log_area[lag:],
                             deg=1,
                             # w=area[lag:],
                             full=True)
            r2 = 1 - fit[1][0] / np.sum(np.square(log_area[lag:] - log_area[lag:].mean()))

            growth_rate.append({'td': 60 / fit[0][0],
                                'r': np.sqrt(r2),
                                'FOV': c[0],
                                'object': c[1],
                                'index': c})
        gr = pd.DataFrame(growth_rate)
        gr.to_pickle(self.dir + 'growth_rate.pkl')

    def find_fluorescence(self):
        """
        Applies the masks generated from the phase data to the fluorescent channel and finds the total fluorescent
        of each object in every field of view for every time point, accounting for flat-field correction. Saves the
        data as a dataframe.
        :return: None
        """
        with ND2Reader(self.bkg_path) as images:
            i_bkg = np.mean(images, axis=(0))
        with ND2Reader(self.ref_path) as images:
            i_ref = np.mean(images, axis=0)

        if not os.path.exists(f'{self.dir}/cell_areas.pkl'):
            self.save_area()

        areas = pd.read_pickle(f'{self.dir}/cell_areas.pkl')

        areas = areas.dropna(axis=0, how='all')
        fluorescence = pd.DataFrame(index=areas.index,
                                    columns=areas.columns)
        green = ND2Reader(self.filename)
        green.default_coords['c'] = 1
        green.iter_axes = 'vtz'
        t_total = green.sizes['t']

        # creates a binary image that reflects where the FOV is poorly illuminated based
        # on reference image and also the pixels that are close to the edge
        edge = i_ref < threshold_otsu(i_ref) * 0.8
        edge[:, :10] = True
        edge[:, -10:] = True
        edge[:10, :] = True
        edge[-10:, :] = True

        # calculates flat-field correction based on the reference and background image
        norm = np.sum((i_ref - i_bkg) * ~edge) / np.sum(~edge)

        for f in range(self.FOV):
            idx = f'FOV_{f + 1}'
            print(idx)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            if idx in fluorescence.columns.get_level_values(0):
                labeled_mask = io.imread(f'{self.dir}/labeled_masks/FOV_{f + 1}.tif').T
                cells = areas[idx].columns.values

                # iterates through the different individual cells as identified from the phase image
                # TODO: vectorize this operation rather than relying on for loops
                for c in cells:
                    mask = labeled_mask[:, :, :] == c
                    mask = cv.dilate(mask.astype(np.uint8), kernel)

                    # only considers cells that are not close to the edge are well illuminated
                    if np.sum(mask * edge[:, :, np.newaxis]) == 0:
                        raw_fluor = np.array(green[self.FOV * np.arange(t_total) + f])
                        raw_fluor = np.moveaxis(raw_fluor, 0, 2)

                        # applies flat-field correction to the raw fluorescence image
                        corrected_fluor = (raw_fluor - i_bkg[:, :, np.newaxis]) / \
                                          (i_ref - i_bkg)[:, :, np.newaxis] * norm

                        # calculates the total fluorescence of the cell after flat-field correction
                        total_fluor = np.sum(mask * corrected_fluor, axis=(0, 1))
                        fluorescence.loc[:, (idx, c)] = total_fluor
        fluorescence.to_pickle(f'{self.dir}/fluorescence.pkl')

    def calc_expression(self, r_thresh=0.99):
        """
        Calculates the median expression for each object in each field of view during a timelapse movie from the
        fluorescence and growth rate of that object, accounting for the fact that not all fluorescent proteins will be
        fully mature and that in balanced growth, the fraction of mature fluorescent proteins is dependent on the
        maturation kinetics and growth rate of the cell. Saves the expression of each object as a dataframe.

        See https://www.nature.com/articles/nmeth.4509 for more details

        :param r_thresh: (float) minimum correlation coefficient for measured change in area over time to an
        exponential fit that will be considered
        :return: None
        """
        if not os.path.exists(f'{self.dir}/growth_rate.pkl'):
            self.calc_gr()
        gr = pd.read_pickle(f'{self.dir}/growth_rate.pkl')

        gr = gr.set_index(keys='index').copy()
        gr = gr[gr.r > r_thresh]

        if not os.path.exists(f'{self.dir}/fluorescence.pkl'):
            self.find_fluorescence()
        fluorescence = pd.read_pickle(f'{self.dir}/fluorescence.pkl')

        area = pd.read_pickle(f'{self.dir}/cell_areas.pkl').dropna(axis=0)
        norm_fluorescence = fluorescence / area

        gr['normalized fluorescence'] = norm_fluorescence[gr.index].median()
        gr['initial fluorescence'] = norm_fluorescence.loc[0, gr.index].astype(float)

        gr = gr.reset_index().copy()

        tf = 13.6  # kinetics of folding for sf-GFP
        gr['f_mat'] = 1 / (1 + tf / gr['td'])
        gr['expression'] = gr['normalized fluorescence'] / gr['f_mat']
        gr['initial expression'] = gr['initial fluorescence'] / gr['f_mat']
        gr['doublings per hour'] = 60 / gr['td']

        gr.to_pickle(f'{self.dir}/expression_data.pkl')

# %%
