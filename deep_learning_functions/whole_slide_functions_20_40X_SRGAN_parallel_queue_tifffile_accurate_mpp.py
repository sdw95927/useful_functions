"""
Output features are under 40X. 

Works for any magnitude images:
    40X: directly apply maskRCNN
    20X: SRGAN to upscale to 40X, than apply maskRCNN
    
default_magnitude: deprecated; use mpp instead
"""

import os
import math
import cv2
import skimage
import skimage.measure
import sys
import pickle
import numpy as np
import pandas as pd

# import self-defined image processing functions
from segmentation_functions import *

# pytorch
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

# parallel_queue
from parallel_queue import QueueChain

RGB_MARKERS = {
    'BG': [255, 255, 255],  # Background
    'Tumor': [0, 255, 0],  # Tumor
    'Stroma': [255, 0, 0],  # Stroma
    'Lymphocyte': [0, 0, 255],  # Lymphocyte
    'Blood': [255, 0, 255],  # Blood cell
    'Macrophage': [255, 255, 0],  # Macrophage
    'Karyorrhexis': [0, 148, 255]  # Dead nuclei
}

CLASSES = ['BG', 'Tumor', 'Stroma', 'Lymphocyte', 'Blood', 'Macrophage', 'Karyorrhexis']

MEAN = [149.91262888, 125.60738922, 155.41683322]
STD = [64.47969016, 65.98274776, 53.71798459]

DESIRED_LEVEL = 0  # open-slide read_region level
# This setting is for 20X; times 2 for 40X
STEP_SIZE = 226
PADDING = 15  # 226 + 15 * 2 = 256
PATCH_SIZE = STEP_SIZE + PADDING * 2
SAVE_IMG_PER_NUM = 2000 # Save 1 example image per 2000 tissue patches

colnames = ["coordinate_x", "coordinate_y", "cell_type", "probability",
            "area", "convex_area", "eccentricity", "extent",
            "filled_area", "major_axis_length", "minor_axis_length",
            "orientation", "perimeter", "solidity", "pa_ratio", "n_contour"]

def display_segmentation(r, rgb_markers, classes):
    """Display segmentation result.
    
    Args:
        r: model.detect[0]
        rgb_markers: dict
        classes: array of ordered keys in rgb_markers 
    """
    masks = r['masks']
    class_ids = r['class_ids']
    mask = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
    mask[:, :] = rgb_markers['BG']  # Background
    if masks.shape[0] > 0:
        # print(masks.shape)
        for i in range(masks.shape[2]):
            mask[masks[:, :, i] == 1] = rgb_markers[classes[class_ids[i]]]
        plt.imshow(mask)

def get_segmentation_mask(r, rgb_markers, classes):
    """Display segmentation result for Photoshop.
    
    Args:
        r: model.detect[0]
        rgb_markers: dict
        classes: array of ordered keys in rgb_markers 
    """
    masks = r['masks']
    class_ids = r['class_ids']
    mask = np.zeros((masks.shape[0], masks.shape[1], 4), dtype=np.uint8)  # 0 is transparent
    mask[:, :, :3] = rgb_markers['BG']  # Background
    if masks.shape[0] > 0:
        # print(masks.shape)
        for i in range(masks.shape[2]):
            mask[masks[:, :, i] == 1, :3] = rgb_markers[classes[class_ids[i]]]
            mask[masks[:, :, i] == 1, 3] = 255
    return mask

class Slide:
    def __init__(self, slide_file, cuda, model=None, model_srgan=None, 
                 BATCH_SIZE=4,
                 display_progress=False, print_progress=True, save_mask=False,
                 step_size=STEP_SIZE, padding=PADDING, patch_size=PATCH_SIZE,
                 SAVE_IMG_PER_NUM=SAVE_IMG_PER_NUM,
                 MEAN=MEAN, STD=STD, default_mpp=0.25, default_magnitude="40", 
                 output_dir="./example_results"):
        """Initiate a Slide class.
        
        Args:
            slide_file: Path to slide
            model: A mask RCNN model class in "inference" mode.## WARNING: disabled for parallel_queue mode
                                                               ## Please define "model" in main function for paralle_queue mode
            model_srgan: A SRGAN model.  ## WARNING: disabled for parallel_queue mode
                                         ## Please define "model_srgan" in main function for paralle_queue mode
            cuda: -1 for CPU, 0/1 for GPU id.  ## WARNING: do not set to torch.device for parallel_queue mode
            display_progress/print_progress/save_mask: Whether to draw/print progress/save mask as pngs for future labeling
            step_size, padding, PATCH_SIZE: s + p * 2 = patch_size, to ensure overlap among patches
            SAVE_IMG_PER_NUM: 1 per SAVE_IMG_PER_NUM patches will be saved
            output_dir: Directory to save output folder
            
        Attributes:
            slide_mask, slide_image (on lowest dimension)
            slide: an open_slide object  ## WARNING: will be set to None after __init__(). 
                                         ## Please define "slide" in main function
            magnitude: 20X or 40X
            level_dims: N*2 array
        """
        global slide
        
        assert patch_size == step_size + padding * 2
        self.slide_file = slide_file
        self.model = model
        self.model_srgan = model_srgan
        self.cuda = cuda
        self.BATCH_SIZE = BATCH_SIZE
        self.save_mask = save_mask
        self.step_size = step_size
        self.padding = padding
        self.patch_size = patch_size
        self.SAVE_IMG_PER_NUM = SAVE_IMG_PER_NUM
        self.MEAN = MEAN
        self.STD = STD
        self.error = 0
        
        try: 
            slide = open_slide(slide_file)
            self.slide = slide
        except Exception as error:
            self.error = 1
            self.slide = None
        
        try:
            self.mpp = float(self.slide.properties['aperio.MPP'])
        except:
            try:
                import re
                self.mpp = float(re.search('(?<=MPP = )[0-9\.]+', self.slide.properties['openslide.comment'])[0])
            except:
                print("MPP not found, using default mpp {}".format(default_mpp))
                self.mpp = default_mpp
        try:
            self.magnitude = self.slide.properties['openslide.objective-power']
        except:
            print("Improper magnitude, using default magnitude")
            self.magnitude = default_magnitude # "40"
            # self.magnitude = None
            # self.error = 1
        
        if self.mpp * float(self.magnitude) != 10:
            # 40x: 0.2500; 20x: 0.5000
            print("WARNING: inconsistant map and magnitude detected; kept with {} mpp".format(self.mpp))
        if self.mpp < 0.4:
            self.magnitude = "40"
            self.scale = self.mpp / 0.25
        else:
            self.magnitude = "20"
            self.scale = self.mpp / 0.5
        # Update patch size
        self.step_size = int(self.step_size/self.scale)
        self.padding = int(self.padding/self.scale)
        self.patch_size = int(self.step_size + self.padding * 2)
        # Update scale again
        self.scale = patch_size / self.patch_size
        
        try:
            self.level_dims = self.slide.level_dimensions
            self.zoom = self.level_dims[0][0] / self.level_dims[-1][0]
            self.n_col = int(math.ceil(self.level_dims[0][0]/self.step_size))
            self.n_row = int(math.ceil(self.level_dims[0][1]/self.step_size))
        except:
            print("Error finding level dimensions")
            self.error = 1
        
        if self.error == 0:
            self.check_empty = False
            try:
                if len(self.level_dims) > 1:
                    for level_to_analyze, _shape in enumerate(self.level_dims): 
                        if self.level_dims[0][0]/_shape[0] > 5 or (_shape[0] < 5000 and _shape[1] < 5000):
                            break
                    self.zoom = self.level_dims[0][0] / _shape[0]
                    self.slide_mask, self.slide_image = get_mask_for_slide_image(slide_file, display_progress=display_progress, level_to_analyze=level_to_analyze)
                    self.use_tifffile = True
                else:
                    self.slide_mask, self.slide_image, self.zoom = get_mask_for_slide_image_tifffile(slide_file, display_progress=display_progress)
                    self.use_tifffile = True
            except Exception as error:
                print("IO Error: {}".format(error))
                print("No tissue segmentation used")
                try:
                    # If none of above works
                    self.slide_mask = np.array(np.zeros((10, 10)) + 1).astype(np.uint8)
                    # self.slide_image = np.array(slide.read_region((0, 0), 0, self.level_dims[0]))[..., 0:3]  ## Notice: do not use too large attribute
                    self.slide_image = np.zeros((10, 10))
                    self.zoom = 1
                    self.use_tifffile = True
                    self.check_empty = True
                except Exception as error:
                    print("IO Error: {}".format(error))
                    self.error = 1
                    self.slide_mask = None
                    self.slide_image = None
                    
            
            if print_progress:
                print("slide file: {}".format(slide_file))
                print("mpp: {}".format(self.mpp))
                print("magnitude: {}".format(self.magnitude))
                print("scale: {}".format(self.scale))
                print("level dimensions: {}".format(self.level_dims))
                print("zoomed in: {}".format(self.zoom))
            
            if not np.any(self.slide_mask):
                print("Mask too small")
                self.error = 1
            
            self.slide_id = slide_file.split("/")[-1]
            self.save_dir = os.path.join(output_dir, self.slide_id)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            print("Saving to {}.".format(self.save_dir))
            self.slide = None
            
    def sliding_through_mask(self, to_plot=True):
        """Segmentation for every sliding window in the mask.
        """
        zoom = self.zoom
        n_col = self.n_col
        n_row = self.n_row
        step_size = self.step_size
        padding = self.padding
        patch_size = self.patch_size
        
        if self.magnitude == "40":
            step_size = step_size * 2
            self.step_size = step_size
            patch_size = patch_size * 2
            self.patch_size = patch_size
            padding = padding * 2
            self.padding = padding
            n_col = int(np.ceil(n_col/2))
            n_row = int(np.ceil(n_row/2))
            self.n_col = n_col
            self.n_row = n_row
            
        print("(column number, row number) = ({}, {})".format(self.n_col, self.n_row))
        
        n_patch = 0
        if not self.check_empty:
            _, contours, _ = cv2.findContours(self.slide_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            contours = [1]

        coordinate_x = []
        coordinate_y = []
        cell_type = []
        probability = []
        
        area = []
        convex_area = []
        eccentricity = []
        extent = []
        filled_area = []
        major_axis_length = []
        minor_axis_length = []
        orientation = []
        perimeter = []
        solidity = []
        pa_ratio = []
        
        n_contour = []  # Which tissue
        
        _count = 0
        plotted = 0
        image_batch = []
        info_batch = []
        for n_contours in range(0, len(contours)):
            print("Start the {}th contour.".format(n_contours))
            # i is the y axis in the image
            for i in range(0, n_row):      
                if not self.check_empty:
                    min_row = i * step_size / zoom
                    max_row = (i + 1) * step_size / zoom
                    matches = [x for x in range(0, len(contours[n_contours][:, 0, 0]))
                               if (contours[n_contours][x, 0, 1] > min_row and contours[n_contours][x, 0, 1] < max_row)]
                try:
                    if not self.check_empty:
                        print([min(contours[n_contours][matches, 0, 0]), max(contours[n_contours][matches, 0, 0])])
                        # print ("range is "+str(len(contours[nContours][:, 0, 0])))
                        
                        min_col = min(contours[n_contours][matches, 0, 0]) * zoom
                        max_col = max(contours[n_contours][matches, 0, 0]) * zoom
                        min_col_int = int(math.floor(min_col/step_size))
                        max_col_int = int(math.ceil(max_col/step_size))
                    else:
                        min_col_int = 0
                        max_col_int = self.n_col
                        print([min_col_int, max_col_int])
                    
                    for j in range(min_col_int, max_col_int):
                        if n_patch % 2000 == 1:
                            print("{} patches processed".format(n_patch))

                        start_col = j * step_size - padding
                        start_row = i * step_size - padding
                        patch = slide.read_region((start_col, start_row), DESIRED_LEVEL, (patch_size, patch_size))
                        patch = np.array(patch)[:, :, 0:3]
                        
                        # Check empty
                        if self.check_empty:
                            if np.mean(patch) > 230 or np.mean(patch) < 10:
                                continue
                        
                        if self.magnitude == "20" and self.scale != 1:
                            #******** Resize to exactly 20X ********
                            patch = skimage.transform.resize(patch, (PATCH_SIZE, PATCH_SIZE), preserve_range=True,
                                                                 mode='constant', anti_aliasing=True).astype(np.uint8)
                        elif self.magnitude == "40" and self.scale != 1:
                            patch = skimage.transform.resize(patch, (PATCH_SIZE*2, PATCH_SIZE*2), preserve_range=True,
                                                                 mode='constant', anti_aliasing=True)
                        
                        image_batch.append(patch)
                        info_batch.append([i, j, n_contours])
                        _count += 1
                        n_patch += 1
                        if _count == self.BATCH_SIZE:
                            # Perform preprocess, prediction, and yield output
                            
                            if self.magnitude == "20":
                                #******** SRGAN ********
                                with torch.no_grad():
                                    if self.cuda != -1:
                                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0).cuda(self.cuda)  #[B, C, H, W], range 0~1
                                        image_batch = model_srgan(input_tensor.float()).data.cpu().numpy()*255
                                    else:
                                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0)  #, requires_grad=False; [B, C, H, W], range 0~1
                                        # print(input_tensor.size(), input_tensor)
                                        image_batch = model_srgan(input_tensor.float()).data.numpy()*255
                                image_batch = np.transpose(image_batch, (0, 2, 3, 1))

                                #******** Resize to 40X ********
                                # patch = skimage.transform.resize(patch, (patch_size*2, patch_size*2), preserve_range=True,
                                #                                  mode='constant', anti_aliasing=True)
                                
                            else:
                                image_batch = np.array(image_batch)
                            
                            #******** Prediction ********
                            # Normalize
                            image_batch_normalized = np.zeros_like(image_batch, dtype='float')
                            for ch in range(3):
                                image_batch_normalized[..., ch] = (image_batch[..., ch] - self.MEAN[ch])/self.STD[ch]

                            # Run detection
                            results = model.detect(image_batch_normalized, verbose=0)
                            
                            for _n_in_batch, _in_batch in enumerate(zip(results, info_batch)):
                                r, image_info = _in_batch
                                i, j, n_contours = image_info
                                
                                #******** Extract features ********
                                masks = r['masks']
                                class_ids = r['class_ids']
                                scores = r['scores']
                                # If there are any cells within the mask
                                if masks.shape[0] > 0:
                                    #******** Postprocessing ********
                                    classes = r['class_ids']
                                    occlusion = np.ones(np.shape(masks)[0:2])
                                    to_keep = []
                                    for k in range(masks.shape[2]):
                                        # Remove the mask with tooooo much overlapping with previously detected region
                                        original_area = np.sum(masks[:, :, k])
                                        masks[:, :, k] = masks[:, :, k] * occlusion
                                        new_area = np.sum(masks[:, :, k])
                                        if new_area > 0 and new_area/original_area > 0.6:
                                            to_keep.append(True)
                                            occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, k]))
                                        else:
                                            to_keep.append(False)
                                    # print(masks.shape)
                                    # print(classes)
                                    # print(to_keep)
                                    masks = masks[:, :, to_keep]
                                    classes = classes[to_keep]
                                    r['rois'] = r['rois'][to_keep]
                                    r['masks'] = r['masks'][..., to_keep]
                                    r['class_ids'] = r['class_ids'][to_keep]
                                    r['score'] = r['scores'][to_keep]
                                    
                                    # Plot
                                    if to_plot & (n_patch % self.SAVE_IMG_PER_NUM <= self.BATCH_SIZE) & (plotted==0):
                                        try:
                                            f = plt.figure(figsize=(8, 4))
                                            plt.subplot(121)
                                            plt.imshow(image_batch[_n_in_batch].astype(np.uint8))
                                            plt.subplot(122)
                                            display_segmentation(r, RGB_MARKERS, CLASSES)
                                            # plt.show()
                                            f.savefig(os.path.join(self.save_dir, "example_segmentation_{}.jpg".format(n_patch)), 
                                                      bbox_inches='tight')
                                            plt.close()
                                            plotted = 1
                                            if self.save_mask:
                                                skimage.io.imsave('example_image_{}.png'.format(n_patch), patch.astype(np.uint8))
                                                skimage.io.imsave('example_mask_{}.png'.format(n_patch), get_segmentation_mask(r, RGB_MARKERS, CLASSES))
                                        except Exception as error:
                                            print("Error saving patch plot: ", error)
                                    
                                    #******** Extract features ********
                                    # if self.magnitude == "20":
                                    if True:
                                        for n_mask in range(masks.shape[2]):
                                            try: 
                                                # label_img = skimage.measure.label(masks[:, :, n_mask])
                                                label_img = masks[:, :, n_mask]
                                                regions = skimage.measure.regionprops(label_img)
                                                if len(regions) >= 1:
                                                    this_region = regions[0]
                                                else:
                                                    continue
                                            except Exception as error:
                                                print("Error detect regions.", error)
                                                continue
                                            centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                                            # NOTE: the coordinates are under 40X
                                            if (centroid_y > PADDING * 2) & (centroid_y < (PATCH_SIZE - PADDING) * 2) &\
                                               (centroid_x > PADDING * 2) & (centroid_x < (PATCH_SIZE - PADDING) * 2):                     
                                                coordinate_y.append(centroid_y - PADDING * 2 + i * STEP_SIZE * 2)
                                                coordinate_x.append(centroid_x - PADDING * 2 + j * STEP_SIZE * 2)
                                                cell_type.append(class_ids[n_mask])
                                                probability.append(scores[n_mask])
                                                
                                                # Cell features
                                                area.append(this_region.area)
                                                convex_area.append(this_region.convex_area)
                                                eccentricity.append(this_region.eccentricity)
                                                extent.append(this_region.extent)
                                                filled_area.append(this_region.filled_area)
                                                major_axis_length.append(this_region.major_axis_length)
                                                minor_axis_length.append(this_region.minor_axis_length)
                                                orientation.append(this_region.orientation)
                                                perimeter.append(this_region.perimeter)
                                                solidity.append(this_region.solidity)
                                                pa_ratio.append(1.0 * this_region.perimeter**2 / this_region.filled_area)
                                                
                                                # Which tissue
                                                n_contour.append(n_contours)
                                    elif self.magnitude == "40":
                                        for n_mask in range(masks.shape[2]):
                                            try: 
                                                # label_img = skimage.measure.label(masks[:, :, n_mask])
                                                label_img = masks[:, :, n_mask]
                                                regions = skimage.measure.regionprops(label_img)
                                                if len(regions) >= 1:
                                                    this_region = regions[0]
                                                else:
                                                    continue
                                            except Exception as error:
                                                print("Error detect regions.", error)
                                                continue
                                            centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                                            # NOTE: the coordinates are under 40X
                                            if (centroid_y > PADDING) & (centroid_y < (PATCH_SIZE - PADDING)) &\
                                               (centroid_x > PADDING) & (centroid_x < (PATCH_SIZE - PADDING)):                     
                                                coordinate_y.append(centroid_y - PADDING + i * STEP_SIZE)
                                                coordinate_x.append(centroid_x - PADDING + j * STEP_SIZE)
                                                cell_type.append(class_ids[n_mask])
                                                probability.append(scores[n_mask])
                                                
                                                # Cell features
                                                area.append(this_region.area)
                                                convex_area.append(this_region.convex_area)
                                                eccentricity.append(this_region.eccentricity)
                                                extent.append(this_region.extent)
                                                filled_area.append(this_region.filled_area)
                                                major_axis_length.append(this_region.major_axis_length)
                                                minor_axis_length.append(this_region.minor_axis_length)
                                                orientation.append(this_region.orientation)
                                                perimeter.append(this_region.perimeter)
                                                solidity.append(this_region.solidity)
                                                pa_ratio.append(1.0 * this_region.perimeter**2 / this_region.filled_area)
                                                
                                                # Which tissue
                                                n_contour.append(n_contours)
                                                
                            # Initialize
                            image_batch = []
                            info_batch = []
                            _count = 0
                            plotted = 0
                            
                except Exception as error:
                    print([e for e in error.args])
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    continue
                    
        # In case any remained
        if _count != 0:
            # Perform preprocess, prediction, and yield output
            if self.magnitude == "20":
                #******** SRGAN ********
                with torch.no_grad():
                    if self.cuda != -1:
                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                   dim=0).cuda(self.cuda)  #[B, C, H, W], range 0~1
                        image_batch = model_srgan(input_tensor).data.cpu().numpy()*255
                    else:
                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                   dim=0)  #, requires_grad=False; [B, C, H, W], range 0~1
                        # print(input_tensor.size(), input_tensor)
                        image_batch = model_srgan(input_tensor).data.numpy()*255
                image_batch = np.transpose(image_batch, (0, 2, 3, 1))

                #******** Resize to 40X ********
                # patch = skimage.transform.resize(patch, (patch_size*2, patch_size*2), preserve_range=True,
                #                                  mode='constant', anti_aliasing=True)
            else:
                image_batch = np.array(image_batch)
            
            #******** Prediction ********
            # Normalize
            image_batch_normalized = np.zeros_like(image_batch, dtype='float')
            for ch in range(3):
                image_batch_normalized[..., ch] = (image_batch[..., ch] - self.MEAN[ch])/self.STD[ch]
                
            # Make image_batch_normalized same as batch size
            image_batch_normalized = [image_batch_normalized[_] for _ in range(_count)]
            for _ in range(_count, self.BATCH_SIZE):
                image_batch_normalized.append(np.zeros((10, 10, 3)))
                info_batch.append([-1, -1, -1])

            # Run detection
            results = model.detect(image_batch_normalized, verbose=0)
            
            for r, image_info in zip(results, info_batch):
                i, j, n_contours = image_info
                
                #******** Extract features ********
                masks = r['masks']
                class_ids = r['class_ids']
                scores = r['scores']
                # If there are any cells within the mask
                if masks.shape[0] > 0:
                    #******** Postprocessing ********
                    classes = r['class_ids']
                    occlusion = np.ones(np.shape(masks)[0:2])
                    to_keep = []
                    for k in range(masks.shape[2]):
                        # Remove the mask with tooooo much overlapping with previously detected region
                        original_area = np.sum(masks[:, :, k])
                        masks[:, :, k] = masks[:, :, k] * occlusion
                        new_area = np.sum(masks[:, :, k])
                        if new_area > 0 and new_area/original_area > 0.6:
                            to_keep.append(True)
                            occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, k]))
                        else:
                            to_keep.append(False)
                    # print(masks.shape)
                    # print(classes)
                    # print(to_keep)
                    masks = masks[:, :, to_keep]
                    classes = classes[to_keep]
                    r['rois'] = r['rois'][to_keep]
                    r['masks'] = r['masks'][..., to_keep]
                    r['class_ids'] = r['class_ids'][to_keep]
                    r['score'] = r['scores'][to_keep]
                    
                    # Plot
                    if to_plot & (n_patch % self.SAVE_IMG_PER_NUM <= self.BATCH_SIZE):
                        try:
                            f = plt.figure(figsize=(8, 4))
                            plt.subplot(121)
                            plt.imshow(patch.astype(np.uint8))
                            plt.subplot(122)
                            display_segmentation(r, RGB_MARKERS, CLASSES)
                            # plt.show()
                            f.savefig(os.path.join(self.save_dir, "example_segmentation_{}.jpg".format(n_patch)), 
                                      bbox_inches='tight')
                            plt.close()
                            if self.save_mask:
                                skimage.io.imsave(os.path.join(self.save_dir, 'example_image_{}.png'.format(n_patch)), patch.astype(np.uint8))
                                skimage.io.imsave(os.path.join(self.save_dir, 'example_mask_{}.png'.format(n_patch)), get_segmentation_mask(r, RGB_MARKERS, CLASSES))
                        except Exception as error:
                            print("Error saving patch plot: ", error)
                    
                    #******** Extract features ********
                    # if self.magnitude == "20":
                    if True:
                        for n_mask in range(masks.shape[2]):
                            try: 
                                # label_img = skimage.measure.label(masks[:, :, n_mask])
                                label_img = masks[:, :, n_mask]
                                regions = skimage.measure.regionprops(label_img)
                                if len(regions) >= 1:
                                    this_region = regions[0]
                                else:
                                    continue
                            except Exception as error:
                                print("Error detect regions.", error)
                                continue
                            centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                            # NOTE: the coordinates are under 40X
                            if (centroid_y > PADDING * 2) & (centroid_y < (PATCH_SIZE - PADDING) * 2) &\
                               (centroid_x > PADDING * 2) & (centroid_x < (PATCH_SIZE - PADDING) * 2):                     
                                coordinate_y.append(centroid_y - PADDING * 2 + i * STEP_SIZE * 2)
                                coordinate_x.append(centroid_x - PADDING * 2 + j * STEP_SIZE * 2)
                                cell_type.append(class_ids[n_mask])
                                probability.append(scores[n_mask])
                                
                                # Cell features
                                area.append(this_region.area)
                                convex_area.append(this_region.convex_area)
                                eccentricity.append(this_region.eccentricity)
                                extent.append(this_region.extent)
                                filled_area.append(this_region.filled_area)
                                major_axis_length.append(this_region.major_axis_length)
                                minor_axis_length.append(this_region.minor_axis_length)
                                orientation.append(this_region.orientation)
                                perimeter.append(this_region.perimeter)
                                solidity.append(this_region.solidity)
                                pa_ratio.append(1.0 * this_region.perimeter**2 / this_region.filled_area)
                                
                                # Which tissue
                                n_contour.append(n_contours)
                    elif self.magnitude == "40":
                        for n_mask in range(masks.shape[2]):
                            try: 
                                # label_img = skimage.measure.label(masks[:, :, n_mask])
                                label_img = masks[:, :, n_mask]
                                regions = skimage.measure.regionprops(label_img)
                                if len(regions) >= 1:
                                    this_region = regions[0]
                                else:
                                    continue
                            except Exception as error:
                                print("Error detect regions.", error)
                                continue
                            centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                            # NOTE: the coordinates are under 40X
                            if (centroid_y > PADDING) & (centroid_y < (PATCH_SIZE - PADDING)) &\
                               (centroid_x > PADDING) & (centroid_x < (PATCH_SIZE - PADDING)):                     
                                coordinate_y.append(centroid_y - PADDING + i * STEP_SIZE)
                                coordinate_x.append(centroid_x - PADDING + j * STEP_SIZE)
                                cell_type.append(class_ids[n_mask])
                                probability.append(scores[n_mask])
                                
                                # Cell features
                                area.append(this_region.area)
                                convex_area.append(this_region.convex_area)
                                eccentricity.append(this_region.eccentricity)
                                extent.append(this_region.extent)
                                filled_area.append(this_region.filled_area)
                                major_axis_length.append(this_region.major_axis_length)
                                minor_axis_length.append(this_region.minor_axis_length)
                                orientation.append(this_region.orientation)
                                perimeter.append(this_region.perimeter)
                                solidity.append(this_region.solidity)
                                pa_ratio.append(1.0 * this_region.perimeter**2 / this_region.filled_area)
                                
                                # Which tissue
                                n_contour.append(n_contours)

        print("{} patches analyzed".format(n_patch))
        self.n_patch = n_patch
        self.coordinate_x = coordinate_x
        self.coordinate_y = coordinate_y
        self.cell_type = cell_type
        self.probability = probability
        self.area = area
        self.convex_area = convex_area
        self.eccentricity = eccentricity
        self.extent = extent
        self.filled_area = filled_area
        self.major_axis_length = major_axis_length
        self.minor_axis_length = minor_axis_length
        self.orientation = orientation
        self.perimeter = perimeter
        self.solidity = solidity
        self.pa_ratio = pa_ratio
        self.n_contour = n_contour
        
    def save_scatter_plot(self, s=0.1):
        """Save to pdf"""
        f = plt.figure(figsize=(16, 32))
        plt.subplot(211)
        if self.check_empty:
            self.slide_image = np.array(slide.read_region((0, 0), 0, self.level_dims[0]))[..., 0:3]  ## Notice: do not use too large attribute
        plt.imshow(self.slide_image)
        plt.subplot(212)
        mycolor = ["black", "green", "red", "blue", "pink", "yellow", "cyan"]
        if self.magnitude == "20":
            plt.scatter([this_x/self.zoom/2/self.scale for this_x in self.coordinate_x], 
                        [-this_y/self.zoom/2/self.scale for this_y in self.coordinate_y], 
                        c=[mycolor[this_cell_type] for this_cell_type in self.cell_type],
                        s=s, alpha=1)
        elif self.magnitude == "40":
            plt.scatter([this_x/self.zoom/self.scale for this_x in self.coordinate_x], 
                        [-this_y/self.zoom/self.scale for this_y in self.coordinate_y], 
                        c=[mycolor[this_cell_type] for this_cell_type in self.cell_type],
                        s=s, alpha=1)
        plt.gca().set_aspect("equal")
        if not self.use_tifffile:
            plt.xlim(0, self.level_dims[-1][0])
            plt.ylim(-self.level_dims[-1][1], 0)
        else:
            plt.xlim(0, self.level_dims[0][0]/self.zoom)
            plt.ylim(-self.level_dims[0][1]/self.zoom, 0)
        # plt.colorbar()
        # plt.show()
        f.savefig(os.path.join(self.save_dir, "scatter_plot.jpg"), bbox_inches='tight')
        plt.close()
        
    def save_cell_summary(self):
        """Save to csv"""
        cell_summary_df = pd.DataFrame({"coordinate_x": self.coordinate_x, 
                                        "coordinate_y": self.coordinate_y, 
                                        "cell_type": self.cell_type,
                                        "probability": self.probability,
                                        "area": self.area,
                                        "convex_area": self.convex_area,
                                        "eccentricity": self.eccentricity,
                                        "extent": self.extent,
                                        "filled_area": self.filled_area,
                                        "major_axis_length": self.major_axis_length,
                                        "minor_axis_length": self.minor_axis_length,
                                        "orientation": self.orientation,
                                        "perimeter": self.perimeter,
                                        "solidity": self.solidity,
                                        "pa_ratio": self.pa_ratio,
                                        "n_contour": self.n_contour})
        cell_summary_df.to_csv(os.path.join(self.save_dir, "cell_summary_{}X_{}_to_{}.csv".format(self.magnitude, self.patch_size, PATCH_SIZE)), 
                               index=False)
        
    ########################################
    # Functions for parallel_queue
    ########################################
    
    def segmentation_generator(self, to_plot=True):
        zoom = self.zoom
        n_col = self.n_col
        n_row = self.n_row
        step_size = self.step_size
        padding = self.padding
        patch_size = self.patch_size
        
        if self.magnitude == "40":
            step_size = step_size * 2
            self.step_size = step_size
            patch_size = patch_size * 2
            self.patch_size = patch_size
            padding = padding * 2
            self.padding = padding
            n_col = int(np.ceil(n_col/2))
            n_row = int(np.ceil(n_row/2))
            self.n_col = n_col
            self.n_row = n_row
        
        print("(column number, row number) = ({}, {})".format(self.n_col, self.n_row))
        
        if not self.check_empty:
            _, contours, _ = cv2.findContours(self.slide_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            contours = [1]
        
        image_batch = []
        info_batch = []
        _count = 0  # count for BATCH_SIZE
        n_patch = 0  # count for total patches
        for n_contours in range(0, len(contours)):
            print("Start the {}th contour.".format(n_contours))
            # i is the y axis in the image
            for i in range(0, n_row):
                if not self.check_empty:
                    min_row = i * step_size / zoom
                    max_row = (i + 1) * step_size / zoom
                    matches = [x for x in range(0, len(contours[n_contours][:, 0, 0]))
                               if (contours[n_contours][x, 0, 1] > min_row and contours[n_contours][x, 0, 1] < max_row)]
                try:
                    if not self.check_empty:
                        print([min(contours[n_contours][matches, 0, 0]), max(contours[n_contours][matches, 0, 0])])
                        # print ("range is "+str(len(contours[nContours][:, 0, 0])))
                        
                        min_col = min(contours[n_contours][matches, 0, 0]) * zoom
                        max_col = max(contours[n_contours][matches, 0, 0]) * zoom
                        min_col_int = int(math.floor(min_col/step_size))
                        max_col_int = int(math.ceil(max_col/step_size))
                    else:
                        min_col_int = 0
                        max_col_int = self.n_col
                        print([min_col_int, max_col_int])
                    
                    for j in range(min_col_int, max_col_int):
                        if n_patch % 2000 == 1:
                            print("{} patches processed".format(n_patch))
                        
                        start_col = j * step_size - padding
                        start_row = i * step_size - padding
                        patch = slide.read_region((start_col, start_row), DESIRED_LEVEL, (patch_size, patch_size))
                        patch = np.array(patch)[:, :, 0:3]
                        
                        # Check empty
                        if self.check_empty:
                            if np.mean(patch) > 230 or np.mean(patch) < 10:
                                continue
                        
                        if self.magnitude == "20" and self.scale != 1:
                            #******** Resize to exactly 20X ********
                            patch = skimage.transform.resize(patch, (PATCH_SIZE, PATCH_SIZE), preserve_range=True,
                                                                 mode='constant', anti_aliasing=True).astype(np.uint8)
                        elif self.magnitude == "40" and self.scale != 1:
                            patch = skimage.transform.resize(patch, (PATCH_SIZE*2, PATCH_SIZE*2), preserve_range=True,
                                                                 mode='constant', anti_aliasing=True)
                        
                        image_batch.append(patch)
                        info_batch.append([i, j, n_contours])
                        _count += 1
                        n_patch += 1
                        if _count == self.BATCH_SIZE:
                            # Perform preprocess, prediction, and yield output
                            
                            if self.magnitude == "20":
                                #******** SRGAN ********
                                with torch.no_grad():
                                    if self.cuda != -1:
                                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0).cuda(self.cuda)  #[B, C, H, W], range 0~1
                                        image_batch = model_srgan(input_tensor).data.cpu().numpy()*255
                                    else:
                                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0)  #, requires_grad=False; [B, C, H, W], range 0~1
                                        # print(input_tensor.size(), input_tensor)
                                        image_batch = model_srgan(input_tensor).data.numpy()*255
                                image_batch = np.transpose(image_batch, (0, 2, 3, 1))

                                #******** Resize to 40X ********
                                # patch = skimage.transform.resize(patch, (patch_size*2, patch_size*2), preserve_range=True,
                                #                                  mode='constant', anti_aliasing=True)
                            else:
                                image_batch = np.array(image_batch)
                            
                            #******** Prediction ********
                            # Normalize
                            image_batch_normalized = np.zeros_like(image_batch, dtype='float')
                            for ch in range(3):
                                image_batch_normalized[..., ch] = (image_batch[..., ch] - self.MEAN[ch])/self.STD[ch]

                            # Run detection
                            results = model.detect(image_batch_normalized, verbose=0)

                            # Yield
                            for _ in zip(results, info_batch):
                                yield _

                            # Plot
                            if to_plot and n_patch % self.SAVE_IMG_PER_NUM <= self.BATCH_SIZE:
                                try:
                                    patch = image_batch[0]
                                    r = results[0]
                                    f = plt.figure(figsize=(8, 4))
                                    plt.subplot(121)
                                    plt.imshow(patch.astype(np.uint8))
                                    plt.subplot(122)
                                    display_segmentation(r, RGB_MARKERS, CLASSES)
                                    # plt.show()
                                    f.savefig(os.path.join(self.save_dir, "example_segmentation_{}.jpg".format(n_patch)), 
                                              bbox_inches='tight')
                                    plt.close()
                                    if self.save_mask:
                                        skimage.io.imsave(os.path.join(self.save_dir, 'example_image_{}.png'.format(n_patch)), patch.astype(np.uint8))
                                        skimage.io.imsave(os.path.join(self.save_dir, 'example_mask_{}.png'.format(n_patch)), get_segmentation_mask(r, RGB_MARKERS, CLASSES))
                                except Exception as error:
                                    print("Error saving patch plot: ", error)

                            # Initialize
                            image_batch = []
                            info_batch = []
                            _count = 0
                except Exception as error:
                    print([e for e in error.args])
                    # raise Exception
                    #continue
        
        print("Main batches finished")
        # In case any remained
        if _count != 0:
            # Perform preprocess, prediction, and yield output
            if self.magnitude == "20":
                #******** SRGAN ********
                with torch.no_grad():
                    if self.cuda != -1:
                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0).cuda(self.cuda)  #[B, C, H, W], range 0~1
                        image_batch = model_srgan(input_tensor).data.cpu().numpy()*255
                    else:
                        input_tensor = torch.stack([Variable(ToTensor()(_)) for _ in image_batch], 
                                                                   dim=0)  #, requires_grad=False; [B, C, H, W], range 0~1
                        # print(input_tensor.size(), input_tensor)
                        image_batch = model_srgan(input_tensor).data.numpy()*255
                image_batch = np.transpose(image_batch, (0, 2, 3, 1))

                #******** Resize to 40X ********
                # patch = skimage.transform.resize(patch, (patch_size*2, patch_size*2), preserve_range=True,
                #                                  mode='constant', anti_aliasing=True)
            else:
                image_batch = np.array(image_batch)

            #******** Prediction ********
            # Normalize
            image_batch_normalized = np.zeros_like(image_batch, dtype='float')
            for ch in range(3):
                image_batch_normalized[..., ch] = (image_batch[..., ch] - self.MEAN[ch])/self.STD[ch]
                
            # Make image_batch_normalized same as batch size
            image_batch_normalized = [image_batch_normalized[_] for _ in range(_count)]
            for _ in range(_count, self.BATCH_SIZE):
                image_batch_normalized.append(np.zeros((10, 10, 3)))
                info_batch.append([-1, -1, -1])

            # Run detection
            results = model.detect(image_batch_normalized, verbose=0)
            
            # Yield
            for _ in zip(results, info_batch):
                yield _
                
        print("{} patches analyzed".format(n_patch))
        
    
    def feature_generator(self, inputs):
        # print("This is feature generator")
        r, image_info = inputs
        i, j, n_contours = image_info
        padding = self.padding
        patch_size = self.patch_size
        step_size = self.step_size
        
        #******** Extract features ********
        masks = r['masks']
        class_ids = r['class_ids']
        scores = r['scores']
        # If there are any cells within the mask
        if masks.shape[0] > 0:
            #******** Postprocessing ********
            classes = r['class_ids']
            occlusion = np.ones(np.shape(masks)[0:2])
            to_keep = []
            for k in range(masks.shape[2]):
                # Remove the mask with tooooo much overlapping with previously detected region
                original_area = np.sum(masks[:, :, k])
                masks[:, :, k] = masks[:, :, k] * occlusion
                new_area = np.sum(masks[:, :, k])
                if new_area > 0 and new_area/original_area > 0.6:
                    to_keep.append(True)
                    occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, k]))
                else:
                    to_keep.append(False)
            # print(masks.shape)
            # print(classes)
            # print(to_keep)
            masks = masks[:, :, to_keep]
            classes = classes[to_keep]
            r['rois'] = r['rois'][to_keep]
            r['masks'] = r['masks'][..., to_keep]
            r['class_ids'] = r['class_ids'][to_keep]
            r['score'] = r['scores'][to_keep]

            #******** Extract features ********
            # if self.magnitude == "20":
            if True:
                for n_mask in range(masks.shape[2]):
                    try: 
                        # label_img = skimage.measure.label(masks[:, :, n_mask])
                        label_img = masks[:, :, n_mask]
                        regions = skimage.measure.regionprops(label_img, coordinates='xy')
                        if len(regions) >= 1:
                            this_region = regions[0]
                        else:
                            continue
                    except Exception as error:
                        print("Error detect regions.", error)
                        continue
                    centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                    # NOTE: the coordinates are under 40X
                    if (centroid_y > PADDING * 2) & (centroid_y < (PATCH_SIZE - PADDING) * 2) &\
                       (centroid_x > PADDING * 2) & (centroid_x < (PATCH_SIZE - PADDING) * 2):                        
                        yield [centroid_x - PADDING * 2 + j * STEP_SIZE * 2, 
                               centroid_y - PADDING * 2 + i * STEP_SIZE * 2, 
                               class_ids[n_mask],
                               scores[n_mask],
                               this_region.area,
                               this_region.convex_area,
                               this_region.eccentricity,
                               this_region.extent,
                               this_region.filled_area,
                               this_region.major_axis_length,
                               this_region.minor_axis_length,
                               this_region.orientation,
                               this_region.perimeter,
                               this_region.solidity,
                               1.0 * this_region.perimeter**2 / this_region.filled_area,
                               n_contours]
            elif self.magnitude == "40":
                for n_mask in range(masks.shape[2]):
                    try: 
                        # label_img = skimage.measure.label(masks[:, :, n_mask])
                        label_img = masks[:, :, n_mask]
                        regions = skimage.measure.regionprops(label_img, coordinates='xy')
                        if len(regions) >= 1:
                            this_region = regions[0]
                        else:
                            continue
                    except Exception as error:
                        print("Error detect regions.", error)
                        continue
                    centroid_y, centroid_x = this_region.centroid  # y: rows; x: columns
                    # NOTE: the coordinates are under 40X
                    if (centroid_y > PADDING) & (centroid_y < (PATCH_SIZE - PADDING)) &\
                       (centroid_x > PADDING) & (centroid_x < (PATCH_SIZE - PADDING)):
                        yield [centroid_x - PADDING + j * STEP_SIZE, 
                               centroid_y - PADDING + i * STEP_SIZE, 
                               class_ids[n_mask],
                               scores[n_mask],
                               this_region.area,
                               this_region.convex_area,
                               this_region.eccentricity,
                               this_region.extent,
                               this_region.filled_area,
                               this_region.major_axis_length,
                               this_region.minor_axis_length,
                               this_region.orientation,
                               this_region.perimeter,
                               this_region.solidity,
                               1.0 * this_region.perimeter**2 / this_region.filled_area,
                               n_contours]
                        
    def export_res(self, res):
        print('Saving cell summary...')
        cell_summary_df = pd.DataFrame(res, columns=colnames)
        cell_summary_df.to_csv(os.path.join(self.save_dir, "cell_summary_{}X_{}_to_{}.csv".format(self.magnitude, self.patch_size, PATCH_SIZE*2)), 
                       index=False)
        
    def export_scatter_plot(self, res, s=0.1):
        """Save to pdf"""
        print('Saving scatter plot...')
        f = plt.figure(figsize=(16, 32))
        plt.subplot(211)
        if self.check_empty:
            self.slide_image = np.array(slide.read_region((0, 0), 0, self.level_dims[0]))[..., 0:3]  ## Notice: do not use too large attribute
        plt.imshow(self.slide_image)
        plt.subplot(212)
        mycolor = ["black", "green", "red", "blue", "pink", "yellow", "cyan"]
        
        cell_summary_df = pd.DataFrame(res, columns=colnames)
        coordinate_x = cell_summary_df['coordinate_x'].values
        coordinate_y = cell_summary_df['coordinate_y'].values
        cell_type = cell_summary_df['cell_type'].values
        
        if self.magnitude == "20":
            plt.scatter([this_x/self.zoom/2/self.scale for this_x in coordinate_x], 
                        [-this_y/self.zoom/2/self.scale for this_y in coordinate_y], 
                        c=[mycolor[this_cell_type] for this_cell_type in cell_type],
                        s=s, alpha=1)
        elif self.magnitude == "40":
            plt.scatter([this_x/self.zoom/self.scale for this_x in coordinate_x], 
                        [-this_y/self.zoom/self.scale for this_y in coordinate_y], 
                        c=[mycolor[this_cell_type] for this_cell_type in cell_type],
                        s=s, alpha=1)
        plt.gca().set_aspect("equal")
        if not self.use_tifffile:
            plt.xlim(0, self.level_dims[-1][0])
            plt.ylim(-self.level_dims[-1][1], 0)
        else:
            plt.xlim(0, self.level_dims[0][0]/self.zoom)
            plt.ylim(-self.level_dims[0][1]/self.zoom, 0)
        # plt.colorbar()
        # plt.show()
        f.savefig(os.path.join(self.save_dir, "scatter_plot.jpg"), bbox_inches='tight')
        plt.close()