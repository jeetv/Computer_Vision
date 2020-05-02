#!/usr/bin/env python
'''
===============================================================================
Draw Rectangle using left mouse button
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

import sys
import numpy as np
import cv2 as cv
import igraph as ig
from GMM import GaussianMixture

BLUE = [255, 0, 0]        # bounding_boxangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

# setting up flags
bounding_box = (0, 0, 1, 1)
drawing = False         # flag for drawing curves
bounding_boxangle = False       # flag for drawing bounding_box
bounding_box_or_mask = 100      # flag for selecting bounding_box or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
skip_learn_GMMs = False # whether to skip learning GMM parameters


def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, bounding_boxangle, bounding_box, bounding_box_or_mask, ix, iy

    # Draw bounding_boxangle
    if event == cv.EVENT_LBUTTONDOWN:
        bounding_boxangle = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if bounding_boxangle == True:
            img = img2.copy()
            cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            bounding_box = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
            #bounding_box_or_mask = 0

    elif event == cv.EVENT_LBUTTONUP:
        bounding_boxangle = False
        #bounding_box_over = True
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        bounding_box = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        #sbounding_box_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")



class GrabCut:
    def __init__(self, img, mask, bounding_box=None, gmm_components=5):
        self.img = np.asarray(img, dtype=np.float64)
        self.rows, self.cols = img.shape[0],img.shape[1]

        self.mask = mask
        if bounding_box is not None:
            self.mask[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = DRAW_PR_FG['val']
        self.classify_pixels()

        # Best number of GMM components K suggested in paper
        self.gmm_components = gmm_components
        self.gamma = 50  # Best gamma suggested in paper formula (5)
        self.beta = 0

        self.dis_W = np.empty((self.rows, self.cols - 1))
        self.dis_NW = np.empty((self.rows - 1, self.cols - 1))
        self.dis_N = np.empty((self.rows - 1, self.cols))
        self.dis_NE = np.empty((self.rows - 1, self.cols - 1))

        self.bgd_gmm = None
        self.fgd_gmm = None
        self.comp_idxs = np.empty((self.rows, self.cols), dtype=np.uint32)

        self.gc_graph = None
        self.gc_graph_capacity = None           # Edge capacities
        self.gc_source = self.cols * self.rows  # "object" terminal S
        self.gc_sink = self.gc_source + 1       # "background" terminal T
        
        #calculate ||Zm-Zn||^2 (four directions enough)
        left_diffr = self.img[:, 1:] - self.img[:, :-1]
        upleft_diffr = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diffr = self.img[1:, :] - self.img[:-1, :]
        upright_diffr = self.img[1:, :-1] - self.img[:-1, 1:]

        #calculate Beta
        self.beta = np.sum(np.square(left_diffr)) + np.sum(np.square(upleft_diffr)) + np.sum(np.square(up_diffr)) + np.sum(np.square(upright_diffr))
        self.beta = 1 / (2 * self.beta / (4 * self.cols * self.rows - 3 * self.cols - 3 * self.rows + 2)) 

        # Smoothness term V described in formula (11)
        # define V edges
        self.dis_W = self.gamma * np.exp(-self.beta * np.sum(np.square(left_diffr), axis=2))
        self.dis_NW = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upleft_diffr), axis=2))
        self.dis_N = self.gamma * np.exp(-self.beta * np.sum(np.square(up_diffr), axis=2))
        self.dis_NE = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upright_diffr), axis=2))
            
        # Apply GaussianMixture for both foreground and background   
        self.bgd_gmm = GaussianMixture(self.img[self.bgd_indexes])
        self.fgd_gmm = GaussianMixture(self.img[self.fgd_indexes])

    def classify_pixels(self):
        self.bgd_indexes = np.where(np.logical_or(self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))
        self.fgd_indexes = np.where(np.logical_or(self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))

        assert self.bgd_indexes[0].size > 0
        assert self.fgd_indexes[0].size > 0

        #print('(pr_)bgd count: %d, (pr_)fgd count: %d' % (self.bgd_indexes[0].size, self.fgd_indexes[0].size))


    def assign_GMMs_components(self):
        '''Step 1 in Figure 3: Assign GMM components to pixels'''
        self.comp_idxs[self.bgd_indexes] = self.bgd_gmm.which_component(self.img[self.bgd_indexes])
        self.comp_idxs[self.fgd_indexes] = self.fgd_gmm.which_component(self.img[self.fgd_indexes])

    def learn_GMMs(self):
        '''Step 2 in Figure 3: Learn GMM parameters from data z'''
        self.bgd_gmm.fit(self.img[self.bgd_indexes],self.comp_idxs[self.bgd_indexes])
        self.fgd_gmm.fit(self.img[self.fgd_indexes],self.comp_idxs[self.fgd_indexes])

    def construct_gc_graph(self):
        bgd_indexes = np.where(self.mask.reshape(-1) == DRAW_BG['val'])
        fgd_indexes = np.where(self.mask.reshape(-1) == DRAW_FG['val'])
        pr_indexes = np.where(np.logical_or(self.mask.reshape(-1) == DRAW_PR_BG['val'], self.mask.reshape(-1) == DRAW_PR_FG['val']))

        #print('bgd count: %d, fgd count: %d, uncertain count: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

        edges = []
        self.gc_graph_capacity = []

        # t-links
        # construct the cut graph
        edges.extend(list(zip([self.gc_source] * pr_indexes[0].size, pr_indexes[0])))
        _D = -np.log(self.bgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes]))
        self.gc_graph_capacity.extend(_D.tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(list(zip([self.gc_sink] * pr_indexes[0].size, pr_indexes[0])))
        _D = -np.log(self.fgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes]))
        self.gc_graph_capacity.extend(_D.tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(list(zip([self.gc_source] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(list(zip([self.gc_sink] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(list(zip([self.gc_source] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(list(zip([self.gc_sink] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        img_indexes = np.arange(self.rows * self.cols, dtype=np.uint32).reshape(self.rows, self.cols)

        # W Direction
        mask1 = img_indexes[:, 1:].reshape(-1)
        mask2 = img_indexes[:, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.dis_W.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # NW Direction
        mask1 = img_indexes[1:, 1:].reshape(-1)
        mask2 = img_indexes[:-1, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.dis_NW.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # N Direction
        mask1 = img_indexes[1:, :].reshape(-1)
        mask2 = img_indexes[:-1, :].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.dis_N.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # NE Direction
        mask1 = img_indexes[1:, :-1].reshape(-1)
        mask2 = img_indexes[:-1, 1:].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.dis_NE.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)
        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + 2 * self.cols * self.rows

        self.gc_graph = ig.Graph(self.cols * self.rows + 2)
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        """Step 3 in Figure 3: Estimate segmentation"""
        mincut = self.gc_graph.st_mincut(self.gc_source, self.gc_sink, self.gc_graph_capacity)
        print('foreground pixels: %d, background pixels: %d' % (len(mincut.partition[0]), len(mincut.partition[1])))
        pr_indexes = np.where(np.logical_or(self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        img_indexes = np.arange(self.rows * self.cols,dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        self.classify_pixels()

    def run(self, num_iters=1):
        for _ in range(num_iters):
            self.assign_GMMs_components()
            self.learn_GMMs()
            self.construct_gc_graph()
            self.estimate_segmentation()
         


if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # reading images
    filename = 'input_data/images/bush.jpg'

    img = cv.imread(filename)
    img2 = img.copy()                               # a copy of original image
    #mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
    mask = np.zeros((img.shape[0],img.shape[1]),dtype = np.uint8)
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

    # input and output windows
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', img.shape[1]+10, 90)

    print(" Instructions: \n")
    print(" Draw a bounding_boxangle around the object using left mouse button \n")

    while(1):

        cv.imshow('output', output)
        cv.imshow('input', img)
        k = cv.waitKey(1)

        # key bindings
        if k == ord('e'):         # esc to exit
            break
        elif k == ord('0'):  # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('1'):  # FG drawing
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('2'):  # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'):  # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('s'):  # save image
            bar = np.zeros((img.shape[0], 5, 3), np.uint8)
            res = np.hstack((img2, bar, img, bar, output))
            cv.imwrite('grabcut_output.png', res)
            print(" Result saved as image \n")
        elif k == ord('r'):  # reset everything
            print("resetting \n")
            bounding_box = (0, 0, 1, 1)
            drawing = False
            bounding_boxangle = False
            value = DRAW_FG
            img = img2.copy()
            # mask initialized to PR_BG
            mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            # output image to be shown
            output = np.zeros(img.shape, np.uint8)
        elif k == ord('n'):  # segment the image
            gc = GrabCut(img2, mask, bounding_box)
            gc.run()

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2, img2, mask=mask2)

    cv.destroyAllWindows()

