import typing

import numpy as np
import cv2
from scipy import ndimage

from PIL import Image
import imageio

from base import SeamCarving

# SCORE -2 if additional modules are imported
# Hint: https://github.com/andrewdcampbell/seam-carving/blob/master/seam_carving.py


class SeamCarvingImplementation(SeamCarving):
    # The cost function could be backward or forward energy (either ways work)
    # The backward energy function computes the gradient magnitude
    # The forward energy function is more complex and an improvement to the backward energy function

    # Implement as much helper functions as you need
    ENERGY_MASK_CONST = 100000.0
    MASK_THRESHOLD = 0 
    DOWNSIZE_WIDTH = 500
    SHOULD_DOWNSIZE = True
    ctr = 0
    maskctr = 0
    isgray = False

    def resize(self, image, width, mask = False):
        dim = None
        h, w = image.shape[:2]
        dim = (width, int(h * width / float(w)))

        if mask == True:
            retain_mask = image.astype(float)
            resized = cv2.resize(retain_mask, dsize=dim, interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(image, dim)

        return resized

    def rotate_image(self, image, clockwise):
        print("Rotating....")
        k = 1 if clockwise else 3
        return np.rot90(image, k) 
    
    def backward_energy(self, image):
        """
        Simple gradient magnitude energy map. (represent salient features)
        """
        xgrad = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
        ygrad = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')
    
        grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

        # vis = visualize(grad_mag)
        # cv2.imwrite("backward_energy_demo.jpg", vis)
        print("Got the backward energy map!")
        return grad_mag


    def forward_energy(self, im):
        """
        Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
        by Rubinstein, Shamir, Avidan.
        Vectorized code adapted from
        https://github.com/axu2/improved-seam-carving.
        """
        h, w = im.shape[:2]
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

        energy = np.zeros((h, w))
        m = np.zeros((h, w))
        
        U = np.roll(im, 1, axis=0)
        L = np.roll(im, 1, axis=1)
        R = np.roll(im, -1, axis=1)
        
        cU = np.abs(R - L)
        cL = np.abs(U - L) + cU
        cR = np.abs(U - R) + cU
        
        for i in range(1, h):
            mU = m[i-1]
            mL = np.roll(mU, 1)
            mR = np.roll(mU, -1)
            
            mULR = np.array([mU, mL, mR])
            cULR = np.array([cU[i], cL[i], cR[i]])
            mULR += cULR

            argmins = np.argmin(mULR, axis=0)
            m[i] = np.choose(argmins, mULR)
            energy[i] = np.choose(argmins, cULR)
        
        # vis = visualize(energy)
        # cv2.imwrite("forward_energy_demo.jpg", vis)     
        print("Got the forward energy map!")
        return energy

    def add_seam(self, image, seam_idx):
        print("Adding seam....")
        self.ctr += 1
        """
        Add a vertical seam to a 3-channel color image at the indices provided 
        by averaging the pixels values to the left and right of the seam.
        Code adapted from https://github.com/vivianhylee/seam-carving.
        """
        h, w = image.shape[:2]
        output = np.zeros((h, w + 1, 3))
        for row in range(h):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(image[row, col: col + 2, ch])
                    output[row, col, ch] = image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = image[row, col:, ch]
                else:
                    p = np.average(image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = image[row, col:, ch]

        return output

    def add_seam_grayscale(self, image, seam_idx):
        print("Adding seam grayscale...")
        """
        Add a vertical seam to a grayscale image at the indices provided 
        by averaging the pixels values to the left and right of the seam.
        """    
        h, w = image.shape[:2]
        output = np.zeros((h, w + 1))
        for row in range(h):
            col = seam_idx[row]
            if col == 0:
                p = np.average(image[row, col: col + 2])
                output[row, col] = image[row, col]
                output[row, col + 1] = p
                output[row, col + 1:] = image[row, col:]
            else:
                p = np.average(image[row, col - 1: col + 1])
                output[row, : col] = image[row, : col]
                output[row, col] = p
                output[row, col + 1:] = image[row, col:]

        return output

    def remove_seam(self, image, boolmask):
        print("removing seam...")
        self.ctr += 1
        #print(self.ctr)
        h, w = image.shape[:2]
        boolmask3c = np.stack([boolmask] * 3, axis=2)
        return image[boolmask3c].reshape((h, w - 1, 3))

    def remove_seam_grayscale(self, image, boolmask):
        print("removing seam grayscale....")
        h, w = image.shape[:2]
        return image[boolmask].reshape((h, w - 1))

    def get_minimum_seam(self, image, mask=None, remove_mask=None):
        """
        DP algorithm for finding the seam of minimum energy. Code adapted from 
        https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
        """
        print("START -> get_minimum_seam()")
        h, w = image.shape[:2]
        #energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
        energyfn = self.backward_energy
        #energyfn = self.forward_energy
        M = energyfn(image)

        if mask is not None:
            M[np.where(mask > self.MASK_THRESHOLD)] = self.ENERGY_MASK_CONST

        # give removal mask priority over protective mask by using larger negative value
        if remove_mask is not None:
            M[np.where(remove_mask > self.MASK_THRESHOLD)] = -self.ENERGY_MASK_CONST * 100

        backtrack = np.zeros_like(M, dtype=np.int)

        print("START -> populate dp matrix")
        # populate DP matrix
        for i in range(1, h):
            for j in range(0, w):
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i-1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        print("END -> populate dp matrix")

        # backtrack to find path
        seam_idx = []
        boolmask = np.ones((h, w), dtype=np.bool)
        j = np.argmin(M[-1])
        for i in range(h-1, -1, -1):
            boolmask[i, j] = False
            seam_idx.append(j)
            j = backtrack[i, j]

        seam_idx.reverse()

        print("END -> get_minimum_seam()")

        return np.array(seam_idx), boolmask


    # MAIN FUNCTION FOR SEAM CARVING
    def seams_removal(self, image, num_remove, mask=None, vis=False, rot=False):
        print("START -> seams_removal")
        print("num_remove value:", num_remove)
        x = 0 
        for x in range(num_remove):
            print(x)
            seam_idx, boolmask = self.get_minimum_seam(image, mask)
            #if vis:
            #    visualize(image, boolmask, rotate=rot)
            image = self.remove_seam(image, boolmask)
            if mask is not None:
                mask = self.remove_seam_grayscale(mask, boolmask)

        print("END -> seams_removal")
        return image, mask


    def seams_insertion(self, im, num_add, mask=None, vis=False, rot=False):
        print("START -> seams_insertion")
        print("num_add value:", num_add)
        seams_record = []
        temp_im = im.copy()
        temp_mask = mask.copy() if mask is not None else None
        x = 0
        for x in range(num_add):
            print(x)
            seam_idx, boolmask = self.get_minimum_seam(temp_im, temp_mask)
            #if vis:
            #    visualize(temp_im, boolmask, rotate=rot)

            seams_record.append(seam_idx)
            temp_im = self.remove_seam(temp_im, boolmask)
            if temp_mask is not None:
                temp_mask = self.remove_seam_grayscale(temp_mask, boolmask)

        seams_record.reverse()

        for _ in range(num_add):
            seam = seams_record.pop()
            print("enter add seam")
            im = self.add_seam(im, seam)
            #if vis:
            #    visualize(im, rotate=rot)
            if mask is not None:
                mask = self.add_seam_grayscale(mask, seam)

            # update the remaining seam indices
            for remaining_seam in seams_record:
                remaining_seam[np.where(remaining_seam >= seam)] += 2         
        
        print("END -> seams_insertion")
        return im, mask


    def seam_carve(self, im, dy, dx, mask=None, vis=False):
        #im = im.astype(np.float64)
        h, w = im.shape[:2]
        print(im)
        
        assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

        if mask is not None:
            mask = mask.astype(np.float64)

        output = im

        if dx < 0:
            print("hello:")
            print(output.astype(np.float64))
            output, mask = self.seams_removal(output.astype(np.float64), -dx, mask, vis)

        elif dx > 0:
            output, mask = self.seams_insertion(output.astype(np.float64), dx, mask, vis)

        if dy < 0:
            output = self.rotate_image(output.astype(np.float64), True)
            if mask is not None:
                mask = self.rotate_image(mask, True)
            output, mask = self.seams_removal(output.astype(np.float64), -dy, mask, vis, rot=True)
            output = self.rotate_image(output.astype(np.float64), False)

        elif dy > 0:
            output = self.rotate_image(output.astype(np.float64), True)
            if mask is not None:
                mask = self.rotate_image(mask, True)
            output, mask = self.seams_insertion(output.astype(np.float64), dy, mask, vis, rot=True)
            output = self.rotate_image(output.astype(np.float64), False)

        print("times removed/added: ", self.ctr )

        return output

    def object_removal(self, im, dy, dx, rmask, mask=None, vis=False, horizontal_removal=False):
        im = im.astype(np.float64)
        rmask = rmask.astype(np.float64)
        if mask is not None:
            mask = mask.astype(np.float64)
        output = im

        h, w = im.shape[:2]

        if horizontal_removal:
            output = self.rotate_image(output, True)
            rmask = self.rotate_image(rmask, True)
            if mask is not None:
                mask = self.rotate_image(mask, True)
        #print("this is the rmask")
        #print(rmask)
        #print("len:", np.where(rmask > self.MASK_THRESHOLD))
        #print("len:", len(np.where(rmask > self.MASK_THRESHOLD)))
        #print(len(rmask))
        #print(np.count_nonzero(rmask))
        while len(np.where(rmask > self.MASK_THRESHOLD)[0]) > 0:
            seam_idx, boolmask = self.get_minimum_seam(output, mask, rmask)
            self.maskctr += 1
            print("mask ctr: ", self.maskctr)
            #if vis:
            #    visualize(output, boolmask, rotate=horizontal_removal)            
            output = self.remove_seam(output, boolmask)
            rmask = self.remove_seam_grayscale(rmask, boolmask)
            if mask is not None:
                mask = self.remove_seam_grayscale(mask, boolmask)

        print("total mask ctr: ", self.maskctr )

        num_add = (h if horizontal_removal else w) - output.shape[1]
        output, mask = self.seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)
        if horizontal_removal:
            output = self.rotate_image(output, False)

        if dx < 0:
            print("hello:")
            print(output.astype(np.float64))
            output, mask = self.seams_removal(output.astype(np.float64), -dx, mask, vis)

        elif dx > 0:
            output, mask = self.seams_insertion(output.astype(np.float64), dx, mask, vis)

        if dy < 0:
            output = self.rotate_image(output.astype(np.float64), True)
            if mask is not None:
                mask = self.rotate_image(mask, True)
            output, mask = self.seams_removal(output.astype(np.float64), -dy, mask, vis, rot=True)
            output = self.rotate_image(output.astype(np.float64), False)

        elif dy > 0:
            output = self.rotate_image(output.astype(np.float64), True)
            if mask is not None:
                mask = self.rotate_image(mask, True)
            output, mask = self.seams_insertion(output.astype(np.float64), dy, mask, vis, rot=True)
            output = self.rotate_image(output.astype(np.float64), False)

        print("times removed/added: ", self.ctr)

        return output  

     # image, vertical seams (dx), horizontal seams(dy), retain mask, remove mask

    def __call__(self, image: typing.Union[str, np.ndarray], vertical_seams: int = 0, horizontal_seams: int = 0, retain_mask: typing.Union[str, np.ndarray] = None, remove_mask: typing.Union[str, np.ndarray] = None) -> np.ndarray:
       
        #print(image.shape)
        self.ctr = 0
        self.maskctr = 0

        if(len(image.shape)==2):
            self.isgray = True
            image = image.astype('float32') 
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            self.isgray = False

        h, w = image.shape[:2]
        if self.SHOULD_DOWNSIZE and w > self.DOWNSIZE_WIDTH:
            image = self.resize(image, width=self.DOWNSIZE_WIDTH)
            print("resized!")
            if retain_mask is not None:
                retain_mask = self.resize(retain_mask, width=self.DOWNSIZE_WIDTH, mask = True)
            if remove_mask is not None:
                remove_mask = self.resize(remove_mask, width=self.DOWNSIZE_WIDTH, mask = True)

            if vertical_seams is not None:

                if vertical_seams > 0:
                    vertical_seams = int(min(image.shape[:2]) * 0.25)

                elif vertical_seams < 0:
                    vertical_seams = -int(min(image.shape[:2]) * 0.25)

            if horizontal_seams is not None:
                if horizontal_seams > 0:
                    horizontal_seams = int(min(image.shape[:2]) * 0.25)

                elif horizontal_seams < 0:
                    horizontal_seams = -int(min(image.shape[:2]) * 0.25)
        

        if remove_mask is not None:
            assert remove_mask is not None
            output = self.object_removal(image, horizontal_seams, vertical_seams, remove_mask, retain_mask)
            #output = self.seam_carve(output, vertical_seams, horizontal_seams, retain_mask)
            #cv2.imwrite(OUTPUT_NAME, output)
        else:
            assert vertical_seams is not None and horizontal_seams is not None
            output = self.seam_carve(image, horizontal_seams, vertical_seams, retain_mask)
            #cv2.imwrite(OUTPUT_NAME, output)

        output = cv2.normalize(output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if self.isgray is True:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        return output