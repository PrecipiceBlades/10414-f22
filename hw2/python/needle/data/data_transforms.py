import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # Flip horizontally by reversing the width dimension (axis 1)
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        
        # Add zero padding to all sides
        # pad_width: ((top, bottom), (left, right), (channels_before, channels_after))
        padded = np.pad(img, 
                       ((self.padding, self.padding), 
                        (self.padding, self.padding), 
                        (0, 0)), 
                       mode='constant', 
                       constant_values=0)
        
        # Calculate crop position
        # shift_x is for height dimension, shift_y is for width dimension
        start_h = self.padding + shift_x
        start_w = self.padding + shift_y
        
        # Crop back to original size
        cropped = padded[start_h:start_h + H, start_w:start_w + W, :]
        
        return cropped
        ### END YOUR SOLUTION
