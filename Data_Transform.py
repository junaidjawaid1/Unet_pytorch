import numpy as np
class transformation:

    def flip(self, x_in, flip_axis):

        flipped= np.flip(x_in, flip_axis)
        return flipped
    
    def rotation(self, x_in, plane):

        rotated=np.rot90(x_in, k=1, axes=plane)
        return rotated



