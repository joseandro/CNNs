"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_out = int((H - self.kernel_size) / self.stride) + 1
        W_out = int((W - self.kernel_size) / self.stride) + 1
        out = np.zeros((N, C, H_out, W_out))

        for h_axis in range(H_out):
            for w_axis in range(W_out):
                h_start = self.stride * h_axis
                h_end = self.kernel_size + (h_axis * self.stride)

                w_start = self.stride * w_axis
                w_end = self.kernel_size + (w_axis * self.stride)

                # slice out array to max in it
                pool_slice = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, h_axis, w_axis] = np.max(pool_slice, axis=(-1, -2))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        dx = np.zeros_like(x)
        N = x.shape[0]
        C = x.shape[1]
        for h_axis in range(H_out):
            for w_axis in range(W_out):
                h_start = self.stride * h_axis
                h_end = self.kernel_size + (h_axis * self.stride)

                w_start = self.stride * w_axis
                w_end = self.kernel_size + (w_axis * self.stride)

                for n_axis in range(N):
                    for c_axis in range(C):
                        pool_slice = x[n_axis, c_axis, h_start:h_end, w_start:w_end]
                        ind = np.unravel_index(pool_slice.argmax(axis=None), (pool_slice.shape))
                        dx[n_axis, c_axis, h_start:h_end, w_start:w_end][ind] = dout[n_axis, c_axis, h_axis, w_axis]

        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
