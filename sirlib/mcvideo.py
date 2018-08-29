""" video import wrapper for legacy pre-generated video files """
import struct
import numpy as np
import h5py


class MCVideo(object):
    """An object to abstractify pre-generated multi-channel video"""

    def __init__(self, pix_path, mod_path, gt_path):
        self.pix_path = pix_path
        self.mod_path = mod_path
        self.gt_path = gt_path

        # load all information from the ground-truth file
        self.load_gt(gt_path)
        self.pix_read_size = self.width*self.height * 8
        self.mod_read_size = self.width*self.height*self.n_channels*8

        self.mod_file = open(mod_path, 'rb')

    def __del__(self):
        pass
        # self.pix_file.close()
        # self.mod_file.close()

    def load_gt(self, gt_path):
        """ load ground truth """
        with h5py.File(gt_path) as gt_file:
            self.g_t = np.array(gt_file.get('gt/save_gt'))
            self.n_levels = int(gt_file.get('numLevels')[0][0])
            self.n_orien = int(gt_file.get('numOrien')[0][0])
            self.height = int(gt_file.get('video_height')[0][0])
            self.width = int(gt_file.get('video_width')[0][0])
            self.length = int(gt_file.get('video_length')[0][0])
        self.n_channels = self.n_levels * self.n_orien

    def get_pix_frame(self, frame_num):
        """Retrieve pixel-domain frame by number"""
        if frame_num > self.length - 1:
            return []

        seek_pos = self.pix_read_size * frame_num     # zero indexed
        with open(self.pix_path, 'rb') as pix_file:
            pix_file.seek(seek_pos)
            block = pix_file.read(self.pix_read_size)

        fmt = '{}d'.format(int(len(block)/8))
        data = np.array(struct.unpack(fmt, block))
        data = np.multiply(data, 255).astype(np.uint8)
        data = data.reshape(self.width, self.height)
        data = np.transpose(data)
        return data

    def get_mod_frame(self, frame_num):
        """Retrieve modulation-domain frame by number"""
        if frame_num > self.length - 1:
            return []

        seek_pos = self.mod_read_size * frame_num     # zero indexed
        with open(self.mod_path, 'rb') as mod_file:
            mod_file.seek(seek_pos)
            block = mod_file.read(self.mod_read_size)

        fmt = '{}d'.format(int(len(block)/8))
        data = np.array(struct.unpack(fmt, block))

        data = data.reshape(self.n_levels*self.n_orien,
                            self.width, self.height)
        data = np.transpose(data, (0, 2, 1))
        return data

    def get_gt_center(self, frame_num):
        """get ground-truth center by frame_number"""
        return self.g_t[[1, 0], frame_num]

    def get_gt_ulhc(self, frame_num):
        """get upper-left-hand corner of ground-truth"""
        return (self.g_t[0, frame_num] - self.g_t[2, frame_num],
                self.g_t[1, frame_num] - self.g_t[3, frame_num])

    def get_gt_lrhc(self, frame_num):
        """get lower-right-hand corner of ground-truth"""
        return (self.g_t[0, frame_num] + self.g_t[2, frame_num],
                self.g_t[1, frame_num] + self.g_t[3, frame_num])

    def get_gt_tsize(self, frame_num):
        """get template size"""
        return 2*self.g_t[2, frame_num], 2*self.g_t[3, frame_num]
