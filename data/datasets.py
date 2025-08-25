import os, glob
from torch.utils.data import Dataset
from .data_utils import *
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class SJTUTestliverDataset(Dataset):
    def __init__(self, A_path, transforms):
        self.dir_A = os.path.join(A_path, 'MR/')
        self.dir_A_seg = os.path.join(A_path, 'mr_seg/')
        self.dir_B = os.path.join(A_path, 'CT/')
        self.dir_B_seg = os.path.join(A_path, 'ct_seg/')

        self.A_paths = sorted(glob.glob(self.dir_A+'/*.nii'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.transforms = transforms

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_name = os.path.basename(A_path)
        A_seg_path = os.path.join(self.dir_A_seg,A_name) #.replace('.nii','.nii.gz'))

        B_path = os.path.join(self.dir_B,A_name)
        B_seg_path = os.path.join(self.dir_B_seg,A_name)

        A_vol = nib.load(A_path)
        B_vol = nib.load(B_path)
        A_seg = nib.load(A_seg_path)
        B_seg = nib.load(B_seg_path)

        A_data = A_vol.get_fdata()
        A_seg_data = A_seg.get_fdata()
        B_data = B_vol.get_fdata()
        B_seg_data = B_seg.get_fdata()

        x, y = A_data[None, ...], B_data[None, ...]
        x_seg, y_seg = A_seg_data[None, ...], B_seg_data[None, ...]

        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y = torch.from_numpy(x).permute(0,3,1,2), torch.from_numpy(y).permute(0,3,1,2)
        x_seg, y_seg = torch.from_numpy(x_seg).permute(0,3,1,2), torch.from_numpy(y_seg).permute(0,3,1,2)
        return x, y,x_seg,y_seg

    def __len__(self):
        return self.A_size














if __name__=='__main__':
    # train_dir = '/home/ubuntu/run_program/regdata/intra/train/TT/'
    # train_composed = transforms.Compose([
    #     # trans.RandomRotion(20),
    #     trans.NumpyType((np.float32, np.float32)),
    # ])  # trans.RandomFlip(0),
    # train_set = SJTUliverDataset(train_dir, transforms=train_composed)
    def display_slices(data, slice_range):
        """
        显示指定范围内的多个切片。
        slice_range: (start, end) 范围，指定显示的切片索引范围
        """
        start, end = slice_range

        # 确保范围合法
        if start < 0 or end >= data.shape[2] or start > end:
            raise ValueError(f"Invalid slice range: ({start}, {end})")

        num_slices = end - start + 1

        # 创建子图，按2x4的网格显示8个切片
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))  # 2行4列的布局
        axes = axes.flatten()  # 将二维轴数组展平为一维数组

        # 显示指定范围的切片
        for i in range(num_slices):
            slice_index = start + i
            slice_data = data[slice_index,:, :]  # 获取Z轴方向的切片

            axes[i].imshow(slice_data.T, cmap="gray", origin="lower")
            axes[i].set_title(f"Slice {slice_index}")
            axes[i].axis('off')  # 关闭坐标轴

        plt.tight_layout()
        plt.show()




