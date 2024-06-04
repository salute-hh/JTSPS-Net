import os
import numpy as np
from torch.utils.data import Dataset
import torch
import csv
import math

np.random.seed(1)

class FitnessRepData(Dataset):
    def __init__(self, root_path, video_path, label_path, num_frame, scale, istrain, skeleton_num):
        self.root_path = root_path
        self.video_path = video_path
        self.label_dir = os.path.join(root_path, label_path)
        self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        self.label_dict = get_labels_dict(self.label_dir)
        self.num_frame = num_frame
        self.skeleton_num = skeleton_num
        self.istrain = istrain
        self.scale = scale

    def __getitem__(self, inx):
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.root_path, self.video_path, video_file_name)

        if video_file_name in self.label_dict.keys():
            video_rd = VideoRead(file_path, num_frames=self.num_frame, scale=self.scale, skeleton_num=self.skeleton_num, istrain=self.istrain)
            time_points = self.label_dict[video_file_name]
            video_tensor, time_points = video_rd.crop_frame(time_points)
            video_frame_length = video_rd.frame_length
            label, map = preprocess(video_frame_length, time_points, num_frames=self.num_frame)
            label = torch.tensor(label)

            if not self.istrain:
                return [video_tensor, label, map, video_file_name]
            else:
                return [video_tensor, label]
        else:
            print(video_file_name, 'not exist')
            return

    def __len__(self):
        return len(self.video_dir)

class VideoRead:
    def __init__(self, video_path, num_frames, scale, skeleton_num, istrain):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames
        self.istrain = istrain
        self.scale = scale
        self.skeleton_num = skeleton_num

    def get_frame(self):
        frame = np.load(self.video_path)
        frames = []
        for j in range(self.skeleton_num):
            frames_ske = [frame[i][j] for i in range(len(frame))]
            frames.append(frames_ske)
        self.frame_length = len(frames[0])
        return frames

    def crop_frame(self, timespoint):
        frames = self.get_frame()
        frames_tensor_all = []
        for num in range(len(self.scale)):
            frames_tensor = []
            if self.scale[num] <= len(frames[0]):
                for j in range(self.skeleton_num):
                    frames_tensor_temp = [frames[j][i * len(frames[0]) // self.scale[num]] for i in range(self.scale[num])]
                    frames_tensor.append(frames_tensor_temp)
            else:
                for j in range(self.skeleton_num):
                    frames_tensor_temp = [frames[j][i] for i in range(self.frame_length)]
                    frames_tensor_temp += [frames[j][self.frame_length - 1]] * (self.scale[num] - self.frame_length)
                    frames_tensor.append(frames_tensor_temp)

            frames_tensor = np.asarray(frames_tensor, dtype=np.int16)
            frames_tensor = torch.Tensor(frames_tensor)
            frames_tensor /= 1280.0
            frames_tensor = torch.clamp(frames_tensor, min=0)

            if num == 0:
                frames_tensor_all = frames_tensor
            else:
                frames_tensor_all = torch.cat((frames_tensor_all, frames_tensor), 1)

        return frames_tensor_all, timespoint

def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'M' in key and row[key] != '' and row[key] != None]
            labels_dict[row['name']] = cycle
    return labels_dict

def preprocess(video_length, y_frame, num_frames):
    center_label = y_frame
    label_down = [min(math.ceil((float((center_label[i])) / float(video_length)) * num_frames), num_frames - 1) for i in range(len(center_label))]
    y_label = [label_down.count(j) for j in range(num_frames)]
    return y_label, y_label



def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
