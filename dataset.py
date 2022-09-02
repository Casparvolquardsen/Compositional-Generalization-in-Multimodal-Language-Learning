import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MultimodalSimulation(Dataset):
    def __init__(self, path, part, visible_objects, different_actions, different_colors, different_objects,
                 exclusive_colors, num_samples, max_frames=16, same_size=True,
                 frame_stride=1, precooked=False, feature_dim=None, transform=None):

        assert isinstance(path, str) and isinstance(part, str)
        assert part in ["training", "validation", "constant-test", "generalization-test"]
        assert isinstance(visible_objects, list) and len(visible_objects) <= 6

        if part == "training":
            max_samples_per_dir = 5000
        elif part == "validation":
            max_samples_per_dir = 2500
        elif part == "constant-test" or part == "generalization-test":
            max_samples_per_dir = 2000
        else:
            raise ValueError("Wrong part parameter. This dataset is not available!")

        self.path = path[:-1] if path[-1] == "/" else path # instead of removesuffix for python version >= 3.9
        self.part = part
        self.visible_objects = visible_objects
        self.different_actions = different_actions
        self.different_colors = different_colors
        self.different_objects = different_objects
        self.exclusive_colors = exclusive_colors
        self.max_frames = max_frames
        self.same_size = same_size
        self.frame_stride = frame_stride
        self.num_sub_dirs = len(self.visible_objects)
        self.num_samples_per_dir = min(num_samples // self.num_sub_dirs, max_samples_per_dir)
        self.num_samples = len(self.visible_objects) * self.num_samples_per_dir
        self.transform = transform
        self.precooked = precooked
        if precooked:
            self.feature_dim = feature_dim
            assert type(feature_dim) == int
        self.LABEL_LENGTH = 19
        self.DICTIONARY = ["put down", "picked up", "pushed left", "pushed right",
                           "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                           "red", "green", "blue", "yellow", "white", "brown"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # label: path/Vi-Cc-Oo/part/sequence_xxxx/label.npy -> one-hot-encoded
        # imgs:  path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.png - frame_eeeeee.png
        # joints:path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.txt - frame_eeeeee.txt

        dir_number = self.visible_objects[item // self.num_samples_per_dir]
        sequence_number = item % self.num_samples_per_dir
        if self.part == "constant-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-test"
        elif self.part == "generalization-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-generalization-test"
        else:
            dir_path = f"{self.path}/V{dir_number}-A{self.different_actions}-C{self.different_colors}-O{self.different_objects}{'-X' if self.exclusive_colors else ''}/{self.part}"
        sequence_path = f"{dir_path}/sequence_{sequence_number:04d}"

        # reading sentence out of label.npy - NOT one-hot-encoded
        label = torch.from_numpy(np.load(f"{sequence_path}/label.npy")).to(dtype=torch.long)

        joint_paths = glob.glob(f"{sequence_path}/joints_*.npy")

        if self.precooked:
            frame_paths = glob.glob(f"{sequence_path}/resnet18_layer4_features_{self.feature_dim}_frame_*.pt")
        else:
            frame_paths = glob.glob(f"{sequence_path}/frame_*.png")

        # glob returns unordered
        joint_paths.sort()
        frame_paths.sort()

        num_frames = len(frame_paths)
        assert num_frames > 0

        # Only take the last max_frames=32 frames.
        # Scenes with more than 32 frames had one or more unsuccessful actions in the beginning,
        # or a long way of the arm to the objects position.
        # The last frames show the successful action, typically less than 32 frames long.
        # Only take frames after last reset of robot arm.

        if num_frames >= self.max_frames:
            frame_numbers = list(range(num_frames - self.max_frames, num_frames, self.frame_stride))
        else:
            frame_numbers = list(range(0, num_frames, self.frame_stride))

        num_frames = len(frame_numbers)

        if self.same_size:
            num_frames = self.max_frames // self.frame_stride

        joints = torch.zeros(num_frames, 6, dtype=torch.float32)
        if self.precooked:
            frames = torch.zeros(num_frames, self.feature_dim, dtype=torch.float32)
        else:
            frames = torch.zeros(num_frames, 3, 224, 398, dtype=torch.float32)  # img shape (3, 224, 398)

        for i, frame_number in enumerate(frame_numbers):
            joint_path = joint_paths[frame_number]
            frame_path = frame_paths[frame_number]

            joints[i] = torch.from_numpy(np.load(joint_path)).to(torch.float32)

            if self.precooked:
                frames[i] = torch.load(frame_path)
            else:
                frames[i] = read_image(frame_path).to(torch.float32) / 255
                if self.transform is not None:
                    frames[i] = self.transform(frames[i])

        # frames.shape -> (frames, 3, 224, 398)
        # joints.shape -> (frames, 6)
        # label.shape  -> (3) : word tokens
        # all have dtype=torch.float32
        return frames, joints, label

    def get_sentence_string(self, label):
        return f"{self.DICTIONARY[int(label[0])]} {self.DICTIONARY[int(label[1])]} {self.DICTIONARY[int(label[3])]}"
