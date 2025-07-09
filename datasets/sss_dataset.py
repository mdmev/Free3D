import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision
from einops import rearrange
import math
import torchvision.transforms.functional as F
from utils import camera
from utils.vis_camera import vis_points_images_cameras
import matplotlib.pyplot as plt
from torchvision import transforms

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, (255, 255, 255), 'constant') 

class SSSDataset(Dataset):
    def __init__(self,
                 root_dir: str = '',
                 cfg = None,
                 debug: bool = False) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.debug = debug
        self.validation = getattr(cfg, "validation", False)
        # self.load_view = cfg.load_view

        self.scale = 1.0


        # Gather all object‐folders under root_dir
        self.obj_dirs = [
            self.root_dir / d
            for d in os.listdir(self.root_dir)
            if (self.root_dir / d).is_dir() and d != "white_marble_bowl"
        ]

        print(f"Found {len(self.obj_dirs)} object directories under {self.root_dir}")

        self.json_cache = {}
        self.samples = []  # Each entry is a list of “frame dicts” for one object.

        for obj_dir in self.obj_dirs:
            exclude_list = []
            excl_file = obj_dir / "exclude_list.txt"
            if excl_file.exists():
                with open(excl_file, 'r') as f:
                    exclude_list = [line.strip().replace('.png','') for line in f if line.strip()]
                print(f"  • Excluding {len(exclude_list)} images from {obj_dir.name}")

            json_fname = "transforms_test.json" if self.validation else "transforms_train.json"
            json_path = Path(obj_dir) / json_fname
            if str(json_path) not in self.json_cache:
                with open(json_path, 'r') as f:
                    self.json_cache[str(json_path)] = json.load(f)
            data = self.json_cache[str(json_path)]

            frames = []
            light_position = 0  # Always pick index 0 for “light_position”
            excluded_count = 0
            for frame in data["frames"]:
                stem = Path(frame["file_paths"][light_position]).name
                if stem in exclude_list:
                    excluded_count += 1
                    continue

                png_rel = frame["file_paths"][light_position] + ".png"
                full_png = str(obj_dir / png_rel)

                frames.append({
                    "file_path": full_png,
                    "light_position": frame["light_positions"][light_position],
                    "rotation": frame["rotation"],
                    "transform_matrix": frame["transform_matrix"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "cx": frame["cx"],
                    "cy": frame["cy"],
                    "camera_angle_x": data["camera_angle_x"],
                })
            if excluded_count:
                print(f"  • Excluded {excluded_count} frames from {obj_dir.name}")
            if len(frames) == 0:
                print(f"  • No valid frames found in {obj_dir.name}, skipping.")
                continue
            self.samples.append(frames)

        print(f"============= length of dataset {len(self.samples)} =============")
        print(f"============= Dataset contains {sum(len(frames) for frames in self.samples)} total frames =============")
        self.image_transforms = transforms.Compose([
            SquarePad(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ])

        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)
        
        min_frames = min(len(frames) for frames in self.samples)
        self.load_view = min(cfg.load_view, min_frames)
        print(f"Using {self.load_view} views per object (min frames: {min_frames})")

    def __len__(self):
        return len(self.samples)


    def load_im(self, path: str):
        arr = plt.imread(path)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]

        comp = rgb * alpha + (1.0 - alpha)

        return Image.fromarray((comp * 255).astype(np.uint8))


    def pre_data(self, frames: list):

        import math, torch
        from einops import rearrange

        
        index = range(len(frames)) if self.validation else torch.randperm(len(frames))
        
        imgs            = []
        paths           = []
        w2cs_full       = []
        c2ws_full       = []
        intrinsics_list = []

        for i in index[:self.load_view]:
            f = frames[i]
            paths.append(f["file_path"])

            img_raw = self.load_im(f["file_path"])
            # img_sq  = SquarePad()(img_raw)
            img_t   = self.image_transforms(img_raw)
            imgs.append(img_t.unsqueeze(0))

            W_raw, H_raw = img_raw.size
            S_pad        = max(W_raw, H_raw)
            pad_left     = (S_pad - W_raw) // 2
            pad_top      = (S_pad - H_raw) // 2
            
            # focal in px from horizontal FOV
            fx = 0.5 * W_raw / math.tan(0.5 * f["camera_angle_x"])
            fy = fx * H_raw / W_raw
            # principal point + padding
            cx = f.get("cx", W_raw * 0.5) + pad_left
            cy = f.get("cy", H_raw * 0.5) + pad_top

            # scale from raw square to 256
            scale = 256.0 / S_pad
            fx   *= scale; fy   *= scale
            cx   *= scale; cy   *= scale        

            # normalize into [0,1]
            fxn, fyn = fx / 256.0, fy / 256.0
            cxn, cyn = cx / 256.0, cy / 256.0

            K_norm = torch.tensor([
                [fxn,  0.0, cxn],
                [0.0,  fyn, cyn],
                [0.0,  0.0, 1.0]
            ], dtype=torch.float32)
            intrinsics_list.append(K_norm)

            # --- extrinsics: Blender→repo axis flip, then invert to world→camera ---
            c2w = torch.tensor(np.array(f["transform_matrix"], dtype=np.float32))
            c2w = self.opengl_to_colmap @ c2w
            w2c = torch.linalg.inv(c2w)
            w2cs_full.append(w2c)
            c2ws_full.append(c2w)

        # stack everything
        imgs        = torch.cat(imgs, dim=0)                   # [V,256,256,3]
        w2cs        = torch.stack(w2cs_full, dim=0)            # [V,4,4]
        c2ws        = torch.stack(c2ws_full, dim=0)            # [V,4,4]
        intrinsics  = torch.stack(intrinsics_list, dim=0)      # [V,3,3]

        # normalize camera centers so first view has radius=2.0
        centers = c2ws[:, :3, 3]
        scale_t = 2.0 / centers[0].norm()
        w2cs[:, :3, 3] *= scale_t
        c2ws[:, :3, 3] *= scale_t

        assert imgs.shape[0] == w2cs.shape[0] == intrinsics.shape[0] == c2ws.shape[0]
        return imgs, w2cs, c2ws, intrinsics, paths

    def __getitem__(self, idx: int):

        frames = self.samples[idx]
        imgs, w2cs, c2ws, intrinsics, _ = self.pre_data(frames)

        if self.debug:
            # import pdb
            # pdb.set_trace()
            intrinsics[:, 0, :] *= 256
            intrinsics[:, 1, :] *= 256
            vis_points_images_cameras(w2cs, intrinsics, imgs, frustum_size=0.5, filename=os.path.join('debug_images/',frames[0]['file_path'].split('/')[-3]) + '_camera_ori.html')

        
        data = {
            'images': imgs,                                      # [V, H, W, 3]
            'w2cs': w2cs,                                        # [V, 4, 4]
            'c2ws': c2ws,                                        # [V, 4, 4]
            'intrinsics': intrinsics,                            # [V, 3, 3]
            'filename': frames[0]['file_path'].split('/')[-3]
        }

        return data

    def process_img(self, img):
        img = img.convert("RGB")
        return self.image_transforms(img)