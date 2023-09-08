import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "CoDeF", "data_preprocessing", "RAFT", "core"))

import hashlib
import torch
import re
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import argparse
import random

import cv2
import torch.nn.functional as F
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import comfy.model_management
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "CoDeF"))
from opt import get_opts
from train import train_CoDef


class LoadImageSequence:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        image_folder = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name)) and len(os.listdir(os.path.join(input_dir, name))) != 0]
        return {"required":
                    {"image_sequence_folder": (sorted(image_folder), ),
                     "sample_start_idx": ("INT", {"default": 1, "min": 1, "max": 10000}),
                     "sample_frame_rate": ("INT", {"default": 1, "min": 1, "max": 10000}),
                     "n_sample_frames": ("INT", {"default": 1, "min": 1, "max": 10000})
                     }
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_sequence",)
    FUNCTION = "load_image_sequence"

    def load_image_sequence(self, image_sequence_folder, sample_start_idx, sample_frame_rate, n_sample_frames):
        image_path = folder_paths.get_annotated_filepath(image_sequence_folder)
        file_list = sorted(os.listdir(image_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_frames = []
        sample_index = list(range(sample_start_idx-1, len(file_list), sample_frame_rate))[:n_sample_frames]
        for num in sample_index:
            i = Image.open(os.path.join(image_path, file_list[num]))
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image = image.squeeze()
            sample_frames.append(image)
        return (torch.stack(sample_frames), )

    @classmethod
    def IS_CHANGED(s, image_sequence_folder, sample_start_idx, sample_frame_rate, n_sample_frames):
        image_path = folder_paths.get_annotated_filepath(image_sequence_folder)
        m = hashlib.sha256()
        for root, dirs, files in os.walk(image_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image_sequence_folder, sample_start_idx, sample_frame_rate, n_sample_frames):
        if not folder_paths.exists_annotated_filepath(image_sequence_folder):
            return "Invalid image folder: {}".format(image_sequence_folder)
        image_path = folder_paths.get_annotated_filepath(image_sequence_folder)
        resolutions = set()
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            try:
                img = Image.open(file_path)
                resolutions.add(img.size)
                img.close()
            except (IOError, OSError):
                return "Invalid image file: {}".format(file_name)
        if len(resolutions) != 1:
            return "All images must have the same resolution"
        file_list = sorted(os.listdir(image_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_index = list(range(sample_start_idx - 1, len(file_list), sample_frame_rate))
        if len(sample_index) < n_sample_frames:
            return "Not enough frames in sequence"
        return True


class RunRAFT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_sequence": ("IMAGE", ),}
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flow_dir",)
    FUNCTION = "run_raft"

    def viz(self, img, flo, img_name=None):
        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        cv2.imwrite(f'{img_name}', img_flo[:, :, [2, 1, 0]])

    def run_raft(self, image_sequence):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("RAFT", folder_paths.get_output_directory())
        outdir = os.path.join(full_output_folder, f"{filename}_{counter:05}")
        args = argparse.ArgumentParser().parse_args()
        args.model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CoDeF", "data_preprocessing", "RAFT", "models", "raft-sintel.pth")
        args.outdir = os.path.join(outdir, "flow")
        args.small = False
        args.mixed_precision = False
        args.if_mask = False
        args.confidence = True
        args.discrete = False
        args.thres = 4
        args.outdir_conf = os.path.join(outdir, "flow_confidence")
        args.name = "test"
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        DEVICE = comfy.model_management.get_torch_device()
        model = model.module
        model.to(DEVICE)
        model.eval()
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(args.outdir_conf, exist_ok=True)
        with torch.no_grad():
            i = 0
            for image1, image2 in zip(image_sequence[:-1], image_sequence[1:]):
                image1 = torch.unsqueeze(torch.from_numpy((image1.numpy() * 255.0).astype(np.uint8)).permute(2, 0, 1).float(), dim=0).to(comfy.model_management.get_torch_device())
                image2 = torch.unsqueeze(torch.from_numpy((image2.numpy() * 255.0).astype(np.uint8)).permute(2, 0, 1).float(), dim=0).to(comfy.model_management.get_torch_device())
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow_low_, flow_up_ = model(image2, image1, iters=20, test_mode=True)
                flow_1to2 = flow_up.clone()
                flow_2to1 = flow_up_.clone()

                _, _, H, W = image1.shape
                x = torch.linspace(0, 1, W)
                y = torch.linspace(0, 1, H)
                grid_x, grid_y = torch.meshgrid(x, y)
                grid = torch.stack([grid_x, grid_y], dim=0).to(DEVICE)
                grid = grid.permute(0, 2, 1)
                grid[0] *= W
                grid[1] *= H
                grid_ = grid + flow_up.squeeze()

                grid_norm = grid_.clone()
                grid_norm[0, ...] = 2 * grid_norm[0, ...] / (W - 1) - 1
                grid_norm[1, ...] = 2 * grid_norm[1, ...] / (H - 1) - 1

                flow_bilinear_ = F.grid_sample(flow_up_, grid_norm.unsqueeze(0).permute(0, 2, 3, 1),
                                               mode='bilinear', padding_mode='zeros')

                rgb_bilinear_ = F.grid_sample(image2, grid_norm.unsqueeze(0).permute(0, 2, 3, 1), mode='bilinear',
                                              padding_mode='zeros')
                rgb_np = rgb_bilinear_.squeeze().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                cv2.imwrite(f'{args.outdir}/warped.png', rgb_np)

                if args.confidence:
                    grid_2to1 = grid + flow_2to1.squeeze()
                    norm_grid_2to1 = grid_2to1.clone()
                    norm_grid_2to1[0, ...] = 2 * norm_grid_2to1[0, ...] / (W - 1) - 1
                    norm_grid_2to1[1, ...] = 2 * norm_grid_2to1[1, ...] / (H - 1) - 1
                    warped_image2 = F.grid_sample(image1, norm_grid_2to1.unsqueeze(0).permute(0, 2, 3, 1),
                                                  mode='bilinear', padding_mode='zeros')

                    grid_1to2 = grid + flow_1to2.squeeze()
                    norm_grid_1to2 = grid_1to2.clone()
                    norm_grid_1to2[0, ...] = 2 * norm_grid_1to2[0, ...] / (W - 1) - 1
                    norm_grid_1to2[1, ...] = 2 * norm_grid_1to2[1, ...] / (H - 1) - 1
                    warped_image1 = F.grid_sample(warped_image2, norm_grid_1to2.unsqueeze(0).permute(0, 2, 3, 1),
                                                  mode='bilinear', padding_mode='zeros')

                    error = torch.abs(image1 - warped_image1)
                    confidence_map = torch.mean(error, dim=1, keepdim=True)
                    confidence_map[confidence_map < args.thres] = 1
                    confidence_map[confidence_map >= args.thres] = 0

                grid_bck = grid + flow_up.squeeze() + flow_bilinear_.squeeze()
                res = grid - grid_bck
                res = torch.norm(res, dim=0)
                mk = (res < 10) & (flow_up.norm(dim=1).squeeze() > 5)

                pts_src = grid[:, mk]

                pts_dst = (grid[:, mk] + flow_up.squeeze()[:, mk])

                pts_src = pts_src.permute(1, 0).cpu().numpy()
                pts_dst = pts_dst.permute(1, 0).cpu().numpy()
                indx = torch.randperm(pts_src.shape[0])[:30]
                # use cv2 to draw the matches in image1 and image2
                img_new = np.zeros((H, W * 2, 3), dtype=np.uint8)
                img_new[:, :W, :] = image1[0].permute(1, 2, 0).cpu().numpy()
                img_new[:, W:, :] = image2[0].permute(1, 2, 0).cpu().numpy()

                for j in indx:
                    cv2.line(img_new, (int(pts_src[j, 0]), int(pts_src[j, 1])),
                             (int(pts_dst[j, 0]) + W, int(pts_dst[j, 1])), (0, 255, 0), 1)

                cv2.imwrite(f'{args.outdir}/matches.png', img_new)

                np.save(f'{args.outdir}/{i:06d}.npy', flow_up.cpu().numpy())
                if args.confidence:
                    np.save(f'{args.outdir_conf}/{i:06d}_c.npy', confidence_map.cpu().numpy())
                i += 1

                self.viz(image1, flow_up, f'{args.outdir}/flow_up{i:03d}.png')
        return (outdir,)


class TrainMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"CoDeF_Parameters": ("CODEF_PARAMETERS",)}
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ckpts_dir",)
    FUNCTION = "train_multi"

    def train_multi(self, CoDeF_Parameters):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("ckpts", folder_paths.get_output_directory())
        train_parameters = deepcopy(CoDeF_Parameters)
        train_parameters.encode_w = True
        train_parameters.annealed = True
        train_parameters.model_save_path = os.path.join(full_output_folder, f"{filename}_{counter:05}")
        with torch.inference_mode(mode=False):
            train_CoDef(train_parameters)
        return (train_parameters.model_save_path,)


class CoDeFParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_sequence_dir": ("STRING", {"default": None, "multiline": False}),
                     "flow_dir": ("STRING", {"default": None, "multiline": False}),
                     "mask_dir": ("STRING", {"default": None, "multiline": False}),
                     "images_width": ("INT", {"default": 540, "min": 64, "max": 8192, "step": 2}),
                     "images_height": ("INT", {"default": 540, "min": 64, "max": 8192, "step": 2}),
                     "canonical_image_width": ("INT", {"default": 640, "min": 64, "max": 8192, "step": 2}),
                     "canonical_image_height": ("INT", {"default": 640, "min": 64, "max": 8192, "step": 2}),
                     "train_steps": ("INT", {"default": 10000, "min": 100, "max": 100000, "step": 100}),
                     "save_model_iters": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 100}),
                     }
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("CODEF_PARAMETERS",)
    RETURN_NAMES = ("CoDeF_Parameters",)
    FUNCTION = "generate_CoDeF_parameters"

    def generate_CoDeF_parameters(self, image_sequence_dir, flow_dir, mask_dir, images_width, images_height, canonical_image_width, canonical_image_height, train_steps, save_model_iters):
        hparams = get_opts()
        device = comfy.model_management.get_torch_device()
        if device.type == "cuda":
            hparams.gpus = [device.index]

        if canonical_image_width <= images_width or canonical_image_height <= images_height:
            raise Exception("The canonical_wh option in the configuration file should be set with caution, usually a little larger than img_wh, as it determines the field of view of the canonical image.")
        hparams.canonical_wh = [canonical_image_width, canonical_image_height]
        hparams.img_wh = [images_width, images_height]

        if not os.path.exists(image_sequence_dir):
            raise Exception("Invalid image sequence directory: {}".format(image_sequence_dir))
        resolutions = set()
        image_sequence_files = os.listdir(image_sequence_dir)
        for file_name in image_sequence_files:
            file_path = os.path.join(image_sequence_dir, file_name)
            try:
                img = Image.open(file_path)
                resolutions.add(img.size)
                img.close()
            except (IOError, OSError):
                raise Exception("Invalid image file: {}".format(file_name))
        if len(resolutions) != 1:
            raise Exception("All images must have the same resolution")
        hparams.root_dir = image_sequence_dir

        if flow_dir != "":
            flow_flow_dir = os.path.join(flow_dir, "flow")
            flow_flow_confidence_dir = os.path.join(flow_dir, "flow_confidence")
            if os.path.exists(flow_flow_dir) and os.path.exists(flow_flow_confidence_dir):
                flow_files = os.listdir(flow_flow_dir)
                for file in flow_files:
                    _, ext = os.path.splitext(file)
                    if ext not in {".npy", ".png"}:
                        raise Exception("Invalid flow directory: {}".format(flow_dir))
                flow_confidence_files = os.listdir(flow_flow_confidence_dir)
                for file in flow_confidence_files:
                    _, ext = os.path.splitext(file)
                    if ext not in {".npy"}:
                        raise Exception("Invalid flow directory: {}".format(flow_dir))
                if (len(image_sequence_files) != (len(flow_files) // 2)) or (len(image_sequence_files) != (len(flow_confidence_files) + 1)):
                    raise Exception("Flow directory does not match image sequence directory")
                hparams.flow_dir = flow_flow_dir
                hparams.flow_loss = 1
            else:
                raise Exception("Invalid flow directory: {}".format(flow_dir))

        if mask_dir != "":
            mask_0_dir = os.path.join(flow_dir, "masks_0")
            mask_1_dir = os.path.join(flow_dir, "masks_1")
            if os.path.exists(mask_0_dir) and os.path.exists(mask_1_dir):
                masks_0_files = os.listdir(mask_0_dir)
                for file in masks_0_files:
                    _, ext = os.path.splitext(file)
                    if ext not in {".png"}:
                        raise Exception("Invalid mask directory: {}".format(flow_dir))
                masks_1_files = os.listdir(mask_1_dir)
                for file in masks_1_files:
                    _, ext = os.path.splitext(file)
                    if ext not in {".png"}:
                        raise Exception("Invalid mask directory: {}".format(flow_dir))
                if len(image_sequence_files) != len(masks_0_files) or len(image_sequence_files) != len(masks_1_files):
                    raise Exception("Mask directory does not match image sequence directory")
                hparams.mask_dir = [mask_0_dir, mask_1_dir]
                hparams.N_xyz_w = [8, 8]
            else:
                raise Exception("Invalid mask directory: {}".format(mask_dir))

        hparams.num_steps = train_steps
        hparams.save_model_iters = save_model_iters
        return (hparams,)


class ImageSequenceToDirPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_sequence": ("IMAGE",),}
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_sequence_dir",)
    FUNCTION = "image_sequence_to_dir_path"

    def image_sequence_to_dir_path(self, image_sequence):
        filename_prefix ="ComfyUI" + "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_temp_directory(), image_sequence[0].shape[1], image_sequence[0].shape[0])
        os.makedirs(os.path.join(full_output_folder, filename), exist_ok=True)
        for image in image_sequence:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{counter:05}_.png"
            img.save(os.path.join(full_output_folder, filename, file), compress_level=4)
            counter += 1
        return (os.path.join(full_output_folder, filename),)


class TestMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"CoDeF_Parameters": ("CODEF_PARAMETERS",),
                     "ckpts_dir": ("STRING", {"default": None, "multiline": False}),
                     "ckpts_name": ("STRING", {"default": "last.ckpt", "multiline": False}),
                     }
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canonical_image",)
    FUNCTION = "test_multi"

    def test_multi(self, CoDeF_Parameters, ckpts_dir, ckpts_name):
        filename_prefix = "results" + "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_temp_directory())
        if not os.path.exists(ckpts_dir):
            raise Exception("Invalid ckpts directory: {}".format(ckpts_dir))
        ckpts_path = os.path.join(ckpts_dir, ckpts_name)
        if not os.path.exists(ckpts_path):
            raise Exception("Invalid ckpts path: {}".format(ckpts_path))
        test_parameters = deepcopy(CoDeF_Parameters)
        test_parameters.encode_w = True
        test_parameters.test = True
        test_parameters.save_deform = False
        test_parameters.results_path = os.path.join(full_output_folder, filename)
        test_parameters.weight_path = ckpts_path
        with torch.inference_mode(mode=False):
            train_CoDef(test_parameters)
        image_path = os.path.join(test_parameters.results_path, "canonical_0.png")
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


class TestCanonical:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"CoDeF_Parameters": ("CODEF_PARAMETERS",),
                     "ckpts_dir": ("STRING", {"default": None, "multiline": False}),
                     "ckpts_name": ("STRING", {"default": "last.ckpt", "multiline": False}),
                     "canonical_image": ("IMAGE",),
                     }
                }

    CATEGORY = "CoDeF"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_sequence",)
    FUNCTION = "test_canonical"

    def test_canonical(self, CoDeF_Parameters, ckpts_dir, ckpts_name, canonical_image):
        filename_prefix = "results_transformed" + "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_temp_directory())
        if not os.path.exists(ckpts_dir):
            raise Exception("Invalid ckpts directory: {}".format(ckpts_dir))
        ckpts_path = os.path.join(ckpts_dir, ckpts_name)
        if not os.path.exists(ckpts_path):
            raise Exception("Invalid ckpts path: {}".format(ckpts_path))
        test_parameters = deepcopy(CoDeF_Parameters)
        test_parameters.encode_w = True
        test_parameters.test = True
        test_parameters.results_path = os.path.join(full_output_folder, filename)
        test_parameters.weight_path = ckpts_path

        filename_prefix = "canonical" + "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_temp_directory())
        os.makedirs(os.path.join(full_output_folder, filename), exist_ok=True)
        i = 255. * canonical_image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(os.path.join(full_output_folder, filename, "canonical_0.png"), compress_level=4)
        test_parameters.canonical_dir = os.path.join(full_output_folder, filename)
        with torch.inference_mode(mode=False):
            train_CoDef(test_parameters)

        image_path = test_parameters.results_path
        file_list = sorted(os.listdir(image_path),
                           key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_frames = []
        for num in range(len(file_list)):
            i = Image.open(os.path.join(image_path, file_list[num]))
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image = image.squeeze()
            sample_frames.append(image)
        return (torch.stack(sample_frames),)


NODE_CLASS_MAPPINGS = {
    "LoadImageSequence": LoadImageSequence,
    "RunRAFT": RunRAFT,
    "TrainMulti": TrainMulti,
    "CoDeFParameters": CoDeFParameters,
    "ImageSequenceToDirPath": ImageSequenceToDirPath,
    "TestMulti": TestMulti,
    "TestCanonical": TestCanonical
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageSequence": "Load Image Sequence",
    "RunRAFT": "Run RAFT",
    "TrainMulti": "Train a New Model",
    "CoDeFParameters": "CoDeF Parameters",
    "ImageSequenceToDirPath": "Image Sequence to Directory Path",
    "TestMulti": "Test reconstruction",
    "TestCanonical": "Test video translation"
}