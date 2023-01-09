# Copyright 2004-present Facebook. All Rights Reserved.
#
# Author: Daniel DeTone (ddetone)

import math
import warnings
from typing import NamedTuple, Optional

import kornia
import torch


def truncated_normal_(N: int, mean: float = 0.0, std: float = 1):
    """Draws N samples from a truncated normal distribution ~N_t(mean,std).

    :param N: number of samples to draw
    :param mean: mean of distribution
    :param std: standard deviation of distribution

    """
    tensor = torch.FloatTensor(N)
    tmp = tensor.new_empty(tensor.shape + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def perspective_grid(
    coeffs: torch.Tensor,
    size: torch.Size,
    align_corners: bool = False,
    eps: float = 1e-9,
):
    """
    Generate a grid of u,v sample locations from a perspective transform.
    Acts similarly to torch's affine_grid, but allows for 3x3 perspective warps.

    :param coeffs: Bx8 input batch of perspective transform matrices
    (flattened first 8 elements of 3x3 matrix, assuming last element = 1)
    :param size: NxCxHxW the target output size
    :param align_corners: if True, consider -1 and 1 to refer to the centers
    of the corner pixels
    :param eps: factor to prevent division by zero

    """
    B, H, W = int(size[0]), int(size[-2]), int(size[-1])
    base_grid = torch.empty(B, H, W, 3).to(coeffs)
    # Generate base_grid [-1,1], taking into account align_corners.
    step_x = torch.linspace(-1, 1, W)
    step_y = torch.linspace(-1, 1, H)
    if align_corners is False:
        step_x = step_x * (W - 1) / W
        step_y = step_y * (H - 1) / H
    base_grid[..., 0].copy_(step_x)
    base_grid[..., 1].copy_(step_y.unsqueeze(-1))
    base_grid[..., 2].fill_(1)
    base_grid = base_grid.reshape(B, -1, 3)
    # Generate homography matrix.
    h_mat = torch.ones(B, 9).to(coeffs)
    h_mat[:, :-1] = coeffs
    h_mat = h_mat.reshape(B, 3, 3).transpose(1, 2)
    # Apply homography warp, then normalize by homogenous coordinate.
    grid = base_grid @ h_mat
    grid[:, :, 0] = grid[:, :, 0] / (grid[:, :, 2] + eps)
    grid[:, :, 1] = grid[:, :, 1] / (grid[:, :, 2] + eps)
    grid = grid[:, :, :2].reshape(B, H, W, 2)
    return grid


def check_input(inp: torch.Tensor):
    """Makes sure that input tensor meets input specifications."""
    if inp.dim() != 4:
        raise ValueError("Input tensor must have 4 dims (B, C, H, W)")
    if inp.shape[1] != 1 and inp.shape[1] != 3:
        raise ValueError("Only 1 and 3 channel inputs are supported")
    if inp.dtype != torch.float32:
        raise TypeError("Only torch.float32 supported.")
    if inp.min() < 0.0 or inp.max() > 1.0:
        raise ValueError("Image must be normalized between [0,1].")
    return


class GeoAugParam(NamedTuple):
    min: Optional[float] = 0.0
    max: Optional[float] = 0.0


class RandomGeoAug(object):
    """
    Applies batch-wise geometric augmentations to images.
    Requires that input/output tensors are sized BxCxHxW.
    """

    def __init__(
        self,
        rotate_param: Optional[GeoAugParam] = None,
        scale_param: Optional[GeoAugParam] = None,
        translate_x_param: Optional[GeoAugParam] = None,
        translate_y_param: Optional[GeoAugParam] = None,
        shear_x_param: Optional[GeoAugParam] = None,
        shear_y_param: Optional[GeoAugParam] = None,
        perspective_param: Optional[GeoAugParam] = None,
    ):
        self.rotate_param = (
            GeoAugParam(min=-180, max=180) if rotate_param is None else rotate_param
        )
        self.scale_param = (
            GeoAugParam(min=0.5, max=2.0) if scale_param is None else scale_param
        )
        self.translate_x_param = (
            GeoAugParam(min=-0.3, max=0.3)
            if translate_x_param is None
            else translate_x_param
        )
        self.translate_y_param = (
            GeoAugParam(min=-0.3, max=0.3)
            if translate_y_param is None
            else translate_y_param
        )
        self.shear_x_param = (
            GeoAugParam(min=-20, max=20) if shear_x_param is None else shear_x_param
        )
        self.shear_y_param = (
            GeoAugParam(min=-20, max=20) if shear_y_param is None else shear_y_param
        )
        self.perspective_param = (
            GeoAugParam(min=-0.3, max=0.3)
            if perspective_param is None
            else perspective_param
        )
        self.eps = 1e-9

        self.param_names = []
        for key in dir(self):
            if "_param" in key:
                self.param_names.append(key)

    def __repr__(self):
        ret = self.__class__.__name__ + "(\n"
        for key in dir(self):
            if "_param" in key:
                ret += ("  " + key + " = " + getattr(self, key).__repr__()) + ",\n"
        ret += ")"
        return ret

    def _get_perspective_matrix(
        self,
        angle: torch.Tensor,
        translate: torch.Tensor,
        scale: torch.Tensor,
        shear: torch.Tensor,
        delta_corners: torch.Tensor,
    ):
        B = angle.shape[0]
        # Thus, the inverse is M^-1 = RSS^-1 * T^-1
        rot = torch.deg2rad(-angle)
        sx, sy = [torch.deg2rad(s) for s in shear.t()]
        tx, ty = 2 * translate[:, 0], 2 * translate[:, 1]
        # RSS without scaling
        a = torch.cos(rot - sy) / torch.cos(sy)
        b = -torch.cos(rot - sy) * torch.tan(sx) / torch.cos(sy) - torch.sin(rot)
        c = torch.sin(rot - sy) / torch.cos(sy)
        d = -torch.sin(rot - sy) * torch.tan(sx) / torch.cos(sy) + torch.cos(rot)
        # Inverted rotation matrix with scale and shear
        zeros = torch.zeros_like(a)
        matrix = torch.stack([d, -b, zeros, -c, a, zeros], dim=-1)
        matrix = torch.stack([x / scale for x in matrix.t()], dim=-1)
        # Apply inverse of translation and of center translation: RSS^-1 * T^-1
        matrix[:, 2] += matrix[:, 0] * (-tx) + matrix[:, 1] * (-ty)
        matrix[:, 5] += matrix[:, 3] * (-tx) + matrix[:, 4] * (-ty)
        matrix = torch.cat([matrix, matrix.new_zeros((B, 3))], dim=1)
        matrix[:, -1] = 1.0
        matrix = matrix.reshape(B, 3, 3)
        # Generate non-affine effect by perturbing the four corners.
        start = torch.ones(B, 4, 2).to(a)
        end = start - delta_corners  # Positive delta takes corners inwards.
        corner_signs = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]).to(a)
        corner_signs = corner_signs.repeat(B, 1, 1)
        start *= corner_signs
        end *= corner_signs
        # Set up a linear system to solve for the homography.
        pmat = kornia.geometry.transform.get_perspective_transform(start, end)
        # Apply perspective transform first, then affine.
        matrix = pmat @ matrix
        # Normalize such that H[2,2] = 1.
        eps = 1e-9
        matrix /= matrix[:, 2, 2].clone().reshape(B, 1, 1) + eps
        return matrix

    def _generate_transform(self, B: int, device: torch.device):
        rot_range = (self.rotate_param.min, self.rotate_param.max)
        trans_x_range = (self.translate_x_param.min, self.translate_x_param.max)
        trans_y_range = (self.translate_y_param.min, self.translate_y_param.max)
        shear_x_range = (self.shear_x_param.min, self.shear_x_param.max)
        shear_y_range = (self.shear_y_param.min, self.shear_y_param.max)
        scale_range = (self.scale_param.min, self.scale_param.max)
        pscale_range = (self.perspective_param.min, self.perspective_param.max)
        # Randomly generate parameters that define the perspective transform.
        rot = torch.empty(B, device=device).uniform_(rot_range[0], rot_range[1])
        trans_x = torch.empty(B, device=device).uniform_(
            trans_x_range[0], trans_x_range[1]
        )
        trans_y = torch.empty(B, device=device).uniform_(
            trans_y_range[0], trans_y_range[1]
        )
        trans_xy = torch.stack([trans_x, trans_y], dim=-1)
        shear_x = torch.empty(B, device=device).uniform_(
            shear_x_range[0], shear_x_range[1]
        )
        shear_y = torch.empty(B, device=device).uniform_(
            shear_y_range[0], shear_y_range[1]
        )
        shear_xy = torch.stack([shear_x, shear_y], dim=-1)
        scale = torch.empty(B, device=device).uniform_(scale_range[0], scale_range[1])
        delta_corners = torch.empty(B, 4, 2, device=device).uniform_(
            pscale_range[0], pscale_range[1]
        )
        # Given parameters, generate 3x3 perspective matrices.
        transform = self._get_perspective_matrix(
            rot, trans_xy, scale, shear_xy, delta_corners
        )
        return transform

    def _apply_perspective(
        self, img: torch.Tensor, transform: torch.Tensor, align_corners: bool = True
    ):
        coeffs = transform.reshape(-1, 9)[:, :-1]
        grid2 = perspective_grid(
            coeffs, img.shape, align_corners=align_corners, eps=self.eps
        )
        img2 = torch.nn.functional.grid_sample(img, grid2, align_corners=align_corners)
        return img2

    def _denormalize_transform(self, transform: torch.Tensor, H: int, W: int):
        B = int(transform.shape[0])
        ones = transform.new_ones((B, 1))
        zeros = transform.new_zeros((B, 1))
        scaleW = (1.0 / ((W - 1.0) / 2.0)) * ones
        scaleH = (1.0 / ((H - 1.0) / 2.0)) * ones
        T = torch.stack(
            [scaleW, zeros, -ones, zeros, scaleH, -ones, zeros, zeros, ones], dim=-1
        ).reshape(-1, 3, 3)
        transform2 = torch.inverse(T) @ transform @ T
        transform2 /= transform2[:, 2, 2].reshape(B, 1, 1).clone() + self.eps
        return transform2

    def set_all_identity(self):
        for name in self.param_names:
            if "scale_param" in name:
                val = 1.0
            else:
                val = 0.0
            setattr(self, name, GeoAugParam(min=val, max=val))

    def forward(self, inp, return_transform=False, normalize_returned_transform=False):
        """
        Runs the Random Geometric Augmentation on a batch of images.

        :param inp: BxCxHxW batch of input images
        :param return_transform: if True, also return the transform used to
        warp the inp (in unnormalize image coordinates by default)
        :param normalize_returned_transform: if True, the returned transform
        will be in normalized pixel coords (e.g. [-1,-1] is top left pixel,
        [1,1] is bottom right)

        """
        check_input(inp)  # Make sure input is well-formed.
        if self.scale_param.min <= 0.0 or self.scale_param.max <= 0.0:
            warnings.warn(
                "Scale min and max are are too extreme, resulting image might "
                "be black, range should be (0, inf)",
                RuntimeWarning,
            )
        if self.translate_x_param.min < -1.0 or self.translate_x_param.max > 1.0:
            warnings.warn(
                "Translate_x min and max are too extreme, resulting image might"
                "be black, range should be [-1.0, 1.0]",
                RuntimeWarning,
            )
        if self.translate_y_param.min < -1.0 or self.translate_y_param.max > 1.0:
            warnings.warn(
                "Translate_y min and max are too extreme, resulting image "
                "might be black, range should be [-1.0, 1.0]",
                RuntimeWarning,
            )
        if self.shear_x_param.min < -90 or self.shear_x_param.max > 90:
            warnings.warn(
                "Shear_x min and max are too extreme, resulting image might "
                "be black, range should be [-90, 90]",
                RuntimeWarning,
            )
        if self.shear_y_param.min < -90 or self.shear_y_param.max > 90:
            warnings.warn(
                "Shear_x min and max are too extreme, resulting image might "
                "be black, range should be [-90, 90]",
                RuntimeWarning,
            )
        if self.perspective_param.min < -1.0 or self.perspective_param.max < -1.0:
            warnings.warn(
                "Perspective min and max are too extreme, resulting image "
                "might be black, range should be [-1, inf]",
                RuntimeWarning,
            )

        # Forcing this to be generated on CPU because it's faster.
        inv_transform = self._generate_transform(
            inp.shape[0], device=torch.device("cpu")
        )
        inv_transform = inv_transform.to(inp)
        identity = (
            torch.eye(3).unsqueeze(0).repeat(inp.shape[0], 1, 1).to(inv_transform)
        )
        if torch.allclose(identity, inv_transform):  # Shortcut operation.
            inp2 = inp
            transform = inv_transform
        else:
            inp2 = self._apply_perspective(inp, inv_transform, align_corners=True)
            transform = torch.inverse(inv_transform)
            transform /= transform[:, 2, 2].reshape(-1, 1, 1).clone() + self.eps
        if not return_transform:
            return inp2
        else:
            if normalize_returned_transform:
                return inp2, transform
            else:
                H, W = inp.shape[-2], inp.shape[-1]
                denorm_transform = self._denormalize_transform(transform, H, W)
                return inp2, denorm_transform


class PhotoAugParam(NamedTuple):
    prob: Optional[float] = 0.2
    mag: Optional[float] = 0.5
    fix_pos: Optional[bool] = False


class RandomPhotoAug(object):
    """
    Applies batch-wise photometric distortions to images.
    Requires that input/output tensors are sized BxCxHxW.

    """

    def __init__(
        self,
        contrast_param: Optional[PhotoAugParam] = None,
        sharpen_param: Optional[PhotoAugParam] = None,
        exposure_param: Optional[PhotoAugParam] = None,
        gamma_param: Optional[PhotoAugParam] = None,
        gaussian_smooth_param: Optional[PhotoAugParam] = None,
        motion_blur_param: Optional[PhotoAugParam] = None,
        shadow_highlight_param: Optional[PhotoAugParam] = None,
        gaussian_noise_param: Optional[PhotoAugParam] = None,
        salt_and_pepper_param: Optional[PhotoAugParam] = None,
    ):
        self.contrast_param = (
            PhotoAugParam() if contrast_param is None else contrast_param
        )
        self.sharpen_param = PhotoAugParam() if sharpen_param is None else sharpen_param
        self.exposure_param = (
            PhotoAugParam() if exposure_param is None else exposure_param
        )
        self.gamma_param = PhotoAugParam() if gamma_param is None else gamma_param
        self.gaussian_smooth_param = (
            PhotoAugParam() if gaussian_smooth_param is None else gaussian_smooth_param
        )
        self.motion_blur_param = (
            PhotoAugParam() if motion_blur_param is None else motion_blur_param
        )
        self.shadow_highlight_param = (
            PhotoAugParam()
            if shadow_highlight_param is None
            else shadow_highlight_param
        )
        # Fix the order in which the last two are applied by default.
        self.gaussian_noise_param = (
            PhotoAugParam(fix_pos=True)
            if gaussian_noise_param is None
            else gaussian_noise_param
        )
        self.salt_and_pepper_param = (
            PhotoAugParam(fix_pos=True)
            if salt_and_pepper_param is None
            else salt_and_pepper_param
        )

        # Sets the default order.
        self.fn_names = [
            "contrast",
            "sharpen",
            "exposure",
            "gamma",
            "gaussian_smooth",
            "motion_blur",
            "shadow_highlight",
            "gaussian_noise",
            "salt_and_pepper",
        ]

        # Each fn_name must be a method.
        for name in self.fn_names:
            assert hasattr(self, name) and callable(getattr(self, name))

        self.param_names = []
        for key in dir(self):
            if "_param" in key:
                self.param_names.append(key)

    def __repr__(self):
        ret = self.__class__.__name__ + "(\n"
        for key in dir(self):
            if key.endswith("_param"):
                ret += ("  " + key + " = " + getattr(self, key).__repr__()) + ",\n"
        ret += ")"
        return ret

    def set_all_probs(self, val: int):
        for name in self.param_names:
            param = getattr(self, name)
            setattr(self, name, param._replace(prob=val))

    def set_all_mags(self, val: int):
        for name in self.param_names:
            param = getattr(self, name)
            setattr(self, name, param._replace(mag=val))

    def set_all_fix_pos(self, val: bool):
        for name in self.param_names:
            param = getattr(self, name)
            setattr(self, name, param._replace(fix_pos=val))

    def contrast(self, inp: torch.Tensor, mag: float = 0.5):
        B = inp.shape[0]
        # Kind of arbitrary, asymmetric normal distribution with
        # has median=1 and doesn't have clipping artifacts.
        scales = truncated_normal_(B, mean=1.0, std=(mag * 0.9 * 0.5 + 1e-9))
        scales = torch.where(scales > 1.0, scales + (scales - 1.0) * 3.0, scales)
        scales = scales.to(inp)
        out = inp - 0.5
        out = out * scales.reshape(-1, 1, 1, 1)
        out = out + 0.5
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def sharpen(self, inp: torch.Tensor, mag: float = 0.5):
        B, C, H, W = inp.shape
        scale = truncated_normal_(B, mean=0, std=mag).abs()
        val = (1.0 / 5.0) + scale
        kernel = torch.zeros(B, 3, 3).to(inp)
        kernel[:, 1, 1] = val * 5
        kernel[:, 0, 1] = -((5 * val - 1.0) / 4.0)
        kernel[:, 1, 0] = -((5 * val - 1.0) / 4.0)
        kernel[:, 1, 2] = -((5 * val - 1.0) / 4.0)
        kernel[:, 2, 1] = -((5 * val - 1.0) / 4.0)
        # Use grouped conv to apply different kernel to each batch element.
        kernel = kernel[:, None, ...]
        kernel = torch.repeat_interleave(kernel, C, dim=0)
        inp = inp.reshape(1, B * C, H, W)
        out = torch.nn.functional.conv2d(inp, kernel, padding=1, groups=B * C)
        out = out.reshape(B, C, H, W)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def exposure(self, inp: torch.Tensor, mag: float = 0.5):
        B = inp.shape[0]
        max_delta = mag * 0.9  # Don't let max pixel value go below 0.1.
        new_maxs = truncated_normal_(B, mean=1.0, std=0.5 * max_delta).to(inp)
        out = inp * new_maxs.reshape(-1, 1, 1, 1)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def gamma(self, inp: torch.Tensor, mag: float = 0.5):
        B = inp.shape[0]
        # Kind of arbitrary, asymmetric normal distribution with
        # has median=1 and doesn't have clipping artifacts.
        gam = truncated_normal_(B, mean=1.0, std=mag * 0.9 * 0.5).to(inp)
        gam = torch.where(gam > 1.0, gam + (gam - 1.0) * 3.0, gam)
        out = (inp ** gam.reshape(-1, 1, 1, 1)).clamp(0, 1)
        return out

    def gaussian_smooth(
        self,
        inp: torch.Tensor,
        kmin: int = 1,
        kmax: int = 11,
        smin: float = 1.0,
        smax: float = 5.0,
        mag: float = 0.5,
    ):
        B, C, H, W = inp.shape
        # Maximum possible kernel size.
        kmax = int(kmax * mag) + 1
        if kmax == 1:  # 1x1 gaussian blur does nothing.
            return inp
        half_sizes = torch.randint(size=(B,), low=kmin, high=kmax)  # check kmax
        # To batch the operations, we use the largest kernel size.
        kernel_size = int(half_sizes.max().item() * 2 + 1)
        sigma = mag * torch.FloatTensor(B).uniform_(smin, smax)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.unsqueeze(0).repeat(B, 1, 1, 1)
        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0
        variance = variance.reshape(-1, 1, 1, 1)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1, keepdim=True) / (2 * variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = gaussian_kernel / torch.sum(
            gaussian_kernel, dim=[1, 2, 3], keepdim=True
        )
        kernel = kernel.to(inp)
        # Use grouped conv to apply different kernel to each batch element.
        kernel = kernel.reshape(B, 1, kernel_size, kernel_size)
        kernel = torch.repeat_interleave(kernel, C, dim=0)
        kernel = kernel.reshape(B * C, 1, kernel_size, kernel_size)
        inp = inp.reshape(1, B * C, H, W)
        out = torch.nn.functional.conv2d(
            inp, kernel, padding=int(half_sizes.max()), groups=B * C
        )
        out = out.reshape(B, C, H, W)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def motion_blur(
        self, inp: torch.Tensor, mag: float = 0.5, kmin: int = 1, kmax: int = 11
    ):
        B, C, H, W = inp.shape
        # Maximum possible kernel size.
        kmax = int(kmax * mag) + 1
        if kmax == 1:  # 1x1 gaussian blur does nothing.
            return inp
        half_sizes = torch.randint(size=(B,), low=kmin, high=kmax)  # check kmax
        # To batch the operations, we use the largest kernel size.
        kernel_sizes = half_sizes * 2 + 1
        kernel_size = int(kernel_sizes.max().item())
        # Generate a different motion blur kernel for each batch.
        kernel = torch.zeros((B, kernel_size, kernel_size)).to(inp)
        # TODO(dd): figure out how to vectorize this.
        for i, ks in enumerate(kernel_sizes):
            off = torch.div((kernel_size - ks), 2, rounding_mode="floor")
            start = int(off)
            end = int(kernel_size - off)
            kh = torch.div(kernel_size, 2, rounding_mode="floor")
            kernel[i, kh, start:end] = torch.ones((1, ks)) / ks
        kernel = kernel[:, None, :, :]
        # Rotate the motion blur kernel.
        angle = torch.FloatTensor(B).uniform_(0, 360.0).unsqueeze(0).to(inp)
        rads = (angle * (math.pi / 180.0)).reshape(B, 1, 1)
        r00 = torch.cos(rads)
        r01 = -torch.sin(rads)
        r10 = torch.sin(rads)
        r11 = torch.cos(rads)
        top = torch.cat([r00, r01, 0 * r00], dim=2)
        bot = torch.cat([r10, r11, 0 * r00], dim=2)
        affine = torch.cat([top, bot], dim=1).to(inp)
        grid = torch.nn.functional.affine_grid(
            affine, list(kernel.shape), align_corners=True
        )
        kernel = torch.nn.functional.grid_sample(kernel, grid, align_corners=True)
        # Apply the rotated motion blur kernel to the images.
        kernel = torch.repeat_interleave(kernel, C, dim=0)
        kernel = kernel.reshape(B * C, 1, kernel_size, kernel_size)
        inp = inp.reshape(1, B * C, H, W)
        out = torch.nn.functional.conv2d(
            inp, kernel, padding=int(half_sizes.max()), groups=B * C
        )
        out = out.reshape(B, C, H, W)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def shadow_highlight(
        self,
        inp: torch.Tensor,
        mag: float = 0.5,
        min_quads: int = 2,
        max_quads: int = 5,
    ):
        B, H, W = inp.shape[0], inp.shape[-2], inp.shape[-1]
        out = inp
        num_quads = torch.randint(low=min_quads, high=max_quads, size=(B,))

        kwargs = {
            "scale_param": GeoAugParam(min=0.5, max=0.9),
            "translate_x_param": GeoAugParam(min=-0.8, max=0.8),
            "translate_y_param": GeoAugParam(min=-0.8, max=0.8),
            "rotate_param": GeoAugParam(min=-180, max=180),
            "perspective_param": GeoAugParam(min=0.0, max=0.8),
        }
        rga = RandomGeoAug(**kwargs)
        max_num_quads = num_quads.max()
        for i in range(max_num_quads):
            # Faster to apply conv to mini image then resize later.
            mask = torch.ones(B, 1, 50, 50).to(inp)
            # Randomly distort the quad.
            mask = rga.forward(mask)
            # Smooth the quad.
            mask = self.gaussian_smooth(mask, kmin=3, kmax=11, smin=3, smax=11, mag=1.0)
            mask = torch.nn.functional.interpolate(
                mask, size=[H, W], mode="bilinear", align_corners=True
            )
            val = truncated_normal_(B, mean=0.0, std=(mag * 0.6)).to(inp)
            quad = mask * val.reshape(B, 1, 1, 1)
            out2 = (1.0 - mask) * out + mask * (out + quad)
            apply_quad = torch.tensor(i).reshape(1, 1, 1, 1).repeat(
                B, 1, 1, 1
            ) < num_quads.reshape(B, 1, 1, 1)
            apply_quad = apply_quad.to(inp.device)
            out = torch.where(apply_quad, out2, out)
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def gaussian_noise(self, inp: torch.Tensor, mag: float = 0.5):
        B, C, H, W = inp.shape
        val = torch.FloatTensor(B).uniform_(0, mag * 0.1).abs().reshape(B, 1, 1)
        noise = val * torch.randn((B, H, W))
        noise = noise[:, None, :, :]
        noise = noise.repeat(1, C, 1, 1).to(inp.device)
        out = inp + noise
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def salt_and_pepper(self, inp: torch.Tensor, mag: float = 0.5):
        B, C, H, W = inp.shape
        noise_prob = mag * torch.FloatTensor(B).normal_(
            0.0, (mag * 0.05 + 1e-9)
        ).abs().to(inp)
        coins = torch.rand(size=(B, H, W)).to(inp).unsqueeze(1).repeat(1, C, 1, 1)
        heavy_noise = 0.5 * torch.randn(B, H, W).to(inp).unsqueeze(1).repeat(1, C, 1, 1)
        out = torch.where(
            coins > noise_prob.reshape(-1, 1, 1, 1), inp, inp + heavy_noise
        )
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def forward(self, inp: torch.Tensor, debug: bool = False, seed: int = None):
        """
        Runs the Random Photometric Augmentation on a batch of images.

        :param inp: BxCxHxW batch of input images
        :param debug: if True, print out some info about which transforms were used

        """
        if seed is not None:
            torch.manual_seed(seed)
        check_input(inp)  # Make sure input is well-formed.
        B, C, H, W = inp.shape
        result = inp.clone()  # Creates a copy of input (leaves input unchanged).
        # Randomly permute function ordering, except if "fix_pos" is True.
        permute_names = [
            name for name in self.fn_names if not getattr(self, name + "_param").fix_pos
        ]
        indices = torch.randperm(len(permute_names)).tolist()
        perm_fns = [permute_names[idx] for idx in indices]
        for i, name in enumerate(self.fn_names):
            if getattr(self, name + "_param").fix_pos:
                perm_fns.insert(i, name)
        # Iterate randomly permuted augmentation functions and maybe apply it.
        for fn_name in perm_fns:
            prob = getattr(self, fn_name + "_param").prob
            coins = torch.rand(B).reshape(-1, 1, 1, 1).to(inp)
            mask = coins < prob
            # Optional printing, helpful for debugging.
            if debug:
                print(fn_name, int(mask.sum()), "/", result.shape[0])
            # Skip operation if no elt in batch needs this augmentation.
            if torch.all(torch.eq(mask, False)):
                continue
            # Run the augmentation across all elts in batch.
            mag = getattr(self, fn_name + "_param").mag
            fn_method = getattr(self, fn_name)
            pre = result.clone()
            # Actual call to augmentation is here.
            result2 = fn_method(pre, mag=mag)
            mask = mask.repeat(1, C, H, W)
            # Apply the augmentation only where needed (from coins).
            final = torch.where(mask, result2, result)
            result = final.clone()
        return result
