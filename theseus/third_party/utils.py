import torch


# Obtained from https://github.com/pytorch/pytorch/issues/34704#issuecomment-878940122
def grid_sample(image, optical):
    """
    Custom implementation for torch.nn.functional.grid_sample() to avoid this warning:
    > "RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented"
    """
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape
    ix = optical[..., 0]
    iy = optical[..., 1]
    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1
    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)
    with torch.no_grad():
        ix_nw = torch.clamp(ix_nw, 0, IW - 1)
        iy_nw = torch.clamp(iy_nw, 0, IH - 1)
        ix_ne = torch.clamp(ix_ne, 0, IW - 1)
        iy_ne = torch.clamp(iy_ne, 0, IH - 1)
        ix_sw = torch.clamp(ix_sw, 0, IW - 1)
        iy_sw = torch.clamp(iy_sw, 0, IH - 1)
        ix_se = torch.clamp(ix_se, 0, IW - 1)
        iy_se = torch.clamp(iy_se, 0, IH - 1)
    image = image.view(N, C, IH * IW)
    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )
    return out_val
