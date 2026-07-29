"""
LTX Latent Upsampler Tiled v1.0

Drop-in replacement for ComfyUI's LTXVLatentUpsampler that tiles the
spatial dimension when input is large enough to trigger the upscale
model's aspect-ratio failure modes (color shifts, distortion at
1300+ pixel vertical extent).
"""

import math
import torch
from comfy import model_management


class LTXVLatentUpsamplerTiled:
    """
    Spatially-tiled drop-in replacement for LTXVLatentUpsampler.
    Solves color/distortion artifacts at extreme aspect ratios.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":       ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae":           ("VAE",),
            },
            "optional": {
                "tile_size":             ("INT",     {"default": 24, "min": 8,  "max": 128, "step": 1}),
                "overlap":               ("INT",     {"default": 8,  "min": 2,  "max": 32,  "step": 1}),
                "max_size_for_no_tile":  ("INT",     {"default": 32, "min": 8,  "max": 256, "step": 1}),
                "rotate_for_landscape":  ("BOOLEAN", {"default": False}),
                "debug":                 ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample_latent_tiled"
    CATEGORY = "10S Nodes/Latent"
    DESCRIPTION = (
        "Tiled drop-in replacement for LTXVLatentUpsampler. Solves color shifts and "
        "distortion at extreme aspect ratios by processing the upscale in spatial "
        "tiles with cosine-windowed overlap blending."
    )

    def upsample_latent_tiled(self, samples, upscale_model, vae,
                              tile_size=24, overlap=8,
                              max_size_for_no_tile=32,
                              rotate_for_landscape=False,
                              debug=False):
        device = model_management.get_torch_device()

        # 【修復 1】解包出真正的 PyTorch Model (raw_model)
        raw_model = upscale_model.model if hasattr(upscale_model, "model") else upscale_model

        # 獲取模型 dtype
        if hasattr(raw_model, "parameters"):
            model_dtype = next(raw_model.parameters()).dtype
        elif hasattr(upscale_model, "get_dtype"):
            model_dtype = upscale_model.get_dtype()
        else:
            model_dtype = torch.float16

        latents = samples["samples"]
        input_dtype = latents.dtype
        B, C, F, H, W = latents.shape

        if debug:
            print(f"→ [10S] LatentUpsamplerTiled: input shape={tuple(latents.shape)} "
                  f"dtype={input_dtype}")

        if overlap >= tile_size:
            print(f"→ [10S] LatentUpsamplerTiled: overlap={overlap} >= tile_size={tile_size}; "
                  f"clamping overlap to {tile_size - 1}")
            overlap = max(1, tile_size - 1)

        # 記憶體預估
        memory_required = model_management.module_size(raw_model)
        tile_volume = B * C * F * (tile_size * 2) ** 2
        output_volume = B * C * F * (H * 2) * (W * 2)
        memory_required += tile_volume * 3000.0
        memory_required += output_volume * 4.0  # fp32 accumulator
        model_management.free_memory(memory_required, device)

        try:
            # 【修復 2】正確將模型移至 GPU
            if hasattr(upscale_model, "model_accelerate"):
                model_management.load_model_gpu(upscale_model)
            elif hasattr(raw_model, "to"):
                raw_model.to(device)

            # Un-normalize ONCE on full latent (global per-channel statistics)
            latents_dev = latents.to(dtype=model_dtype, device=device)
            latents_un = vae.first_stage_model.per_channel_statistics.un_normalize(latents_dev)

            # ─── Optional rotation for landscape orientation ─────────────────
            rotated = False
            if rotate_for_landscape and latents_un.shape[-2] > latents_un.shape[-1]:
                latents_un = latents_un.transpose(-1, -2).contiguous()
                rotated = True
                if debug:
                    print(f"  · rotated for landscape: H={H}>W={W} → "
                          f"shape={tuple(latents_un.shape)}")
                H, W = W, H

            # Decide tiling
            should_tile = (H > max_size_for_no_tile) or (W > max_size_for_no_tile)
            if not should_tile:
                if debug:
                    print(f"  · H={H} W={W} both ≤ max_size_for_no_tile="
                          f"{max_size_for_no_tile}; using non-tiled path")
                # 【修復 3】使用 raw_model 進行前向推論
                upsampled = raw_model(latents_un)
            else:
                if debug:
                    print(f"  · tiling triggered: H={H} > {max_size_for_no_tile} "
                          f"or W={W} > {max_size_for_no_tile}")
                upsampled = self._upsample_tiled(
                    latents_un, raw_model, tile_size, overlap, debug
                )

            # Rotate back if we rotated
            if rotated:
                upsampled = upsampled.transpose(-1, -2).contiguous()
                if debug:
                    print(f"  · rotated back: shape={tuple(upsampled.shape)}")

            # Re-normalize ONCE on full output
            upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)

        finally:
            # 【修復 4】安全釋放資源
            if hasattr(raw_model, "cpu"):
                raw_model.cpu()
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()

        upsampled = upsampled.to(
            dtype=input_dtype,
            device=model_management.intermediate_device(),
        )

        if debug:
            print(f"→ [10S] LatentUpsamplerTiled: output shape={tuple(upsampled.shape)}")

        return_dict = samples.copy()
        return_dict["samples"] = upsampled
        return_dict.pop("noise_mask", None)
        return (return_dict,)

    # ─── Core tiled upscale ─────────────────────────────────────────────────

    def _upsample_tiled(self, latents, upscale_model, tile_size, overlap, debug):
        """
        Spatial tiling with cosine-windowed overlap blending.
        Note: upscale_model here is guaranteed to be the PyTorch Module (raw_model).
        """
        device = latents.device
        dtype = latents.dtype
        B, C, F, H, W = latents.shape

        h_starts = self._compute_tile_starts(H, tile_size, overlap)
        w_starts = self._compute_tile_starts(W, tile_size, overlap)

        def actual_overlap_with_prev(starts, idx, tile_sz, total):
            if idx <= 0:
                return 0
            prev_start = starts[idx - 1]
            prev_end = min(prev_start + tile_sz, total)
            this_start = starts[idx]
            return max(0, prev_end - this_start)

        def actual_overlap_with_next(starts, idx, tile_sz, total):
            if idx >= len(starts) - 1:
                return 0
            this_start = starts[idx]
            this_end = min(this_start + tile_sz, total)
            next_start = starts[idx + 1]
            return max(0, this_end - next_start)

        # Process first tile to determine upscale ratio
        h0_end = min(h_starts[0] + tile_size, H)
        w0_end = min(w_starts[0] + tile_size, W)
        first_tile_in = latents[:, :, :, h_starts[0]:h0_end, w_starts[0]:w0_end].contiguous()
        first_tile_out = upscale_model(first_tile_in)

        scale_h = first_tile_out.shape[3] / first_tile_in.shape[3]
        scale_w = first_tile_out.shape[4] / first_tile_in.shape[4]

        scale = (scale_h + scale_w) / 2.0
        if abs(scale_h - scale_w) > 0.01:
            print(f"→ [10S] LatentUpsamplerTiled: WARN non-uniform scale detected "
                  f"({scale_h:.3f} vs {scale_w:.3f}); using average={scale:.3f}")

        out_H = int(round(H * scale))
        out_W = int(round(W * scale))

        if debug:
            print(f"  · detected upscale ratio: {scale:.3f}x "
                  f"→ output {out_H}x{out_W} (from {H}x{W})")
            print(f"  · tile_size={tile_size} overlap={overlap} "
                  f"→ h_starts={h_starts} ({len(h_starts)}) "
                  f"w_starts={w_starts} ({len(w_starts)}) "
                  f"total_tiles={len(h_starts)*len(w_starts)}")

        output = torch.zeros((B, C, F, out_H, out_W), dtype=torch.float32, device=device)
        weights = torch.zeros((1, 1, 1, out_H, out_W), dtype=torch.float32, device=device)

        single_h = len(h_starts) == 1
        single_w = len(w_starts) == 1
        first_tile_logged = False

        for h_idx, h_start in enumerate(h_starts):
            h_end = min(h_start + tile_size, H)
            for w_idx, w_start in enumerate(w_starts):
                w_end = min(w_start + tile_size, W)

                if h_idx == 0 and w_idx == 0:
                    tile_out = first_tile_out
                    tile_in_shape = first_tile_in.shape
                else:
                    tile_in = latents[:, :, :, h_start:h_end, w_start:w_end].contiguous()
                    tile_out = upscale_model(tile_in)
                    tile_in_shape = tile_in.shape

                out_h_start = int(round(h_start * scale))
                out_h_end = out_h_start + tile_out.shape[3]
                out_w_start = int(round(w_start * scale))
                out_w_end = out_w_start + tile_out.shape[4]

                out_h_end = min(out_h_end, out_H)
                out_w_end = min(out_w_end, out_W)
                actual_out_h = out_h_end - out_h_start
                actual_out_w = out_w_end - out_w_start
                if actual_out_h <= 0 or actual_out_w <= 0:
                    continue

                ov_top_in    = actual_overlap_with_prev(h_starts, h_idx, tile_size, H)
                ov_bot_in    = actual_overlap_with_next(h_starts, h_idx, tile_size, H)
                ov_left_in   = actual_overlap_with_prev(w_starts, w_idx, tile_size, W)
                ov_right_in  = actual_overlap_with_next(w_starts, w_idx, tile_size, W)

                fade_top    = int(round(ov_top_in   * scale)) if not single_h else 0
                fade_bot    = int(round(ov_bot_in   * scale)) if not single_h else 0
                fade_left   = int(round(ov_left_in  * scale)) if not single_w else 0
                fade_right  = int(round(ov_right_in * scale)) if not single_w else 0

                fade_top   = min(fade_top,   actual_out_h)
                fade_bot   = min(fade_bot,   actual_out_h)
                fade_left  = min(fade_left,  actual_out_w)
                fade_right = min(fade_right, actual_out_w)

                window = self._make_window_2d(
                    actual_out_h, actual_out_w,
                    fade_top   = fade_top,
                    fade_bot   = fade_bot,
                    fade_left  = fade_left,
                    fade_right = fade_right,
                    device=device,
                )
                window = window.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                tile_out_cropped = tile_out[:, :, :, :actual_out_h, :actual_out_w]

                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += \
                    tile_out_cropped.float() * window
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += window

                if debug and not first_tile_logged:
                    print(f"  · first tile: in={tuple(tile_in_shape)} "
                          f"→ out={tuple(tile_out.shape)} "
                          f"window=({actual_out_h},{actual_out_w}) "
                          f"actual_overlaps=top:{ov_top_in},bot:{ov_bot_in},"
                          f"left:{ov_left_in},right:{ov_right_in} (input coords) "
                          f"→ fades=top:{fade_top},bot:{fade_bot},"
                          f"left:{fade_left},right:{fade_right}")
                    first_tile_logged = True

        if debug:
            w_min = weights.min().item()
            w_max = weights.max().item()
            print(f"  · weight accumulator: min={w_min:.4f} max={w_max:.4f} "
                  f"(should be ≈1.0 everywhere; max>1 indicates blending bug)")
            if w_min < 1e-3:
                print(f"  ⚠️  weight min very low — some output positions "
                      f"have unstable normalization. Increase overlap.")
            if w_max > 1.05:
                print(f"  ⚠️  weight max > 1.05 — cosine fades not summing "
                      f"to 1.0 in overlap zones. Possible window construction issue.")

        output = output / (weights + 1e-8)
        return output.to(dtype=dtype)

    @staticmethod
    def _compute_tile_starts(total_size, tile_size, overlap):
        if total_size <= tile_size:
            return [0]

        starts = []
        stride = tile_size - overlap
        pos = 0
        while pos + tile_size < total_size:
            starts.append(pos)
            pos += stride

        last_start = total_size - tile_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        return starts

    @staticmethod
    def _make_window_1d(size, fade_left_size, fade_right_size, device):
        win = torch.ones(size, dtype=torch.float32, device=device)

        if fade_left_size > 0:
            fl = min(fade_left_size, size)
            i = torch.arange(fl, dtype=torch.float32, device=device)
            win[:fl] = 0.5 * (1.0 - torch.cos(math.pi * i / fl))

        if fade_right_size > 0:
            fr = min(fade_right_size, size)
            i = torch.arange(fr, dtype=torch.float32, device=device)
            win[size - fr:] = 0.5 * (1.0 + torch.cos(math.pi * i / fr))

        return win

    @staticmethod
    def _make_window_2d(h, w, fade_top, fade_bot, fade_left, fade_right, device):
        win_h = LTXVLatentUpsamplerTiled._make_window_1d(h, fade_top, fade_bot, device)
        win_w = LTXVLatentUpsamplerTiled._make_window_1d(w, fade_left, fade_right, device)
        return win_h.unsqueeze(1) * win_w.unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "LTXVLatentUpsamplerTiled": LTXVLatentUpsamplerTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVLatentUpsamplerTiled": "🔍 LTX Latent Upsampler (Tiled)",
}
