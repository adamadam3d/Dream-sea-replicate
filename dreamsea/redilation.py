import torch.nn as nn


class ReDilation:
    """ScaleCrafter-style re-dilation (He et al. 2023, arXiv:2310.07702).

    A UNet trained at 224px has a receptive field tuned to 224px images; sampled
    directly at a larger canvas, each ~224px window is denoised as if it were a
    complete image, which produces the classic repeated-tile artifact. Re-dilation
    multiplies the dilation of every spatial (kernel > 1) Conv2d by the
    resolution scale factor — same weights, sampled with gaps — so each layer's
    field of view grows to match the larger canvas. Padding is scaled by the same
    factor, which keeps every layer's output spatial size identical to the
    undilated case (including the stride-2 downsamplers), so no other code needs
    to change.

    Intended use: enable only for the early high-noise denoising steps, where the
    model lays down global composition, then disable so fine texture in the late
    steps is rendered at native kernel scale and stays sharp.
    """

    def __init__(self, module, scale):
        rounded = max(1, int(round(scale)))
        if abs(scale - rounded) > 1e-6:
            print(f"ReDilation: non-integer scale {scale:.2f} rounded to {rounded} "
                  f"(dilation must be an integer).")
        self.scale = rounded
        # Snapshot each conv's original dilation/padding so disable() is exact.
        self._convs = []
        for m in module.modules():
            if isinstance(m, nn.Conv2d) and max(m.kernel_size) > 1:
                self._convs.append((m, m.dilation, m.padding))
        self.active = False

    def enable(self):
        if self.active or self.scale == 1:
            return
        for conv, dilation, padding in self._convs:
            conv.dilation = tuple(d * self.scale for d in dilation)
            conv.padding = tuple(p * self.scale for p in padding)
        self.active = True

    def disable(self):
        if not self.active:
            return
        for conv, dilation, padding in self._convs:
            conv.dilation = dilation
            conv.padding = padding
        self.active = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *exc):
        self.disable()
        return False
