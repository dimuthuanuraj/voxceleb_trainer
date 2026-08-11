import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from models._frontend import PreEmphasis as _CanonicalPreEmphasis


class PreEmphasis(_CanonicalPreEmphasis):
    """Backwards-compatibility alias for :class:`models._frontend.PreEmphasis`.

    The RawNet pipeline pairs PreEmphasis with ``InstanceNorm1d(1, ...)``
    which expects a 3D ``(B, 1, T)`` input, so this subclass pins
    ``squeeze=False`` on the canonical class. Historical output shape
    is preserved.

    New code should import directly from :mod:`models._frontend` with
    ``squeeze=False`` explicit. See
    ``docs/bugfixes/BUGFIX-025-shared-audio-frontend.md``.
    """

    def __init__(self, coef: float = 0.97) -> None:
        super().__init__(coef=coef, squeeze=False)


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        dilation=None,
        scale=4,
        pool=False,
    ):

        super().__init__()

        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()

        self.width = width

        # nn.Identity is the standard PyTorch no-op placeholder: always an
        # nn.Module (so it shows up in children() / print(model) / state_dict
        # tooling), always callable, returns its input unchanged. Pre-fix this
        # slot held the literal `False`, which made forward() rely on a
        # truthiness guard to avoid calling a non-callable — a fragile pattern.
        self.mp = nn.MaxPool1d(pool) if pool else nn.Identity()
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual
        # self.mp is either MaxPool1d (when pool>0) or Identity (when pool=False/0);
        # always callable, no truthiness guard needed.
        out = self.mp(out)
        out = self.afms(out)

        return out
