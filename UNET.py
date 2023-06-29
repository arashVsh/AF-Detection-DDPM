import torch
import torch.nn as nn
from Settings import IMAGE_SIZE
from Block import MyBlock


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
    return embedding


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        CHANNELS = 25

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, IMAGE_SIZE, IMAGE_SIZE), 1, CHANNELS),
            MyBlock((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), CHANNELS, CHANNELS),
            MyBlock((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), CHANNELS, CHANNELS),
        )
        self.down1 = nn.Conv2d(CHANNELS, CHANNELS, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, CHANNELS)
        self.b2 = nn.Sequential(
            MyBlock(
                (CHANNELS, IMAGE_SIZE // 2, IMAGE_SIZE // 2), CHANNELS, CHANNELS * 2
            ),
            MyBlock(
                (CHANNELS * 2, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                CHANNELS * 2,
                CHANNELS * 2,
            ),
            MyBlock(
                (CHANNELS * 2, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                CHANNELS * 2,
                CHANNELS * 2,
            ),
        )
        self.down2 = nn.Conv2d(CHANNELS * 2, CHANNELS * 2, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, CHANNELS * 2)
        self.b3 = nn.Sequential(
            MyBlock(
                (CHANNELS * 2, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 2,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 4,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 4,
                CHANNELS * 4,
            ),
        )
        self.down3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),  # Add cell padding
            nn.Conv2d(CHANNELS * 4, CHANNELS * 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(CHANNELS * 4, CHANNELS * 4, 4, 2, 1),
        )

        self.te4 = self._make_te(time_emb_dim, CHANNELS * 4)
        self.b4 = nn.Sequential(
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 4,
                CHANNELS * 8,
            ),
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 8,
                CHANNELS * 8,
            ),
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 8,
                CHANNELS * 8,
            ),
        )
        self.down4 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),  # Add cell padding
            nn.Conv2d(CHANNELS * 8, CHANNELS * 8, 2, 1),
            nn.SiLU(),
            nn.Conv2d(CHANNELS * 8, CHANNELS * 8, 4, 2, 1),
        )

        self.te4 = self._make_te(time_emb_dim, CHANNELS * 4)
        self.b4 = nn.Sequential(
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 4,
                CHANNELS * 8,
            ),
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 8,
                CHANNELS * 8,
            ),
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 8,
                CHANNELS * 8,
            ),
        )
        self.down4 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),  # Add cell padding
            nn.Conv2d(CHANNELS * 8, CHANNELS * 8, 2, 1),
            nn.SiLU(),
            nn.Conv2d(CHANNELS * 8, CHANNELS * 8, 4, 2, 1),
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, CHANNELS * 8)
        self.b_mid = nn.Sequential(
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 16, IMAGE_SIZE // 16),
                CHANNELS * 8,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 16, IMAGE_SIZE // 16),
                CHANNELS * 4,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 16, IMAGE_SIZE // 16),
                CHANNELS * 4,
                CHANNELS * 8,
            ),
        )

        # Second half

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS * 8, CHANNELS * 8, 4, 2, 1, 0),
            nn.SiLU(),
            nn.ConvTranspose2d(
                CHANNELS * 8, CHANNELS * 8, 2, 1, 0
            ),  # Decrease padding to 0
            nn.ZeroPad2d((0, -1, 0, -1)),  # Remove one cell from each side
        )

        self.te5 = self._make_te(time_emb_dim, CHANNELS * 16)
        self.b5 = nn.Sequential(
            MyBlock(
                (CHANNELS * 16, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 16,
                CHANNELS * 8,
            ),
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 8,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
                CHANNELS * 4,
                CHANNELS * 4,
            ),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS * 4, CHANNELS * 4, 4, 2, 1, 0),
            nn.SiLU(),
            nn.ConvTranspose2d(
                CHANNELS * 4, CHANNELS * 4, 2, 1, 0
            ),  # Decrease padding to 0
            nn.ZeroPad2d((0, -1, 0, -1)),  # Remove one cell from each side
        )

        self.te6 = self._make_te(time_emb_dim, CHANNELS * 8)
        self.b6 = nn.Sequential(
            MyBlock(
                (CHANNELS * 8, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 8,
                CHANNELS * 4,
            ),
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 4,
                CHANNELS * 2,
            ),
            MyBlock(
                (CHANNELS * 2, IMAGE_SIZE // 4, IMAGE_SIZE // 4),
                CHANNELS * 2,
                CHANNELS * 2,
            ),
        )

        self.up2 = nn.ConvTranspose2d(
            CHANNELS * 2, CHANNELS * 2, 4, 2, 1
        )  # Adjust padding and stride here
        self.te7 = self._make_te(time_emb_dim, CHANNELS * 4)
        self.b7 = nn.Sequential(
            MyBlock(
                (CHANNELS * 4, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                CHANNELS * 4,
                CHANNELS * 2,
            ),
            MyBlock(
                (CHANNELS * 2, IMAGE_SIZE // 2, IMAGE_SIZE // 2), CHANNELS * 2, CHANNELS
            ),
            MyBlock((CHANNELS, IMAGE_SIZE // 2, IMAGE_SIZE // 2), CHANNELS, CHANNELS),
        )

        self.up3 = nn.ConvTranspose2d(CHANNELS, CHANNELS, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, CHANNELS * 2)
        self.b_out = nn.Sequential(
            MyBlock((CHANNELS * 2, IMAGE_SIZE, IMAGE_SIZE), CHANNELS * 2, CHANNELS),
            MyBlock((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), CHANNELS, CHANNELS),
            MyBlock(
                (CHANNELS, IMAGE_SIZE, IMAGE_SIZE), CHANNELS, CHANNELS, normalize=False
            ),
        )

        self.conv_out = nn.Conv2d(CHANNELS, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 1, IMAGE_SIZE, IMAGE_SIZE) (grayscale image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(
            x + self.te1(t).reshape(n, -1, 1, 1)
        )  # (N, 10, IMAGE_SIZE, IMAGE_SIZE)
        out2 = self.b2(
            self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
        )  # (N, 20, IMAGE_SIZE // 2, IMAGE_SIZE // 2)
        out3 = self.b3(
            self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
        )  # (N, 40, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        out4 = self.b4(
            self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1)
        )  # (N, 40, IMAGE_SIZE // 8, IMAGE_SIZE // 8)

        out_mid = self.b_mid(
            self.down4(out4) + self.te_mid(t).reshape(n, -1, 1, 1)
        )  # (N, 40, IMAGE_SIZE // 16, IMAGE_SIZE // 16)

        out5 = torch.cat(
            (out4, self.up0(out_mid)), dim=1
        )  # (N, 80, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
        out5 = self.b5(
            out5 + self.te5(t).reshape(n, -1, 1, 1)
        )  # (N, 20, IMAGE_SIZE // 8, IMAGE_SIZE // 8)

        out6 = torch.cat(
            (out3, self.up1(out5)), dim=1
        )  # (N, 40, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        out6 = self.b6(
            out6 + self.te6(t).reshape(n, -1, 1, 1)
        )  # (N, 10, IMAGE_SIZE // 4, IMAGE_SIZE // 4)

        out7 = torch.cat(
            (out2, self.up2(out6)), dim=1
        )  # (N, 40, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        out7 = self.b7(
            out7 + self.te7(t).reshape(n, -1, 1, 1)
        )  # (N, 10, IMAGE_SIZE // 4, IMAGE_SIZE // 4)

        out = torch.cat(
            (out1, self.up3(out7)), dim=1
        )  # (N, 20, IMAGE_SIZE, IMAGE_SIZE)
        out = self.b_out(
            out + self.te_out(t).reshape(n, -1, 1, 1)
        )  # (N, 1, IMAGE_SIZE, IMAGE_SIZE)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )
