import torch
import logging
import torch.nn as nn

from einops import rearrange
from omegaconf import DictConfig
from torchvision import transforms

import slide2vec.distributed as distributed
import slide2vec.models.vision_transformer_dino as vits_dino

from slide2vec.utils import update_state_dict
from slide2vec.data.augmentations import make_normalize_transform, MaybeToTensor

logger = logging.getLogger("slide2vec")


class ModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "region" and options.name == "panda-vit-s":
            tile_encoder = PandaViT(
                arch="vit_small",
                pretrained_weights=options.pretrained_weights,
                input_size=options.tile_size,
            )
            model = RegionFeatureExtractor(tile_encoder)
        else:
            raise ValueError(
                f"unsupported model configuration: level={options.level!r}, "
                f"name={options.name!r} (this trimmed slide2vec only supports "
                "panda-vit-s at region level)"
            )

        self.model = model.eval()
        self.model = self.model.to(self.model.device)

    def get_model(self):
        return self.model


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
        self.set_device()

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def build_encoder(self):
        raise NotImplementedError

    def get_transforms(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class PandaViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 224,
        ckpt_key: str = "teacher",
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        if input_size != 224:
            print(
                f"Warning: PandaViT will crop input images to 224x224"
            )
        self.input_size = input_size
        self.ckpt_key = ckpt_key
        self.features_dim = 384
        super(PandaViT, self).__init__()
        self.load_weights()

    def load_weights(self):
        if distributed.is_main_process():
            print(f"Loading pretrained weights from: {self.pretrained_weights}")
        state_dict = torch.load(self.pretrained_weights, map_location="cpu", weights_only=False)
        if self.ckpt_key:
            state_dict = state_dict[self.ckpt_key]
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="module."
        )
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="backbone."
        )
        state_dict, msg = update_state_dict(
            model_dict=self.encoder.state_dict(), state_dict=state_dict
        )
        if distributed.is_main_process():
            print(msg)
        self.encoder.load_state_dict(state_dict, strict=False)

    def build_encoder(self):
        encoder = vits_dino.__dict__[self.arch](
            img_size=256, patch_size=16
        )
        return encoder

    def get_transforms(self):
        if self.input_size == 224:
            transform = transforms.Compose(
                [
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )
        return transform

    def forward(self, x):
        embedding = self.encoder(x)
        output = {"embedding": embedding}
        return output


class RegionFeatureExtractor(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        tile_size: int = 256,
    ):
        super(RegionFeatureExtractor, self).__init__()
        self.tile_encoder = tile_encoder
        self.tile_size = tile_size
        self.device = self.tile_encoder.device
        self.features_dim = self.tile_encoder.features_dim

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        # x = [B, num_tiles, 3, 224, 224]
        B = x.size(0)
        x = rearrange(x, "b p c w h -> (b p) c w h")  # [B*num_tiles, 3, 224, 224]
        output = self.tile_encoder(x)["embedding"]  # [B*num_tiles, features_dim]
        embedding = rearrange(
            output, "(b p) f -> b p f", b=B
        )  # [B, num_tiles, features_dim]
        output = {"embedding": embedding}
        return output
