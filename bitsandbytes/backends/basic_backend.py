from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from bitsandbytes.functional import QuantState


class DeviceBackends(ABC):
    """Base class for devices backends that will implement their own 8bits and 4bits functions."""

    @abstractmethod
    def get_name(self) -> str:
        """Name of the device as the backend support."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def double_quant(
        cls,
        A,
        col_stats=None,
        row_stats=None,
        out_col=None,
        out_row=None,
        threshold=0.0,
    ):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transform(
        cls,
        A,
        to_order,
        from_order="row",
        out=None,
        transpose=False,
        state=None,
        ld=None,
    ):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def igemmlt(cls, A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def mm_dequant(
        cls,
        A,
        quant_state,
        row_stats,
        col_stats,
        out=None,
        new_row_stats=None,
        new_col_stats=None,
        bias=None,
    ):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def extract_outliers(cls, A, SA, idx):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def quantize_4bit(
        cls,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type="fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_4bit(
        cls,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type="fp4",
    ) -> torch.Tensor:
        raise NotImplementedError
