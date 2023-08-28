import os
import sys
import argparse

import torch
import numpy as np
import model_navigator as nav

from pathlib import Path
from typing import Iterable
from argparse import Namespace

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.autoshape import AutoShape

from model_navigator.api.config import Sample
from model_navigator.core.package import Package
from model_navigator.exceptions import (
    ModelNavigatorEmptyPackageError,
    ModelNavigatorError,
    ModelNavigatorWrongParameterError,
)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def dir_path(string) -> str:
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(f'Directory `{string}`, does not exist')
    
def img_shape(shape):
    if len(shape) < 3:
        raise ValueError("Please spec image shape correctly")
    return shape


def get_verify_function():
    def verify_func(ys_runner: Iterable[Sample], ys_expected: Iterable[Sample]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-3, atol=1.0e-4) for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True
    return verify_func


def get_dataloader(image_shape, device:str='cuda:0'):
    shape = [1] + image_shape
    return [torch.rand(shape, device=torch.device(device)) for _ in range(10)]


def get_model(
        model_path=None, 
        device:str='cuda:0'
        ) -> AutoShape:
    model = AutoShape(AutoBackend(model_path, torch.device(device))).model
    return model.eval()


def navigation(model:AutoShape, dataloader, verify_function) -> Package:
    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_function,
        verbose=True,
        debug=True
    )
    return package


def export(package:Package, path:str) -> None:
    try:
        nav.triton.model_repository.add_model_from_package(
            model_repository_path=Path(path),
            model_name="yolov8",
            package=package,
            strategy=nav.MinLatencyStrategy(),
        )
    except (ModelNavigatorWrongParameterError, ModelNavigatorEmptyPackageError, ModelNavigatorError) as e:
        print.warning(f"Model repository cannot be created.\n{str(e)}")


def get_parser() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', '-m', type=str, help='Path to model .pt file')
    parser.add_argument('--output-dir', '-o', type=dir_path, default=ROOT/'model_repo', help='Output path for triton navigator')
    parser.add_argument('--device', '-d', default='cuda:0', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-shape', '-i', nargs='+', type=img_shape, default=[3, 640, 640], help='image (h, w), default [3, 640, 640]')
    parser.add_argument('--verify', action='store_true', help='If you want to verify model by triton navigator')
    return parser.parse_args()


def main(args:dict) -> None:
    if args["model_path"]:
        if not os.path.isfile(args["model_path"]) and not args["torch_hub"]:
            raise FileNotFoundError(f'Not found model file in `{args["model_path"]}` path')

    
    model = get_model(
        args["model_path"], 
        args["device"]
    )

    dataloader = get_dataloader(args["img_shape"], args["device"])
    verify_function = get_verify_function() if args["verify"] else None

    package = navigation(model, dataloader, verify_function)

    export(package, args["output_dir"])
 

if __name__ == "__main__":
    args = vars(get_parser())
    main(args)
