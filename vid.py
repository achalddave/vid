import functools
import glob
import logging
import math
import numpy as np
import os
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

import click
import moviepy.editor as mpy
from moviepy.config import get_setting
from natsort import natsorted, ns
from PIL import Image


FFMPEG_BINARY = get_setting('FFMPEG_BINARY')

loglevel_param = click.option('--loglevel', type=str, default='error')
codec_param = click.option('--codec', type=str, default='libx264')
verbose_param = click.option('--verbose/--no-verbose', default=False)


def globbed_paths(filepaths):
    output = []
    for filepath in filepaths:
        globbed_files = glob.glob(filepath, recursive=True)
        if not globbed_files:
            globbed_files = [filepath]
        sorted_paths = natsorted(globbed_files, alg=ns.PATH)
        output.extend([Path(x) for x in sorted_paths])
    return output


def validate_globbed_paths(paths):
    paths = globbed_paths(paths)
    for p in paths:
        if not p.exists():
            raise click.BadParameter('Path %s does not exist' % p)
    return paths


def on_value_only(fn):
    def wrapper(_context, _param, value):
        return fn(value)
    return wrapper


def common_params(func):
    @click.option('--loglevel', type=str, default='error')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.group()
def main():
    pass


@main.command()
@click.argument(
    'images',
    required=True,
    nargs=-1,
    callback=on_value_only(validate_globbed_paths))
@click.argument('output', required=True)
@click.option('--fps', type=int, default=24)
@click.option(
    '--shape',
    type=str,
    help=('Specify size of output video, in WxH format. By default, use the '
          'size of the first frame.'),
    default=None)
@codec_param
@loglevel_param
@verbose_param
def slideshow(images, output, fps, shape, codec, loglevel, verbose):
    # ImageSequenceClip doesn't work with grayscale images, so we have to
    # manually recreate it.
    image_starts = [
        1.0 * i / fps - np.finfo(np.float32).eps for i in range(len(images))
    ]
    last_loaded = {'index': None, 'image': None}

    if shape is None:
        video_size = Image.open(images[0]).size
    else:
        try:
            video_size = shape.split('x')
            video_size = (int(video_size[0]), int(video_size[1]))
        except (ValueError, IndexError):
            logging.error('Could not parse shape specification %s' % shape)
            raise

    def make_frame(t):
        image_index = max(
            [i for i in range(len(images)) if image_starts[i] <= t])
        if image_index != last_loaded['index']:
            last_loaded['index'] = image_index
            image = Image.open(images[image_index])
            if image.size != video_size:
                image = image.resize(video_size)
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack((image, image, image), -1)
            last_loaded['image'] = image
        return last_loaded['image']

    video = mpy.VideoClip(make_frame=make_frame, duration=len(images) / fps)
    if output.endswith('.gif'):
        video.write_gif(
            output, fps=fps, verbose=verbose, progress_bar=verbose)
    else:
        video.write_videofile(
            output, fps=fps, verbose=verbose, progress_bar=verbose)


if __name__ == '__main__':
    main()
