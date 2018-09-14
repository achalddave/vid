#!/usr/bin/env python

import glob
import logging
from pathlib import Path

import click
import numpy as np
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm


loglevel_param = click.option('--loglevel', type=str, default='error')
codec_param = click.option('--codec', type=str, default=None)
verbose_param = click.option('--verbose/--no-verbose', default=False)
audio_param = click.option('--audio/--no-audio', 'save_audio', default=True)


def flatten_once(lst):
    return [x for y in lst for x in y]


def globbed(filepath):
    globbed_files = glob.glob(filepath, recursive=True)
    if not globbed_files:
        globbed_files = [filepath]
    sorted_paths = natsorted(globbed_files, alg=ns.PATH)
    return [Path(x) for x in sorted_paths]


def validate_globbed_paths(paths):
    paths = flatten_once(globbed(path) for path in paths)
    for p in paths:
        if not p.exists():
            raise click.BadParameter('Path %s does not exist' % p)
    return paths


def validate_globbed_paths_allow_dummy(paths, dummy_path):
    globbed_paths = []
    for path in paths:
        if path == dummy_path:
            globbed_paths.append(path)
        else:
            globbed_paths.extend(validate_globbed_paths([path]))
    return globbed_paths


def validate_path(path):
    if not path.exists():
        raise click.BadParameter('Path %s does not exist' % path)
    return path


def on_value_only(fn):
    def wrapper(_context, _param, value):
        return fn(value)
    return wrapper


@click.group()
def main():
    """vid is a command line tool for common video manipulation tasks."""
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
    """Create a video from a sequence of images.

    \b
    Example usage:
        vid slideshow *.png output.mp4

    The images can be specified as an arbitrary list of paths or glob patterns.
    If a glob pattern is specified, `vid` attempts to sort the resulting
    paths in a sensible way using the `natsorted` library. For example, with
    a directory structure as:

    \b
        /data
            /video1
                /frame000.png
                /frame001.png
            /video2
                /frame000.png
                /frame001.png

    \b
    With the following command:
        vid slideshow '/data/*/*.png' output.mp4
    the images will appear, as expected, in the following order:
        ['/data/video1/frame000.png', '/data/video1/frame001.png',
         '/data/video2/frame000.png', '/data/video2/frame001.png']

    Note that the glob pattern must be in quotes for this to apply; otherwise,
    the glob pattern will be expanded (and sorted) by your shell.
    """
    # ImageSequenceClip doesn't play nicely with grayscale images, and
    # VideoClip has issues with images that have alpha channels, so I just roll
    # my own here.
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
    from moviepy.tools import extensions_dict

    if codec is None:
        extension = Path(output).suffix[1:]
        try:
            codec = extensions_dict[extension]['codec'][0]
        except KeyError:
            raise ValueError("Couldn't find the codec associated with the "
                             "filename. Please specify --codec")

    # ImageSequenceClip doesn't work with grayscale images, so we have to
    # manually recreate it.
    image_starts = [
        1.0 * i / fps - np.finfo(np.float32).eps for i in range(len(images))
    ]
    last_loaded = {'index': None, 'image': None}

    if shape is None:
        width, height = Image.open(images[0]).size
    else:
        try:
            width, height = shape.split('x')
            width, height = int(width), int(height)
        except (ValueError, IndexError):
            logging.error('Could not parse shape specification %s' % shape)
            raise

    has_alpha = any(Image.open(x).mode == 'RGBA' for x in images)

    def make_frame(t):
        image_index = max(
            [i for i in range(len(images)) if image_starts[i] <= t])
        if image_index != last_loaded['index']:
            image = Image.open(images[image_index])
            if image.size != (width, height):
                image = image.resize((width, height))
            image = np.array(image)
            if image.ndim == 2 or image.shape[2] == 1:
                if image.ndim == 3 and image.shape[2] == 1:
                    image = image[:, :, 0]
                image = np.stack((image, image, image), -1)

            if has_alpha and image.shape[2] == 3:
                mask = np.ones((height, width, 1), dtype=np.uint8)
                image = np.dstack((image, mask))
            last_loaded['index'] = image_index
            last_loaded['image'] = image
        return last_loaded['image']

    duration = len(images) / fps
    # Handle "height/width not divisible by 2" error for libx264
    ffmpeg_params = [
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-pix_fmt', 'yuv420p'
    ]
    with FFMPEG_VideoWriter(
            output, size=(width, height), fps=fps, withmask=has_alpha,
            codec=codec, ffmpeg_params=ffmpeg_params) as writer:
        for t in tqdm(np.arange(0, duration, 1.0 / fps), disable=not verbose):
            writer.write_frame(make_frame(t))


@main.command()
@click.argument(
    'videos',
    required=True,
    nargs=-1,
    callback=on_value_only(validate_globbed_paths))
@click.argument('output', required=True)
@audio_param
@loglevel_param
@verbose_param
def hstack(videos, output, save_audio, loglevel, verbose):
    """Merge videos horizontally into one video.

    \b
    Example usage:
        vid hstack first.mp4 second.mp4 third.mp4 output.mp4

    \b
    The above command creates puts three videos side-by-side:
        video1.mp4    video2.mp4    video3.mp4
    """

    from moviepy.video.io.VideoFileClip import VideoFileClip
    from utils.moviepy_wrappers.composite_clip import clips_array_maybe_none

    clips = [[
        VideoFileClip(str(v)) if v != Path('/dev/null') else None
        for v in videos
    ]]
    output_clip = clips_array_maybe_none(clips)
    if not save_audio:
        output_clip = output_clip.without_audio()

    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument(
    'videos',
    required=True,
    nargs=-1,
    callback=on_value_only(validate_globbed_paths))
@click.argument('output', required=True)
@audio_param
@loglevel_param
@verbose_param
def vstack(videos, output, save_audio, loglevel, verbose):
    """Merge videos vertically into one video.

    \b
    Example usage:
        vid vstack *.mp4 output.mp4

    \b
    The above command creates puts three videos on top of each other:
        video1.mp4
        video2.mp4
        video3.mp4
    """
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from utils.moviepy_wrappers.composite_clip import clips_array_maybe_none

    clips = [[VideoFileClip(str(v)) if v != Path('/dev/null') else None]
             for v in videos]
    output_clip = clips_array_maybe_none(clips)
    if not save_audio:
        output_clip = output_clip.without_audio()
    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument('videos', required=True, nargs=-1)
@click.argument('output', required=True)
@click.option('--num-rows', type=int, default=2)
@audio_param
@loglevel_param
@verbose_param
def grid(videos, output, num_rows, save_audio, loglevel, verbose):
    """Merge videos in a specific grid layout.

    \b
    Example usage:
        vid grid \\
            --num-rows 2 \\
            video1.mp4 video2.mp4 video3.mp4 video4.mp4 video5.mp4 video6.mp4 \\
            output.mp4

    \b
    The above command creates a grid layout as follows
    video1.mp4    video2.mp4    video3.mp4
    video4.mp4    video5.mp4    video6.mp4

    \b
    To indicate an empty spot in the grid, pass '' as the video path, like so:
        vid grid \\
            --num-rows 2 \\
            video1.mp4 '' video3.mp4 video4.mp4 \\
            output.mp4

    The empty spot will be filled with a black image.
    """
    blank_path = ''
    videos = validate_globbed_paths_allow_dummy(videos, dummy_path=blank_path)

    from utils.moviepy_wrappers.composite_clip import clips_array_maybe_none

    if len(videos) % num_rows != 0:
        raise ValueError('Number of videos (%s) is not evenly divisible by '
                         '--num_rows (%s). This is not supported right now.' %
                         (len(videos), num_rows))

    num_cols = len(videos) / num_rows

    from moviepy.video.io.VideoFileClip import VideoFileClip
    grid = [[] for _ in range(num_rows)]

    for i, video in enumerate(videos):
        row = int(i // num_cols)
        if video != blank_path:
            clip = VideoFileClip(str(video))
        else:
            clip = None
        grid[row].append(clip)

    output_clip = clips_array_maybe_none(grid)
    if not save_audio:
        output_clip = output_clip.without_audio()
    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument('video', type=click.Path(exists=True))
def info(video):
    """Report video duration, fps, and resolution."""
    from moviepy.video.io.VideoFileClip import VideoFileClip
    clip = VideoFileClip(video)
    info = {
        'Path': Path(video).resolve(),
        'Duration': clip.duration,
        'FPS': clip.fps,
        'Resolution': f'{clip.size[0]}x{clip.size[1]}'
    }
    max_width = max(len(x) for x in info)
    for key, value in info.items():
        # Right align all names for pretty output.
        key_pretty = key.rjust(max_width)
        print(f"{key_pretty}: {value}")


if __name__ == '__main__':
    main()
