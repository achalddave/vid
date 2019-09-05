#!/usr/bin/env python3

import glob
import logging
from pathlib import Path

import click
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm

from .utils import split_frames


codec_param = click.option('--codec', type=str, default=None)
verbose_param = click.option('--verbose/--no-verbose', default=False)
audio_param = click.option('--audio/--no-audio', 'save_audio', default=True)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


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


def clip_from_path(path):
    if path == '':
        return None
    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        from moviepy.video.VideoClip import ImageClip
        return ImageClip(str(path))
    else:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        return VideoFileClip(str(path))


def set_image_clip_duration(clips):
    """Set any ImageClip's duration to that of the first clip with a duration.
    """
    if isinstance(clips[0], list):
        clips = flatten_once(clips)

    duration = None
    for clip in clips:
        if hasattr(clip, 'duration') and clip.duration is not None:
            duration = clip.duration
            break
    else:
        raise ValueError('Could not infer duration of clips.')

    for i, clip in enumerate(clips):
        if not hasattr(clip, 'duration') or clip.duration is None:
            clip.duration = duration
            if not hasattr(clip, 'end') or clip.end is None:
                clip.end = duration


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
@click.argument(
    'output', type=click.Path(file_okay=False, dir_okay=False), required=True)
@click.option('--fps', type=float, default=30)
@click.option(
    '--shape',
    type=str,
    help=('Specify size of output video, in WxH format. By default, use the '
          'size of the first frame.'),
    default=None)
@click.option(
    '-j', '--num-workers',
    default=8,
    type=int,
    help='Number of threads for loading images.')
@click.option(
    '--buffer-size',
    default=None,
    type=int,
    help='Number of images to preload. Default: 10 * --num-workers')
@codec_param
@verbose_param
def slideshow(images, output, fps, shape, num_workers, buffer_size, codec,
              verbose):
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
    import numpy as np
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

    def load_frame_raw(frame_index):
        image = Image.open(images[frame_index])
        # Handle palette-based images by replacing pixels with the
        # respective color from the palette.
        if image.palette is not None:
            image = image.convert(image.palette.mode)

        if image.size != (width, height):
            image = image.resize((width, height))

        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        if image.ndim == 2:
            image = np.stack((image, image, image), -1)

        # Create a fake alpha channel if the current image doesn't have it.
        if has_alpha and image.shape[2] == 3:
            mask = np.ones((height, width, 1), dtype=np.uint8)
            image = np.dstack((image, mask))
        return image

    if num_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(num_workers)
        load_futures = [None for _ in images]
        if buffer_size is None:
            buffer_size = 10 * num_workers
        for i in range(min(buffer_size, len(images))):
            load_futures[i] = executor.submit(load_frame_raw, i)

        def load_frame(frame_index):
            next_preload = frame_index + buffer_size
            if (next_preload < len(images)
                    and load_futures[next_preload] is None):
                load_futures[next_preload] = executor.submit(
                    load_frame_raw, next_preload)
            return load_futures[frame_index].result()
    else:
        load_frame = load_frame_raw

    last_loaded = {'index': None, 'image': None}

    def make_frame(t):
        image_index = max(
            [i for i in range(len(images)) if image_starts[i] <= t])
        if image_index != last_loaded['index']:
            last_loaded['index'] = image_index
            last_loaded['image'] = load_frame(image_index)
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
            frame = make_frame(t)
            writer.write_frame(frame)


@main.command()
@click.argument('videos', required=True, nargs=-1)
@click.argument(
    'output', type=click.Path(file_okay=False, dir_okay=False), required=True)
@audio_param
@verbose_param
def hstack(videos, output, save_audio, verbose):
    """Merge videos horizontally into one video.

    \b
    Example usage:
        vid hstack first.mp4 second.mp4 third.mp4 output.mp4

    \b
    The above command creates puts three videos side-by-side:
        video1.mp4    video2.mp4    video3.mp4
    """
    blank_path = ''
    videos = validate_globbed_paths_allow_dummy(videos, dummy_path=blank_path)

    from .utils.moviepy_wrappers.composite_clip import clips_array_maybe_none

    clips = [[clip_from_path(v) for v in videos]]
    set_image_clip_duration(clips)
    output_clip = clips_array_maybe_none(clips)
    if not save_audio:
        output_clip = output_clip.without_audio()

    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument('videos', required=True, nargs=-1)
@click.argument(
    'output', type=click.Path(file_okay=False, dir_okay=False), required=True)
@audio_param
@verbose_param
def vstack(videos, output, save_audio, verbose):
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
    blank_path = ''
    videos = validate_globbed_paths_allow_dummy(videos, dummy_path=blank_path)

    from .utils.moviepy_wrappers.composite_clip import clips_array_maybe_none
    clips = [[clip_from_path(v)] for v in videos]
    set_image_clip_duration(clips)
    output_clip = clips_array_maybe_none(clips)
    if not save_audio:
        output_clip = output_clip.without_audio()
    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument('videos', required=True, nargs=-1)
@click.argument(
    'output', type=click.Path(file_okay=False, dir_okay=False), required=True)
@click.option('--num-rows', type=int, default=2)
@audio_param
@verbose_param
def grid(videos, output, num_rows, save_audio, verbose):
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

    from .utils.moviepy_wrappers.composite_clip import clips_array_maybe_none

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

    set_image_clip_duration(grid)
    output_clip = clips_array_maybe_none(grid)
    if not save_audio:
        output_clip = output_clip.without_audio()
    output_clip.write_videofile(output, verbose=verbose, progress_bar=verbose)


@main.command()
@click.argument('video', type=click.Path(exists=True))
def info(video):
    """Report video duration, fps, and resolution."""
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    info = ffmpeg_parse_infos(video)
    output = {
        'Path': Path(video).resolve(),
        'Duration': info['duration'],
        'FPS': info['video_fps'],
        'Resolution':
            '{w}x{h}'.format(w=info["video_size"][0], h=info["video_size"][1])
    }
    max_width = max(len(x) for x in info)
    for key, value in output.items():
        # Right align all names for pretty output.
        key_pretty = key.rjust(max_width)
        print(f"{key_pretty}: {value}")


@main.command()
@click.argument('video', type=click.Path(exists=True))
@click.argument(
    'output', type=click.Path(file_okay=False, dir_okay=False), required=True)
@click.option('--fps', type=float, default=None)
@verbose_param
def copy(video, output, fps, verbose):
    """Create a copy of a video with different settings.

    Currently, this can be used to change the frame rate, but hopefully this
    will later support other tasks like changing the resolution."""
    import subprocess
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    from moviepy.config import get_setting

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    if fps is None:
        raise click.BadParameter(
            'copy currently only supports changing frame rate.')
    original_fps = ffmpeg_parse_infos(video)['video_fps']
    fps_scale = original_fps / fps
    cmd = [
        get_setting("FFMPEG_BINARY"), "-i",
        str(video), "-vf", 'setpts={}*PTS'.format(fps_scale), '-r',
        str(fps),
        str(output)
    ]
    logging.info('Running command: {}'.format(' '.join(cmd)))

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.exception('[vid] Command returned an error: ')
        logging.fatal(e.decode('utf8'))
        return


@main.command()
@click.argument('url', type=str)
@click.argument('output', type=click.Path(file_okay=False, dir_okay=False))
@click.option(
    '-s', '--start-time', type=str, help="Start time offset.")
@click.option('-e', '--end-time', type=str, help="End time offset.")
@click.option(
    '-d',
    '--duration',
    type=str,
    help="Duration offset from start time; an alternative to --end-time.")
@click.option('--youtubedl_args', type=str)
@click.option('--ffmpeg_args', type=str)
@verbose_param
def download(url, output, start_time, end_time, duration, youtubedl_args,
             ffmpeg_args, verbose):
    """Download video using ffmpeg, optionally trimming it.

    Times are interpreted directly by ffmpeg. The accepted syntax is described
    at <https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax>.
    """
    if end_time is not None and duration is not None:
        raise click.BadParameter(
            '--end-time and --duration cannot both be specified.')

    import subprocess
    import sys
    from moviepy.config import get_setting

    Path(output).parent.mkdir(exist_ok=True, parents=True)

    youtubedl_command = ['youtube-dl', '-g', url]
    if youtubedl_args:
        youtubedl_command += youtubedl_args
    else:
        youtubedl_command += ['-f best']

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    logging.info('Running youtube-dl command:\n%s',
                 ' '.join(youtubedl_command))

    try:
        youtubedl_output = subprocess.check_output(youtubedl_command).decode(
            'utf-8')
    except subprocess.CalledProcessError as e:
        logging.exception('Failed command.\nException: %s\nOutput:\n %s',
                          e.returncode, e.output.decode('utf-8'))
        sys.exit(1)

    urls = youtubedl_output.strip().split('\n')
    if len(urls) == 1:
        video_url = audio_url = urls[0]
    elif len(urls) == 2:
        video_url, audio_url = urls
    else:
        raise ValueError('Unknown youtube-dl output:\n%s', youtubedl_output)

    # Final command:
    #   ffmpeg -ss {start} -i {video_url} -ss {start} \
    #       -i {audio_url} -t {duration} -c copy {output}
    # Note that we need two inputs (one for audio and one for video). See
    #   https://github.com/rg3/youtube-dl/issues/622#issuecomment-320962680
    ffmpeg_command = [get_setting('FFMPEG_BINARY')]
    if start_time is not None:
        ffmpeg_command += [
            '-ss', start_time, '-i', video_url, '-ss', start_time, '-i',
            audio_url
        ]
    else:
        ffmpeg_command += ['-i', video_url, '-i', audio_url]

    if end_time is not None:
        ffmpeg_command += ['-to', end_time]
    elif duration is not None:
        ffmpeg_command += ['-t', duration]

    ffmpeg_command += ['-c', 'copy', output]

    logging.info('Running ffmpeg command:\n%s', ' '.join(ffmpeg_command))
    try:
        output = subprocess.check_output(
            ffmpeg_command,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.exception('Failed command.\nException: %s\nOutput:\n %s\n===',
                          e.returncode, e.output.decode('utf-8'))
        sys.exit(1)


# Explicitly name command to prevent click 7.0 from converting command to use
# dash instead of underscore. <https://github.com/pallets/click/issues/1123>
@main.command(name='dump_frames')
@click.argument('video',
                type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option('--extension', default='.png')
@click.option('--list', 'treat_as_list', is_flag=True,
              help='Treat VIDEO as a list of new-line separated video paths.')
@click.option('--fps', type=float, default=0)
@click.option('--qscale',
              type=int,
              default=2,
              help='Qscale for ffmpeg. Only used for JPEG output.')
@click.option(
    '-j', '--num-workers',
    default=8,
    help='Number of processes extracting images if --list is specified.',
    type=int)
def dump_frames(video, output_dir, treat_as_list, fps, num_workers):
    """Dump frames from VIDEO to OUTPUT_DIR.

    \b
    Example usage:
        vid dump_frames video.mp4 /path/to/output/dir
        vid dump_frames videos.txt /path/to/output/dir --list
    """
    output_dir = Path(output_dir)
    if treat_as_list:
        video_list = video
        with open(video_list, 'r') as f:
            videos = [Path(x.strip()) for x in f]
        for i, path in enumerate(videos):
            if not path.exists():
                raise ValueError(
                    f"Path {path} at line {i} in {video_list} does not exist")
        split_frames.dump_frames_parallel(videos, output_dir, fps, num_workers)
    else:
        split_frames.dump_frames(video, output_dir, fps)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')
    main()
