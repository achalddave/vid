"""Helper functions for dumping frames from video."""

import functools
import json
import logging
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from tqdm import tqdm


def are_frames_dumped(video_path,
                      output_dir,
                      expected_fps,
                      expected_info_path,
                      expected_name_format,
                      log_reason=False):
    """Check if the output directory exists and has already been processed.

        1) Check the info.json file to see if the parameters match.
        2) Ensure that all the frames exist.

    Params:
        video_path (str)
        output_dir (str)
        expected_fps (num)
        expected_info_path (str)
        expected_name_format (str)
    """
    # Ensure that info file exists.
    if not os.path.isfile(expected_info_path):
        if log_reason:
            logging.info("Info path doesn't exist at %s" % expected_info_path)
        return False

    # Ensure that info file is valid.
    with open(expected_info_path, 'r') as info_file:
        info = json.load(info_file)
    info_valid = info['frames_per_second'] == expected_fps \
        and info['input_video_path'] == os.path.abspath(video_path)
    if not info_valid:
        if log_reason:
            logging.info("Info file (%s) is invalid" % expected_info_path)
        return False

    # Check that all frame paths exist.
    offset_if_one_indexed = 0
    if not os.path.exists(expected_name_format % 0):
        # If the 0th frame doesn't exist, either we haven't dumped the frames,
        # or the frames start with index 1 (this changed between versions of
        # moviepy, so we have to explicitly check). We can assume they start
        # with index 1, and continue.
        offset_if_one_indexed = 1

    # https://stackoverflow.com/a/28376817/1291812
    num_frames_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    expected_num_frames = subprocess.check_output(num_frames_cmd,
                                                  stderr=subprocess.STDOUT)
    expected_num_frames = int(expected_num_frames.decode().strip())
    expected_frame_paths = [
        expected_name_format % (i + offset_if_one_indexed)
        for i in range(expected_num_frames)
    ]
    missing_frames = [x for x in expected_frame_paths if not os.path.exists(x)]
    if missing_frames:
        if log_reason:
            logging.info("Missing frames:\n%s" % ('\n'.join(missing_frames)))
        return False

    # All checks passed
    return True


def dump_frames(video_path, output_dir, fps, logger_name=None):
    """Dump frames at frames_per_second from a video to output_dir.

    If frames_per_second is None, the clip's fps attribute is used instead."""
    output_dir.mkdir(exist_ok=True, parents=True)

    if logger_name:
        file_logger = logging.getLogger(logger_name)
    else:
        file_logger = logging.root

    try:
        video_info = ffmpeg_parse_infos(video_path)
        video_fps = video_info['video_fps']
    except OSError:
        logging.error('Unable to open video (%s), skipping.' % video_path)
        logging.exception('Exception:')
        return
    except KeyError:
        logging.error('Unable to extract metadata about video (%s), skipping.'
                      % video_path)
        logging.exception('Exception:')
        return
    info_path = '{}/info.json'.format(output_dir)
    name_format = '{}/frame%04d.png'.format(output_dir)

    if fps is None or fps == 0:
        fps = video_fps  # Extract all frames

    are_frames_dumped_wrapper = functools.partial(
        are_frames_dumped,
        video_path=video_path,
        output_dir=output_dir,
        expected_fps=fps,
        expected_info_path=info_path,
        expected_name_format=name_format)

    if are_frames_dumped_wrapper(log_reason=False):
        file_logger.info('Frames for {} exist, skipping...'.format(video_path))
        return

    successfully_wrote_images = False
    try:
        if fps == video_fps:
            cmd = ['ffmpeg', '-i', video_path, name_format]
        else:
            cmd = ['ffmpeg', '-i', video_path, '-vf',
                   'fps={}'.format(fps), name_format]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        successfully_wrote_images = True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to dump images for %s", video_path)
        logging.error(e)
        logging.error(e.output.decode('utf-8'))

    if successfully_wrote_images:
        info = {'frames_per_second': fps,
                'input_video_path': os.path.abspath(video_path)}
        with open(info_path, 'w') as info_file:
            json.dump(info, info_file)

        if not are_frames_dumped_wrapper(log_reason=True):
            logging.error(
                "Images for {} don't seem to be dumped properly!".format(
                    video_path))


def dump_frames_star(args):
    """Calls dump_frames after unpacking arguments."""
    return dump_frames(*args)


def dump_frames_parallel(videos, output_dir, fps, num_workers):
    if fps == 0:
        fps = None

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dump_frames_tasks = []
    for video_path in videos:
        output_video_dir = output_dir / video_path.stem
        dump_frames_tasks.append(
            (video_path, output_video_dir, fps))

    if num_workers > 1:
        pool = Pool(num_workers)
        try:
            list(
                tqdm(pool.imap_unordered(dump_frames_star, dump_frames_tasks),
                     total=len(dump_frames_tasks)))
        except KeyboardInterrupt:
            print('Parent received control-c, exiting.')
            pool.terminate()
    else:
        for task in tqdm(dump_frames_tasks):
            dump_frames_star(dump_frames_tasks)
