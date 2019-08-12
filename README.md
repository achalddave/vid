# vid

`vid` is a command line tool for simple video manipulation. Current features:

* Create video from a sequences of images:
    ```bash
    ~ vid slideshow "frames/*.png" video.mp4
    ~ vid slideshow --fps 1 "frames/*.png" video_1fps.mp4
    ```
* Report video information:
    ```bash
    ~ vid info video.mp4
            Path: video.mp4
        Duration: 3.03
             FPS: 30.0
      Resolution: 1280x720
    ```
* Combine multiple videos into a single video:
    ```bash
    # Combine videos horizontally
    ~ vid hstack video1.mp4 video2.mp4 output.mp4
    # Combine videos vertically
    ~ vid vstack video1.mp4 video2.mp4 output.mp4
    # Combine videos in a grid
    ~ vid grid --num-rows 2 video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
        output.mp4
    ```
* Download videos and trim to specified start/end times. Uses
  [youtube-dl](https://github.com/rg3/youtube-dl/).
  ```bash
  # Download video, trim clip of 2 second duration starting at t=42s.
  ~ vid download 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' -s 42 -d 2
  ```
* Dump frames for a video or list of videos. Dumps frames in parallel by
  default if multiple videos are specified.
    ```bash
    ~ vid dump_frames video.mp4 ./video_frames
    ~ vid dump_frames --list list_of_videos.txt ./video_frames
    ```

See `vid <command> --help` for more info.

`vid` is primarily a wrapper around some
[`moviepy`](https://github.com/Zulko/moviepy) and [`ffmpeg`](http://ffmpeg.org/)
with a simple, easy-to-remember set of commands.

## Installation

```python
pip install -e 'git+https://github.com/achalddave/vid.git#egg=vid'
```

## Known Issues
NOTE: `vid` is intended for simple visualizations with minimal effort. Some
known issues:

* `slideshow` is slower than using ffmpeg to create a video. This is
  likely because `vid` uses [`moviepy`](https://github.com/Zulko/moviepy)'s
  `FFMPEG_VideoWriter`, writing individual frames to ffmpeg's stdin to
  create the video. This allows us to handle images of varying sizes and
  formats. I haven't found a fool-proof way to ask ffmpeg to create a video
  from a list of jpg and png images, for example.

* `hstack`/`vstack`/`grid` sometimes end up with a couple blank frames if
  the input video frame rates vary.
