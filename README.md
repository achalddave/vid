# vid

`vid` is a command line tool for simple video manipulation. Current features:

* `vid slideshow "frames/*.png" video.mp4`: Create a video from a sequence of
  images.
* `vid info video.mp4`: Report video duration, fps, and resolution.
* `vid [hstack|vstack|grid] video1.mp4 video2.mp4 output.mp4`: Arrange multiple
  videos next to each other. `hstack` puts videos next to each other; `vstack`
  puts them on top of each other; `grid` creates a grid that can be controlled
  with the `--num-rows` flag.
* `vid download 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' -s 42 -d 2`:
  Download videos using [youtube-dl](https://github.com/rg3/youtube-dl/)
  and optionally trim to specified start/end times (in this example, trim a
  2 second clip starting at t=42s).
* `vid dump_frames video.mp4 ./video_frames`: Dump frames for a video or list
  of videos. Dumps frames in parallel by default if multiple videos are
  specified.

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
