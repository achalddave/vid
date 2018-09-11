# vid

`vid` is a command line tool for simple video manipulation. Current features:

* **slideshow**: Create a video from a sequence of images.
* **info**: Report video duration, fps, and resolution.
* **hstack**/**vstack**/**grid**: Arrange multiple videos in a specified layout,
  combining them into a single video.

See `vid <command> --help` for more info.

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
