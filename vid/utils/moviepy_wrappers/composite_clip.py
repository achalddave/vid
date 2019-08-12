import numpy as np
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


def clips_array_maybe_none(array,
                           rows_widths=None,
                           cols_widths=None,
                           bg_color=None):
    """
    Like CompositeVideoClip.clips_array, but support empty clips in array.

    Args:
        rows_widths: widths of the different rows in pixels. If None, is set
            automatically.
        cols_widths: widths of the different colums in pixels. If None, is set
            automatically.
        bg_color: Fill color for the masked and unfilled regions. Set to None
            for these regions to be transparent (will be slower).
    """
    array = np.array(array)
    sizes_array = np.array(
        [[c.size if c else (0, 0) for c in line] for line in array])

    # find row width and col_widths automatically if not provided
    if rows_widths is None:
        rows_widths = sizes_array[:, :, 1].max(axis=1)
    if cols_widths is None:
        cols_widths = sizes_array[:, :, 0].max(axis=0)

    rows_widths[rows_widths == 0] = rows_widths.max()
    cols_widths[cols_widths == 0] = cols_widths.max()

    xx = np.cumsum([0] + list(cols_widths))
    yy = np.cumsum([0] + list(rows_widths))

    for j, (x, cw) in list(enumerate(zip(xx[:-1], cols_widths))):
        for i, (y, rw) in list(enumerate(zip(yy[:-1], rows_widths))):
            clip = array[i, j]
            if clip is None:
                continue
            w, h = clip.size
            if (w < cw) or (h < rw):
                clip = (CompositeVideoClip(
                    [clip.set_pos('center')], size=(cw, rw),
                    bg_color=bg_color).set_duration(clip.duration))
            array[i, j] = clip.set_pos((x, y))

    return CompositeVideoClip(
        [x for x in array.flatten() if x is not None],
        size=(xx[-1], yy[-1]),
        bg_color=bg_color)
