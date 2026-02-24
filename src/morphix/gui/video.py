import ffmpegio
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm


def resample_video(
    frames: np.ndarray, rates: dict[int, float]
) -> tuple[np.ndarray, float]:
    num_frames = frames.shape[0]

    # figure out for how long to show each original frame
    durations = compute_frame_durations(num_frames, rates)
    min_duration = np.min(durations)

    # approximate other original frame durations by showing them for the given
    # number of resampled frames
    repeats = np.round(durations / min_duration).astype(np.int32)

    # we set the fps to support the shortest frame duration
    fps = 1.0 / min_duration

    # we don't want the final fps to exceed a certain threshold, so we
    # interpolate over time if that's the case
    bin_size = 1
    max_fps = 50.0
    if fps > max_fps:
        print(f"FPS would be {fps}, limiting it to {max_fps}...")
        fps_correction_factor = max_fps / fps
        bin_size = round(1.0 / fps_correction_factor)
        fps = max_fps
        print(f"-> interpolating sets of {bin_size} frames")

    # resample the original frames in batches
    batch_size = 10
    resampled_frames = []
    print("Resampling frames...")
    for t in tqdm(range(0, num_frames, batch_size)):
        resampled = resample_frames(
            frames[t : t + batch_size], repeats[t : t + batch_size], bin_size
        )
        resampled_frames.append(resampled)
    resampled_frames = np.concatenate(resampled_frames)

    return resampled_frames, fps


def resample_frames(frames, repeats, bin_size=1):
    # print(f"Resampling frames of shape {frames.shape}")

    resampled_frames = np.repeat(frames, repeats, axis=0)
    # print(f"After resampling, shape of frames is {resampled_frames.shape}")

    if bin_size > 1:
        length, height, width, channels = resampled_frames.shape
        num_bins = length // bin_size

        # print(
        #     f"Reshaping into ({num_bins}, {bin_size}, {height}, {width}, {channels})..."
        # )
        resampled_frames = resampled_frames[: num_bins * bin_size].reshape(
            num_bins, bin_size, height, width, channels
        )

        # print("Averaging...")
        resampled_frames = np.mean(resampled_frames, axis=1, dtype=np.float32)
        resampled_frames = resampled_frames.astype(np.uint8)

    return resampled_frames


def interpolate_frames(frames, factor):
    frames = jnp.array(frames)

    # step 1: smooth frames over time with an exponential moving average
    alpha = factor

    def smooth(prev_smoothed, current_frame):
        smoothed = (1.0 - alpha) * prev_smoothed + alpha * current_frame
        smoothed = smoothed.astype(jnp.uint8)
        return smoothed, smoothed

    _, smoothed = jax.lax.scan(smooth, frames[0], frames)

    # step 2: downsample
    skip = round(1.0 / factor)
    return np.array(smoothed[::skip])


def compute_frame_durations(num_frames, rates: dict[int, float]) -> np.ndarray:
    """Given a few key frame rates, compute for how long each frame should be visible.

    Uses PCHIP interpolation between key frame rates for smooth acceleration
    and deceleration.

    Example:
        If rates is `{0: 1.0, 10: 1.0, 20: 25.0, 40: 1.0, 49: 1.0}` and the
        total number of frames is 50, the first and last 10 frames will be
        shown for one second each, and the frames in between will accelerate
        until 25fps and slow down again.
    """
    key_frames = np.array(list(sorted(rates.keys())))
    key_rates = np.array([rates[f] for f in key_frames])

    spline = PchipInterpolator(key_frames, key_rates)
    frames = np.arange(0, num_frames, 1.0)
    rates = spline(frames)

    durations = 1.0 / rates
    return durations


def encode_video(filename, frames, rate):
    ffmpegio.video.write(filename, rate, frames, show_log=True, overwrite=True)
