import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import glob
import argparse
import multiprocessing as mp
from functools import partial
import cv2
import time
import sys
import threading

# Create a lock for synchronized printing
print_lock = threading.Lock()


def synchronized_print(msg, end="\n"):
    """Print with lock to prevent output from being garbled in multiprocessing"""
    with print_lock:
        print(msg, end=end)
        sys.stdout.flush()


# Use paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(script_dir, ".")  # Current directory
output_dir = os.path.join(script_dir, "..", "density_maps")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate density maps from UMANS simulation data"
    )
    parser.add_argument("--input", default=None, help="Directory containing CSV files")
    parser.add_argument(
        "--output", default=None, help="Directory to save output images"
    )
    parser.add_argument("--fps", type=int, default=4, help="Frames per second")
    parser.add_argument(
        "--resolution", type=int, default=100, help="Resolution of density maps"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove PNG files after video creation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Generate minimal visualizations without titles, colorbars, or axes",
    )
    return parser.parse_args()


def get_scenario_bounds(scenario_name):
    """Return the bounds for each scenario based on its XML definition"""
    # This dictionary stores the x_min, x_max, y_min, y_max for each scenario
    bounds = {
        "scenario1_choke_point": (-5, 35, -10, 10),
        "scenario2_bidirectional_flow": (-30, 30, -5, 5),
        "scenario3_concert_exit": (-30, 30, -35, 30),
        "scenario4_stadium_entrance": (-40, 40, -25, 25),
        "scenario5_intersection_crossing": (-30, 30, -30, 30),
        "scenario6_emergency_evacuation": (-25, 25, -25, 25),
        "scenario7_festival_crowd": (-40, 40, -40, 40),
        "scenario8_merging_flows": (-50, 50, -45, 5),
        "scenario9_train_platform": (-50, 50, -20, 15),
        "scenario10_stadium_competition": (-50, 50, -50, 50),
    }

    for name in bounds:
        if name in scenario_name:
            return bounds[name]

    # Default bounds if scenario not found
    return (-50, 50, -50, 50)


def find_closest_timestamp(timestamps, target_time):
    """Find closest timestamp using binary search"""
    idx = np.searchsorted(timestamps, target_time)
    if idx == 0:
        return timestamps[0]
    elif idx == len(timestamps):
        return timestamps[-1]
    else:
        before = timestamps[idx - 1]
        after = timestamps[idx]
        return before if target_time - before < after - target_time else after


def create_density_map(x_positions, y_positions, bounds, resolution):
    """Create density map with consistent cell sizing"""
    x_min, x_max, y_min, y_max = bounds

    # Calculate physical dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Determine resolution to maintain square cells
    if width > height:
        x_bins = resolution
        y_bins = int(resolution * (height / width))
    else:
        y_bins = resolution
        x_bins = int(resolution * (width / height))

    H, xedges, yedges = np.histogram2d(
        x_positions,
        y_positions,
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    return H, xedges, yedges


# Add a simple progress display function
def print_progress(
    iteration, total, prefix="", suffix="", length=50, fill="█", print_end="\r"
):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# Create a manager for sharing progress between processes
def init_process(counter, total, counter_lock):
    global shared_counter, shared_total, shared_lock
    shared_counter = counter
    shared_total = total
    shared_lock = counter_lock


def process_scenario(
    scenario_name, files, output_dir, fps, resolution, clean_pngs, minimal=False
):
    """Process a single scenario (can be called in parallel)"""
    # Create a scenario-specific logger to avoid overlapping output
    log_prefix = f"[{scenario_name}] "

    synchronized_print(f"\n{log_prefix}Starting with {len(files)} files")

    # Determine scenario bounds
    x_min, x_max, y_min, y_max = get_scenario_bounds(scenario_name)

    # Read all files for this scenario to determine time range
    all_dfs = []
    for file in files:
        df = pd.read_csv(file, header=None)
        # Extract agent ID from filename if available
        agent_type = "unknown"
        if "_agent_" in file:
            agent_type = file.split("_agent_")[1].replace(".csv", "")
        df["agent_type"] = agent_type
        all_dfs.append(df)

    synchronized_print(f"{log_prefix}Combining data from {len(files)} files")
    combined_df = pd.concat(all_dfs)

    # Standard processing code
    time_col, x_col, y_col = 0, 1, 2
    timestamps = sorted(combined_df.iloc[:, time_col].unique())
    time_interval = 1.0 / fps
    time_points = np.arange(min(timestamps), max(timestamps), time_interval)

    # Create output folders
    scenario_output_dir = os.path.join(output_dir, scenario_name)
    frames_dir = os.path.join(scenario_output_dir, "frames")
    os.makedirs(scenario_output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Process frames - print progress at regular intervals
    synchronized_print(f"{log_prefix}Generating {len(time_points)} frames")
    frame_count = 0
    progress_interval = max(1, len(time_points) // 50)

    for i, t in enumerate(time_points):
        # Show progress at intervals without carriage returns
        if i % progress_interval == 0:
            percentage = (i / len(time_points)) * 100
            synchronized_print(
                f"{log_prefix}Processing frames: {i}/{len(time_points)} ({percentage:.1f}%)"
            )

        # Find closest timestamp
        closest_time = find_closest_timestamp(timestamps, t)

        # Get positions at this time
        time_data = combined_df[combined_df.iloc[:, time_col] == closest_time]

        if len(time_data) > 0:
            # Create and save frame
            x_positions = time_data.iloc[:, x_col].values
            y_positions = time_data.iloc[:, y_col].values

            # Create density map
            H, xedges, yedges = create_density_map(
                x_positions, y_positions, (x_min, x_max, y_min, y_max), resolution
            )

            # Normalize density
            H = H / H.max() if H.max() > 0 else H

            # Create figure with appropriate size
            if minimal:
                # For minimal visualization, use a figure with exact dimensions and no padding
                fig = plt.figure(figsize=(10, 10), frameon=False)
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
                fig.add_axes(ax)
            else:
                # For standard visualization with annotations
                fig, ax = plt.figure(figsize=(10, 10)), plt.gca()

            # Plot density map
            im = ax.imshow(
                H.T,  # Transpose for correct orientation
                cmap="hot",
                norm=Normalize(0, 1),
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
            )

            # Add colorbar and labels only if not minimal
            if not minimal:
                plt.colorbar(im, ax=ax, label="Normalized Density")
                ax.set_title(f"{scenario_name} - Time: {closest_time:.2f}s")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
            else:
                # For minimal view, ensure everything is off
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.tight_layout(pad=0)

            # Save figure in frames subfolder
            plt.savefig(
                os.path.join(frames_dir, f"frame_{frame_count:05d}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        frame_count += 1

    # Final frame progress
    synchronized_print(f"{log_prefix}Created {frame_count} frames")

    # Create video
    png_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

    if png_files:
        synchronized_print(f"{log_prefix}Creating video from {len(png_files)} frames")

        # Read first image to get dimensions
        first_img = cv2.imread(png_files[0])
        height, width, layers = first_img.shape

        # Create video writer in scenario folder
        video_path = os.path.join(scenario_output_dir, f"{scenario_name}_density.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Add frames to video with less frequent progress updates
        video_progress_interval = max(1, len(png_files) // 50)
        for i, file in enumerate(png_files):
            if i % video_progress_interval == 0:
                percentage = (i / len(png_files)) * 100
                synchronized_print(
                    f"{log_prefix}Video progress: {i}/{len(png_files)} ({percentage:.1f}%)"
                )

            img = cv2.imread(file)
            video.write(img)

        # Release video writer
        video.release()
        synchronized_print(f"{log_prefix}Video saved to {os.path.basename(video_path)}")

        # Clean up PNG files if requested
        if clean_pngs:
            synchronized_print(f"{log_prefix}Cleaning up {len(png_files)} PNG files")
            for file in png_files:
                try:
                    os.remove(file)
                except Exception as e:
                    synchronized_print(
                        f"{log_prefix}Error removing {os.path.basename(file)}: {e}"
                    )
            # Try to remove the frames directory if empty
            try:
                os.rmdir(frames_dir)
                synchronized_print(f"{log_prefix}Removed empty frames directory")
            except:
                pass  # Directory might not be empty or might have permission issues

    # Update shared counter for overall progress
    if "shared_counter" in globals() and "shared_lock" in globals():
        with shared_lock:
            shared_counter.value += 1
            counter_val = shared_counter.value
            percent_done = (counter_val / shared_total.value) * 100
            synchronized_print(
                f"OVERALL PROGRESS: {counter_val}/{shared_total.value} scenarios done ({percent_done:.1f}%)"
            )

    return f"Completed {scenario_name}"


def process_scenario_wrapper(args):
    """Wrapper function for process_scenario to use with multiprocessing"""
    try:
        name, files, output_dir, fps, resolution, clean_pngs, minimal = args
        return process_scenario(
            name, files, output_dir, fps, resolution, clean_pngs, minimal
        )
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")
        raise


def create_density_maps(
    csv_dir,
    output_dir,
    fps=4,
    resolution=100,
    clean_pngs=False,
    max_workers=None,
    minimal=False,
):
    """Create density map images from CSV files with synchronized output"""
    # Initialize the print lock globally for all processes
    global print_lock
    print_lock = mp.Manager().Lock()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group CSV files by scenario
    scenario_files = {}
    for csv_file in glob.glob(os.path.join(csv_dir, "*.csv")):
        basename = os.path.basename(csv_file)
        # Extract scenario name
        if "_agent_" in basename:
            scenario_name = basename.split("_agent_")[0]
        else:
            scenario_name = basename.replace(".csv", "")

        if scenario_name not in scenario_files:
            scenario_files[scenario_name] = []
        scenario_files[scenario_name].append(csv_file)

    # Show overall progress
    total_scenarios = len(scenario_files)
    synchronized_print(f"Found {total_scenarios} scenarios to process")

    # Determine number of workers (default to CPU count - 1)
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    synchronized_print(f"Processing with {max_workers} parallel workers")

    # Create shared counter for progress tracking
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    total_counter = manager.Value("i", total_scenarios)
    counter_lock = manager.Lock()

    # Create list of tasks
    tasks = [
        (name, files, output_dir, fps, resolution, clean_pngs, minimal)
        for name, files in scenario_files.items()
    ]

    # Use pool with initializer to share counter
    with mp.Pool(
        max_workers,
        initializer=init_process,
        initargs=(counter, total_counter, counter_lock),
    ) as pool:
        # Process scenarios in parallel
        results = pool.map(process_scenario_wrapper, tasks)

    synchronized_print(f"\nAll {total_scenarios} scenarios processed successfully!")


def create_video(image_files, output_path, fps, format="mp4"):
    """Create video from image files with specified format"""
    try:
        if not image_files:
            return False

        first_img = cv2.imread(image_files[0])
        height, width, layers = first_img.shape

        # Select codec and extension based on format
        if format.lower() == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = f"{output_path}.mp4"
        elif format.lower() == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_path = f"{output_path}.avi"
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = f"{output_path}.mp4"

        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for file in image_files:
            img = cv2.imread(file)
            video.write(img)

        video.release()
        return True
    except ImportError:
        print("OpenCV not installed, skipping video creation")
        return False


if __name__ == "__main__":
    # Create density maps with 4 frames per second
    args = parse_args()
    csv_dir = args.input or os.path.join(script_dir, ".")
    output_dir = args.output or os.path.join(script_dir, "..", "density_maps")
    create_density_maps(
        csv_dir,
        output_dir,
        fps=args.fps,
        resolution=args.resolution,
        clean_pngs=args.clean,
        max_workers=args.workers,
        minimal=args.minimal,
    )

    print("Density map generation complete!")
