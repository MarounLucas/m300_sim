"""Module for parsing and visualizing ROS 2 bags.

This module reads telemetry and trajectory data from a specified ROS bag,
normalizes the time, exports the data to CSV files, and generates a
comprehensive dashboard for flight analysis.
"""

import glob
import os
from typing import Dict, List, Tuple

import matplotlib
# FORCE HEADLESS MODE (To resolve Docker/X11 error)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

# =========================================================================
# CONSTANTS & CONFIGURATION
# =========================================================================
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def get_latest_bag(bags_dir: str) -> str:
    """Finds the most recently modified ROS bag directory.

    Searches for directories matching the 'unified_flight_*' pattern.

    Args:
        bags_dir (str): The directory path containing the ROS bags.

    Returns:
        str: The path to the most recently modified bag folder.

    Raises:
        FileNotFoundError: If no matching bag folder is found.
    """
    bag_pattern = os.path.join(bags_dir, "unified_flight_*")
    bag_folders = glob.glob(bag_pattern)
    
    if not bag_folders:
        raise FileNotFoundError(f"No bags found in folder: {bags_dir}")
        
    return max(bag_folders, key=os.path.getmtime)


def extract_bag_data(bag_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts telemetry and trajectory data from a ROS bag.

    Args:
        bag_path (str): The file path to the ROS bag.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the telemetry
        DataFrame and the trajectory DataFrame.
    """
    telemetry_data: Dict[str, List[float]] = {
        "time": [], "x": [], "y": [], "z": [],
        "roll": [], "pitch": [], "yaw": [],
        "u": [], "v": [], "w": []
    }
    trajectory_data: Dict[str, List[float]] = {
        "time": [], "x_ref": [], "y_ref": [], "z_ref": [],
        "yaw_ref": [], "u_ref": [], "v_ref": [], "w_ref": []
    }

    print("Extracting messages from the database...")
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            time_sec = timestamp / 1e9

            # --- Telemetry Topic (Actual Drone State) ---
            if connection.topic == "/m300_sim/telemetry_topic":
                msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)

                telemetry_data["time"].append(time_sec)
                telemetry_data["x"].append(msg.pose.pose.position.x)
                telemetry_data["y"].append(msg.pose.pose.position.y)
                telemetry_data["z"].append(msg.pose.pose.position.z)
                telemetry_data["u"].append(msg.twist.twist.linear.x)
                telemetry_data["v"].append(msg.twist.twist.linear.y)
                telemetry_data["w"].append(msg.twist.twist.linear.z)

                q = [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
                r, p, y = Rotation.from_quat(q).as_euler(
                    "xyz", degrees=True
                )
                telemetry_data["roll"].append(r)
                telemetry_data["pitch"].append(p)
                telemetry_data["yaw"].append(y)

            # --- Trajectory Topic (Desired State / Reference) ---
            elif connection.topic == "/m300_sim/trajectory_topic":
                msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)

                trajectory_data["time"].append(time_sec)
                trajectory_data["x_ref"].append(msg.pose.pose.position.x)
                trajectory_data["y_ref"].append(msg.pose.pose.position.y)
                trajectory_data["z_ref"].append(msg.pose.pose.position.z)
                trajectory_data["u_ref"].append(msg.twist.twist.linear.x)
                trajectory_data["v_ref"].append(msg.twist.twist.linear.y)
                trajectory_data["w_ref"].append(msg.twist.twist.linear.z)

                q = [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
                _, _, y_ref = Rotation.from_quat(q).as_euler(
                    "xyz", degrees=True
                )
                trajectory_data["yaw_ref"].append(y_ref)

    df_tel = pd.DataFrame(telemetry_data)
    df_traj = pd.DataFrame(trajectory_data)

    return df_tel, df_traj


def normalize_time(
    df_tel: pd.DataFrame, df_traj: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalizes the time columns to start from zero.

    Args:
        df_tel (pd.DataFrame): Telemetry DataFrame.
        df_traj (pd.DataFrame): Trajectory DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The normalized DataFrames.
    """
    if df_tel.empty:
        return df_tel, df_traj

    t_start = df_tel["time"].min()
    if not df_traj.empty:
        t_start = min(t_start, df_traj["time"].min())
        df_traj["time"] = df_traj["time"] - t_start

    df_tel["time"] = df_tel["time"] - t_start
    return df_tel, df_traj


def generate_analysis_dashboard(
    df_tel: pd.DataFrame, df_traj: pd.DataFrame, plot_path: str
) -> None:
    """Generates and saves a graphical dashboard of the flight data.

    Args:
        df_tel (pd.DataFrame): Telemetry data.
        df_traj (pd.DataFrame): Reference trajectory data.
        plot_path (str): File path to save the dashboard image.
    """
    print("Generating unified analysis plots...")
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: 3D Trajectory (Ref vs Actual)
    ax1 = fig.add_subplot(2, 3, (1, 4), projection="3d")
    if not df_traj.empty:
        ax1.plot(
            df_traj["x_ref"].to_numpy(),
            df_traj["y_ref"].to_numpy(),
            df_traj["z_ref"].to_numpy(),
            label="Reference (Blue)",
            linestyle="--",
            color="blue",
            linewidth=2,
        )
    ax1.plot(
        df_tel["x"].to_numpy(),
        df_tel["y"].to_numpy(),
        df_tel["z"].to_numpy(),
        label="Actual (Red)",
        color="red",
        alpha=0.8,
    )
    ax1.scatter(
        df_tel["x"].iloc[0],
        df_tel["y"].iloc[0],
        df_tel["z"].iloc[0],
        color="green",
        s=100,
        label="Start",
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory (Ref vs Actual)")
    ax1.legend()

    # Plot 2: XYZ Tracking over time
    ax2 = fig.add_subplot(2, 3, 2)
    if not df_traj.empty:
        ax2.plot(
            df_traj["time"].to_numpy(),
            df_traj["x_ref"].to_numpy(),
            "b--",
            alpha=0.5,
        )
        ax2.plot(
            df_traj["time"].to_numpy(),
            df_traj["y_ref"].to_numpy(),
            "g--",
            alpha=0.5,
        )
        ax2.plot(
            df_traj["time"].to_numpy(),
            df_traj["z_ref"].to_numpy(),
            "m--",
            alpha=0.5,
        )
    ax2.plot(
        df_tel["time"].to_numpy(), df_tel["x"].to_numpy(),
        "b-", label="Actual X"
    )
    ax2.plot(
        df_tel["time"].to_numpy(), df_tel["y"].to_numpy(),
        "g-", label="Actual Y"
    )
    ax2.plot(
        df_tel["time"].to_numpy(), df_tel["z"].to_numpy(),
        "m-", label="Actual Z"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("XYZ Positions over time")
    ax2.legend()

    # Plot 3: Yaw Tracking over time
    ax3 = fig.add_subplot(2, 3, 3)
    if not df_traj.empty:
        ax3.plot(
            df_traj["time"].to_numpy(),
            df_traj["yaw_ref"].to_numpy(),
            "b--",
            label="Ref Yaw",
        )
    ax3.plot(
        df_tel["time"].to_numpy(), df_tel["yaw"].to_numpy(),
        "r-", label="Actual Yaw"
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (degrees)")
    ax3.set_title("Orientation Tracking (Yaw)")
    ax3.legend()

    # Plot 4: Attitude (Roll, Pitch)
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(
        df_tel["time"].to_numpy(), df_tel["roll"].to_numpy(),
        label="Roll (Lat. Tilt)", color="red"
    )
    ax4.plot(
        df_tel["time"].to_numpy(), df_tel["pitch"].to_numpy(),
        label="Pitch (Long. Tilt)", color="green"
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (degrees)")
    ax4.set_title("Attitude Effort (Check Smoothness)")
    ax4.legend()

    # Plot 5: Linear Velocities (u, v, w)
    ax5 = fig.add_subplot(2, 3, 6)
    if not df_traj.empty:
        ax5.plot(
            df_traj["time"].to_numpy(),
            df_traj["u_ref"].to_numpy(),
            "r--",
            alpha=0.3,
        )
        ax5.plot(
            df_traj["time"].to_numpy(),
            df_traj["w_ref"].to_numpy(),
            "b--",
            alpha=0.3,
        )
    ax5.plot(
        df_tel["time"].to_numpy(), df_tel["u"].to_numpy(),
        label="u (vel x)", color="red"
    )
    ax5.plot(
        df_tel["time"].to_numpy(), df_tel["v"].to_numpy(),
        label="v (vel y)", color="green"
    )
    ax5.plot(
        df_tel["time"].to_numpy(), df_tel["w"].to_numpy(),
        label="w (vel z)", color="blue"
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Velocity (m/s)")
    ax5.set_title("Linear Velocity Profile")
    ax5.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graphical dashboard successfully saved to: {plot_path}")


# =========================================================================
# MAIN FUNCTION
# =========================================================================
def process_bag(bag_path: str) -> None:
    """Processes the ROS bag, saves CSV data, and generates a dashboard.

    Orchestrates data extraction, time normalization, CSV saving, and
    plot generation for the specified bag.

    Args:
        bag_path (str): The file path to the ROS bag.
    """
    bag_name = os.path.basename(bag_path)
    print("=" * 50)
    print(f" Starting analysis for UNIFIED bag: {bag_name}")
    print("=" * 50)

    df_tel, df_traj = extract_bag_data(bag_path)

    if df_tel.empty:
        print("ERROR: No telemetry data found in the bag!")
        return

    df_tel, df_traj = normalize_time(df_tel, df_traj)

    results_dir = os.path.join(bag_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Saving CSV files to: {results_dir}")
    tel_csv = os.path.join(results_dir, "unified_telemetry.csv")
    df_tel.to_csv(tel_csv, index=False)
    
    if not df_traj.empty:
        traj_csv = os.path.join(results_dir, "reference_trajectory.csv")
        df_traj.to_csv(traj_csv, index=False)

    plot_path = os.path.join(results_dir, "flight_analysis_dashboard.png")
    generate_analysis_dashboard(df_tel, df_traj, plot_path)
    
    print("Processing completed!")


if __name__ == "__main__":
    workspace_bags_dir = os.path.join(
        os.getcwd(), "src", "m300_sim", "bags"
    )
    try:
        latest_bag = get_latest_bag(workspace_bags_dir)
        process_bag(latest_bag)
    except Exception as err:
        print(f"Error processing data: {err}")