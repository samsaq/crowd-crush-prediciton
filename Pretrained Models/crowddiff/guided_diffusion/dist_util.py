"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a single GPU device without distributed training.
    """
    if th.cuda.is_available():
        device = th.device("cuda")
        th.cuda.set_device(0)  # Use first GPU by default
    else:
        device = th.device("cpu")

    # Initialize process group for single GPU using file-based initialization
    if not dist.is_initialized():
        # Create a temporary file for initialization
        init_file = os.path.join(os.getcwd(), "dist_init")
        with open(init_file, "w") as f:
            f.write("")

        dist.init_process_group(
            backend="gloo",  # Use gloo backend which is more widely available
            init_method=f"file://{init_file}",
            world_size=1,
            rank=0,
        )

        # Clean up the temporary file
        try:
            os.remove(init_file)
        except:
            pass

    return device


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Dummy function for compatibility.
    """
    return


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def get_world_size():
    """
    Get world size (always 1 in non-distributed mode).
    """
    return 1


def get_rank():
    """
    Get rank (always 0 in non-distributed mode).
    """
    return 0
