# Znpy

`znpy` provides [Npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) file reader and writer in Zig.

Note: Right now I have supported reading npy files for my other work, but writing npy files is the next priority.

## Features

- Read and write Npy files
- Create memory-owned arrays
- Create array views from arrays
- Print arrays to stdout in structured array format

## Usage

Make sure [Zig](https://ziglang.org/download/) is installed.

Run the main executable:

```sh
zig build run
```

Run tests:

```sh
zig build test
```

To generate test npy files, you can use the provided Python script. Make sure to have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Build the virtual environment and install dependencies:

```sh
uv sync
```

Run the script to generate npy files:

```sh
uv run main.py
```
