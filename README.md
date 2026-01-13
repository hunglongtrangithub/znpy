# Znpy

`znpy` provides [Npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) file reading implementation in Zig.

Note: Right now I have supported reading npy files for my other work, but writing npy files is the next priority.

## Why?

In my recent project that involves working with large vector datasets, our old codebase use C++ implementation for reading npy files called [cnpy](https://github.com/rogersce/cnpy). However, it does not support reading npy files with Fortran order arrays, slicing arrays, and I think the overall implementation is not very secure and robust. And since I decided to use Zig for the new codebase, I needed a Zig library that can read and write npy files efficiently. So I created `znpy` to fill this gap.

## Features

- Read and write Npy files
- Create memory-owned arrays
- Create array views from arrays
- Print arrays to stdout in structured array format

## Usage

Make sure [Zig](https://ziglang.org/download/) is installed.

Run the main executable for demo:

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
