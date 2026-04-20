const std = @import("std");
const builtin = @import("builtin");

const znpy = @import("znpy");

pub fn main(init: std.process.Init) !void {
    if (builtin.os.tag != .linux) {
        std.debug.print("Memory mapping is for Linux only.\n", .{});
        return;
    }

    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const npy_file_path = "./test-data/shapes/f64_1d_large.npy";
    try stdout.print("Loading NPY file from path: {s}\n", .{npy_file_path});
    try stdout.flush();

    var fallback = std.heap.stackFallback(1024, gpa);
    const allocator = fallback.get();

    const file = std.Io.Dir.cwd().openFile(io, npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close(io);

    // Get file size
    const file_stat = try file.stat(io);
    const read_size = std.math.cast(usize, file_stat.size) orelse {
        std.debug.print("File size is too large to map\n", .{});
        return;
    };
    if (read_size == 0) {
        std.debug.print("File is empty, nothing to read\n", .{});
        return;
    }

    // Read all file contents into memory using mmap
    const file_buffer = try std.posix.mmap(
        null,
        read_size,
        std.posix.PROT{ .READ = true },
        std.posix.system.MAP{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(file_buffer);
    try stdout.print("Mapped file size: {}\n", .{read_size});
    if (file_buffer.len != read_size) {
        std.debug.print("Mapped size does not match file size.\n", .{});
        return;
    }

    // Use a const array since the mmap buffer cannot be mutated
    const ConstArray = znpy.array.ConstStaticArray(f64, 1);

    const array: ConstArray = ConstArray.fromFileBuffer(file_buffer, allocator) catch |e| {
        std.debug.print("Failed to create ArrayView from file buffer: {}\n", .{e});
        return;
    };

    try stdout.print("Array shape: {any}\n", .{array.shape.dims});
    try stdout.print("Number of elements: {any}\n", .{array.data_buffer.len});
    try stdout.print("Array's start address: {any}\n", .{array.data_buffer.ptr});
    try stdout.print("Array's memory order: {any}\n", .{array.shape.order});

    try stdout.print("First 5 elements of the array:\n", .{});
    const array_view = try array.slice(
        &znpy.s(.{.{ null, 5 }}),
        allocator,
    );
    defer array_view.deinit(allocator);

    try stdout.print("{f}\n", .{array_view});

    try stdout.flush();
    return;
}
