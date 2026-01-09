const std = @import("std");

const znpy = @import("znpy");

const dirname = std.fs.path.dirname;

fn processNpyFile(file: std.fs.File) !void {
    // Get file size
    const file_stat = try file.stat();
    const read_size = std.math.cast(usize, file_stat.size) orelse {
        std.debug.print("File size is too large to map\n", .{});
        return;
    };
    if (read_size == 0) {
        std.debug.print("File is empty, nothing to map\n", .{});
        return;
    }
    std.debug.print("Mapping file of size: {}\n", .{read_size});

    // Read all file contents into memory using mmap
    const file_buffer = try std.posix.mmap(
        null,
        read_size,
        std.posix.PROT.READ,
        std.posix.system.MAP{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(file_buffer);

    // If header size is larger than 1024 bytes, use heap allocation
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();

    const ArrayView = znpy.ArrayView(i16, null);

    var array_view: ArrayView = ArrayView.fromFileBuffer(file_buffer, allocator) catch |e| {
        std.debug.print("Failed to create ArrayView from file buffer: {}\n", .{e});
        return;
    };
    defer array_view.deinit(allocator);

    std.debug.assert(array_view.dims.len == 3);

    for (0..array_view.dims[0]) |i| {
        for (0..array_view.dims[1]) |j| {
            for (0..array_view.dims[2]) |k| {
                const value = array_view.at(&[3]usize{ i, j, k }).?.*;
                std.debug.print("Element at ({}, {}, {}) = {}\n", .{ i, j, k, value });
            }
        }
    }
}

pub fn main() !void {
    const source_dir = comptime dirname(@src().file) orelse "src";
    const npy_file_path = comptime source_dir ++ "/" ++ "../test-data/shapes/i16_3d_3x4x5.npy";
    // const npy_file_path = "test.npy";
    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    processNpyFile(file) catch |e| {
        std.debug.print("Error reading header: {}\n", .{e});
    };
}
