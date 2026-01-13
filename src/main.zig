const std = @import("std");

const znpy = @import("znpy");

const dirname = std.fs.path.dirname;
var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

pub fn main() !void {
    const source_dir = comptime dirname(@src().file) orelse "src";
    const npy_file_path = comptime source_dir ++ "/" ++ "../test-data/shapes/f32_2d_4x5.npy";
    try stdout.print("Loading NPY file from path: {s}\n", .{npy_file_path});

    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();

    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    // Get file size
    const file_stat = try file.stat();
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
        std.posix.PROT.READ,
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

    try stdout.flush();

    // Use a const array since the mmap buffer cannot be mutated
    const ConstArray = znpy.array.ConstStaticArray(f32, 2);

    const array: ConstArray = ConstArray.fromFileBuffer(file_buffer, allocator) catch |e| {
        std.debug.print("Failed to create ArrayView from file buffer: {}\n", .{e});
        return;
    };

    try stdout.print("Formatted array:\n{f}\n", .{array});
    try stdout.print("Array shape: {any}\n", .{array.shape.dims});
    try stdout.print("Number of elements: {any}\n", .{array.data_buffer.len});
    try stdout.print("Array's start address: {any}\n", .{array.data_buffer.ptr});
    try stdout.print("Array's memory order: {any}\n", .{array.shape.order});
    try stdout.flush();

    try stdout.print("Getting slice array[:, 0]:\n", .{});
    const array_view = try array.slice(
        &znpy.s(.{ .{}, 1 }),
        allocator,
    );
    defer array_view.deinit(allocator);

    try stdout.print("Sliced array view:\n{f}\n", .{array_view});
    try stdout.print("Sliced array view shape: {any}\n", .{array_view.dims});
    try stdout.print("Sliced array view strides: {any}\n", .{array_view.strides});
    try stdout.print("Sliced array view data ptr: {any}\n", .{array_view.data_ptr});
    std.debug.assert(array_view.dims.len == 1 and array_view.dims[0] == 4);

    // Get array view in slice
    const slice: []const f32 = array_view.data_ptr[0..array_view.dims[0]];
    try stdout.print("Array view in Zig slice: {any}\n", .{slice});
    try stdout.flush();
    return;
}
