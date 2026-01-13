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
        try stdout.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    // Get file size
    const file_stat = try file.stat();
    const read_size = std.math.cast(usize, file_stat.size) orelse {
        try stdout.print("File size is too large to map\n", .{});
        return;
    };
    if (read_size == 0) {
        try stdout.print("File is empty, nothing to read\n", .{});
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
    try stdout.flush();

    // Use a const array since the mmap buffer cannot be mutated
    const ConstArray = znpy.array.ConstDynamicArray(f32);

    const array: ConstArray = ConstArray.fromFileBuffer(file_buffer, allocator) catch |e| {
        try stdout.print("Failed to create ArrayView from file buffer: {}\n", .{e});
        try stdout.flush();
        return;
    };

    try stdout.print("Array shape: {any}\n", .{array.shape.dims});
    try stdout.print("Number of elements: {any}\n", .{array.data_buffer.len});
    try stdout.print("Array's start address: {any}\n", .{array.data_buffer.ptr});
    try stdout.flush();

    // try stdout.print("Getting slice array_view[:]\n", .{});
    // const array_view = try array.slice(
    //     &znpy.s(.{ .{}, .{}, .{} }),
    //     allocator,
    // );
    // defer array_view.deinit(allocator);

    // try stdout.print("Sliced Array shape: {any}\n", .{array_view.dims});
    // std.debug.assert(array_view.dims.len == 1 and array_view.dims[0] == 5);
    //
    // // Get array view in slice
    // const slice: []const f32 = array_view.data_ptr[0..array_view.dims[0]];
    // try stdout.print("Sliced data: {any}\n", .{slice});

    try stdout.print("Formatted Array:\n{f}\n", .{array});
    try stdout.flush();
    return;
}
