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
    try stdout.flush();

    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();

    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    const Array = znpy.array.StaticArray(f32, 2);

    var file_buffer: [1024]u8 = undefined;
    var file_reader = std.fs.File.Reader.init(file, &file_buffer);
    const array = try Array.fromFileAlloc(&file_reader.interface, allocator);
    defer array.deinit(allocator);

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
