const std = @import("std");

const znpy = @import("znpy");

const dirname = std.fs.path.dirname;

fn readNpyArrayMmap(npy_file_path: []const u8, allocator: std.mem.Allocator) !void {
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

    // Use a const array since the mmap buffer cannot be mutated
    const ConstArray3D = znpy.array.ConstStaticArray(i16, 3);

    var array: ConstArray3D = ConstArray3D.fromFileBuffer(file_buffer, allocator) catch |e| {
        std.debug.print("Failed to create ArrayView from file buffer: {}\n", .{e});
        return;
    };

    std.debug.print("Array shape: {any}\n", .{array.shape.dims});
    std.debug.assert(array.shape.dims.len == 3);

    const array_view = try array.asView().slice(
        &[_]znpy.Slice{
            .{ .Index = 0 },
            .Ellipsis,
        },
        allocator,
    );
    std.debug.print("Sliced Array shape: {any}\n", .{array_view.dims});
    return;
}

pub fn main() !void {
    const source_dir = comptime dirname(@src().file) orelse "src";
    const npy_file_path = comptime source_dir ++ "/" ++ "../test-data/shapes/i16_3d_3x4x5.npy";
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();

    readNpyArrayMmap(npy_file_path, allocator) catch |e| {
        std.debug.print("Error reading header: {}\n", .{e});
    };
}
