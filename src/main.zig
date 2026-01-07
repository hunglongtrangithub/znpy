const std = @import("std");
const znpy = @import("znpy");

fn processNpyFile(file: std.fs.File) !void {
    var read_buffer: [1024]u8 = undefined;
    var file_reader = file.reader(read_buffer[0..]);

    // If header size is larger than 1024 bytes, use heap allocation
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();
    const header = try znpy.header.Header.fromReader(&file_reader.interface, allocator);
    defer header.deinit(allocator);

    std.debug.print("Numbfer of bytes read: {}\n", .{file_reader.pos});
    std.debug.print("Numpy Header: {any}\n", .{header});

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

    const file_buffer = try std.posix.mmap(
        null,
        read_size,
        std.posix.PROT.READ,
        std.posix.system.MAP{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(file_buffer);

    const data_buffer = file_buffer[file_reader.pos..];
    std.debug.print("Data buffer length: {}\n", .{data_buffer.len});

    // TODO: Validate shape to prevent overflow
    const total_elements = blk: {
        var prod: usize = 1;
        for (header.shape) |dim| {
            prod *= dim;
        }
        break :blk prod;
    };
    std.debug.print("Total elements: {}\n", .{total_elements});
    std.debug.print("Element type: {any}\n", .{header.descr.element_type});

    // TODO: Provide size for every element type
    std.debug.assert(header.descr.element_type == .Float64);
    std.debug.assert(data_buffer.len == total_elements * 8);
}

pub fn main() !void {
    const npy_file_path = "test.npy";
    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    processNpyFile(file) catch |e| {
        std.debug.print("Error reading header: {}\n", .{e});
    };
}
