const std = @import("std");
const znpy = @import("znpy");

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

    var slice_reader = znpy.header.SliceReader.init(file_buffer);

    // If header size is larger than 1024 bytes, use heap allocation
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();
    const header = try znpy.header.Header.fromSliceReader(&slice_reader, allocator);
    defer header.deinit(allocator);

    std.debug.print("Numbfer of bytes read: {}\n", .{slice_reader.pos});
    std.debug.print("Numpy Header: {any}\n", .{header});

    const data_buffer = file_buffer[slice_reader.pos..];
    std.debug.print("Data buffer length: {}\n", .{data_buffer.len});

    const total_elementss = znpy.shapeSizeChecked(header.descr, header.shape) orelse {
        std.debug.print("Array size overflowed\n", .{});
        return;
    };
    std.debug.print("Element's type descriptor: {any}\n", .{header.descr});
    std.debug.print("Total number of elements: {}\n", .{total_elementss});

    std.debug.assert(header.descr == .Float64);
    std.debug.assert(data_buffer.len == total_elementss * header.descr.byteSize());

    const float64_slice = znpy.Element(f64).bytesAsSlice(
        data_buffer,
        total_elementss,
        header.descr,
    ) catch |e| {
        std.debug.print("Error interpreting data buffer as f64 slice: {}\n", .{e});
        return;
    };

    for (float64_slice, 0..) |value, index| {
        std.debug.print("Element [{}]: {}\n", .{ index, value });
    }
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
