const std = @import("std");
const log = std.log.scoped(.npy_reader);
const header = @import("header/root.zig");
const readNpyHeaderData = header.readNpyHeaderData;
const NpyHeaderData = header.NpyHeaderData;
const Parser = @import("header/parse.zig").Parser;

pub const HeaderEncoding = enum {
    Ascii,
    Utf8,
};

const HeaderSizeType = enum {
    U16,
    U32,
};

pub const VersionProps = struct {
    header_size_type: HeaderSizeType,
    encoding: HeaderEncoding,
};

const NpyFileReadError = error{
    /// The file is not a valid .npy file.
    InvalidFile,
    /// The format version is unsupported.
    UnsupportedVersion,
    /// Fortran order is not supported.
    FortranOrderNotSupported,
    /// Error reading from file, due to I/O issues.
    IoError,
};

const NpyHeaderParseError = error{
    EndingNewlineMissing,
    InvalidHeaderFormat,
};

const MAGIC = "\x93NUMPY";

/// Processes the .npy header content into a structured format.
/// Parameters:
/// - header_buffer: The buffer containing the header content.
/// - header_encoding: The encoding type of the header.
/// Returns:
/// - Parsed Python value on success, or an error on failure.
fn processNpyHeader(header_buffer: []const u8, header_encoding: HeaderEncoding, allocator: std.mem.Allocator) NpyHeaderParseError!NpyHeaderData {
    // Check for ending newline
    if (header_buffer.len == 0 or header_buffer[header_buffer.len - 1] != '\n') {
        return NpyHeaderParseError.EndingNewlineMissing;
    }
    // Trim newline and spaces right before it
    const trimmed_header = std.mem.trimRight(u8, header_buffer[0 .. header_buffer.len - 1], " ");
    log.debug("Trimmed Header: |{s}|", .{trimmed_header});

    const header_data = readNpyHeaderData(trimmed_header, header_encoding, allocator) catch |e| {
        log.err("Error reading header data: {any}", .{e});
        return NpyHeaderParseError.InvalidHeaderFormat;
    };
    log.info("Parsed Header Data: descr={}, fortran_order={}, shape={any}", .{ header_data.descr, header_data.fortran_order, header_data.shape });
    return header_data;
}

/// Reads and validates a .npy file header.
/// Paramters:
/// - file_reader: A file reader for the .npy file.
/// Returns:
/// - NpyFileReadError on failure.
pub fn readNpyFile(file_reader: *std.fs.File.Reader) (NpyFileReadError || std.mem.Allocator.Error)!void {
    var byte_buffer: [8]u8 = undefined;
    file_reader.interface.readSliceAll(byte_buffer[0..]) catch |e| {
        switch (e) {
            error.ReadFailed => return NpyFileReadError.IoError,
            error.EndOfStream => return NpyFileReadError.InvalidFile,
        }
    };

    if (std.mem.eql(u8, byte_buffer[0..6], MAGIC)) {
        log.info("Valid .npy file detected.", .{});
    } else {
        log.err("Invalid .npy file: Magic number mismatch.", .{});
        return NpyFileReadError.InvalidFile;
    }

    const major_version = byte_buffer[6];
    const minor_version = byte_buffer[7];

    const version_props: VersionProps = version: {
        if (minor_version != 0) {
            log.err("Unsupported .npy version: {}.{}", .{ major_version, minor_version });
            return NpyFileReadError.UnsupportedVersion;
        }
        switch (major_version) {
            1 => break :version .{ .header_size_type = .U16, .encoding = .Ascii },
            2 => break :version .{ .header_size_type = .U32, .encoding = .Ascii },
            3 => break :version .{ .header_size_type = .U32, .encoding = .Utf8 },
            else => {
                log.err("Unsupported .npy major version: {}", .{major_version});
                return NpyFileReadError.UnsupportedVersion;
            },
        }
    };

    log.info("Version: {}.{}", .{ major_version, minor_version });
    log.debug("Version Props: {any}", .{version_props});

    // Read the header size in little-endian format
    const header_size: u32 = header_size: switch (version_props.header_size_type) {
        .U16 => {
            var size_buffer: [2]u8 = undefined;
            file_reader.interface.readSliceAll(size_buffer[0..]) catch |e| {
                switch (e) {
                    error.ReadFailed => return NpyFileReadError.IoError,
                    error.EndOfStream => return NpyFileReadError.InvalidFile,
                }
            };
            break :header_size std.mem.readInt(u16, &size_buffer, .little);
        },
        .U32 => {
            var size_buffer: [4]u8 = undefined;
            file_reader.interface.readSliceAll(size_buffer[0..]) catch |e| {
                switch (e) {
                    error.ReadFailed => return NpyFileReadError.IoError,
                    error.EndOfStream => return NpyFileReadError.InvalidFile,
                }
            };
            break :header_size std.mem.readInt(u32, &size_buffer, .little);
        },
    };

    log.debug("Header Size: 0x{x:0>2}", .{header_size});

    for (std.mem.asBytes(&header_size), 0..) |byte, i| {
        log.debug("Header Byte [{}]: 0x{x:0>2}", .{ i, byte });
    }

    // Now read the header content
    // If header size is larger than 1024 bytes, use heap allocation
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();
    const header_buffer = try allocator.alloc(u8, header_size);
    defer allocator.free(header_buffer);

    file_reader.interface.readSliceAll(header_buffer) catch |e| {
        switch (e) {
            error.ReadFailed => return NpyFileReadError.IoError,
            error.EndOfStream => return NpyFileReadError.InvalidFile,
        }
    };

    log.debug("Header Content: {s}", .{header_buffer});

    const header_data = processNpyHeader(header_buffer, version_props.encoding, allocator) catch |e| {
        log.err("Error parsing header: {any}", .{e});
        return NpyFileReadError.InvalidFile;
    };

    if (header_data.fortran_order) {
        log.err("Fortran order arrays are not supported.", .{});
        return NpyFileReadError.FortranOrderNotSupported;
    }
}
