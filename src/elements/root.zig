//! Module for array element functions and types.
const std = @import("std");
const header = @import("../header/root.zig");

const ViewDataError = error{
    EndiannessMismatch,
    MisingBytes,
    ExtraBytes,
    Misaligned,
};

const WriteDataError = error{};

const ReadDataError = error{};

pub fn Element(comptime T: header.ElementType) type {
    return struct {
        value: T,

        const Self = @This();

        pub fn bytesAsSlice(bytes: []const u8, len: usize, type_descr: header.TypeDescriptor) ViewDataError![]const Self {
            _ = bytes;
            _ = len;
            _ = type_descr;
        }

        pub fn writeSlice(slice: []const Self, writer: std.io.Writer) WriteDataError!void {
            _ = slice;
            _ = writer;
        }

        pub fn readSlice(slice: []Self, reader: *std.io.Reader, type_descr: header.TypeDescriptor) ReadDataError!void {
            _ = slice;
            _ = reader;
            _ = type_descr;
        }
    };
}
