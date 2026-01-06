//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const read = @import("read.zig");
pub const readNpyFile = read.readNpyFile;

test {
    // Make all tests in other files imported by this module available during testing
    std.testing.refAllDecls(@This());
}
