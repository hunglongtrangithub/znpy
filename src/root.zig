//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
pub const header = @import("header/root.zig");

test {
    // Make all tests in other files imported by this module available during testing
    std.testing.refAllDecls(@This());
}
