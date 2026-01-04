//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const read = @import("read.zig");
pub const readNpyFile = read.readNpyFile;
