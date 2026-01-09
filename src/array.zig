const std = @import("std");

const dynamic = @import("array/dynamic.zig");
const static = @import("array/static.zig");

pub const DynamicArray = dynamic.DynamicArray;
pub const ConstDynamicArray = dynamic.ConstDynamicArray;
pub const StaticArray = static.StaticArray;
pub const ConstStaticArray = static.ConstStaticArray;

test {
    _ = dynamic;
    _ = static;
}
