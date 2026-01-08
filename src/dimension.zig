const std = @import("std");

const header = @import("header.zig");

/// The number of bytes an array takes, given its shape and element type, must not exceed `std.math.maxInt(usize)`
/// Returns the number of elements in the array on success, or `null` if overflow occurs.
pub fn shapeSizeChecked(T: header.ElementType, shape: []usize) ?usize {
    const usize_max = std.math.maxInt(usize);

    const num_elements = blk: {
        var prod: usize = 1;
        for (shape) |dim| {
            prod, const overflow = @mulWithOverflow(prod, dim);
            if (overflow != 0) {
                return null;
            }
        }
        break :blk prod;
    };

    const num_bytes: usize, const overflow: u1 = @mulWithOverflow(num_elements, T.byteSize());
    if ((overflow != 0) or (num_bytes > usize_max)) {
        return null;
    }
    return num_elements;
}

const Shape = struct {
    dims: []const usize,
    strides: []const isize,

    // pub fn fromHeader(npy_header: header.Header) Shape {
    //     const strides = switch (npy_header.order) {
    //         .C => computeCStrides(npy_header.shape, npy_header.descr.byteSize()),
    //         .Fortran => computeFortranStrides(npy_header.shape, npy_header.descr.byteSize()),
    //     };
    // }
};
