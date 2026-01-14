const std = @import("std");

const view_mod = @import("./view.zig");
const slice_mod = @import("../slice.zig");
const elements_mod = @import("../elements.zig");
const array_mod = @import("../array.zig");
const pointer_mod = @import("../pointer.zig");

const ConstArrayView = view_mod.ConstArrayView;

const FormatConfig = struct {
    axis_collapse_limit: usize,
    axis_collapse_limit_next_last: usize,
    axis_collapse_limit_last: usize,

    const Self = @This();

    // NOTE: These constants are copied from Rust's ndarray. They can be adjusted later.

    /// Default threshold, below this element count, we don't ellipsize
    const ARRAY_MANY_ELEMENT_LIMIT: usize = 500;
    /// Limit of element count for non-last axes before overflowing with an ellipsis.
    const AXIS_LIMIT_STACKED: usize = 6;
    /// Limit for next to last axis (printed as column)
    /// An odd number because one element uses the same space as the ellipsis.
    const AXIS_LIMIT_COL: usize = 11;
    /// Limit for last axis (printed as row)
    /// An odd number because one element uses approximately the space of the ellipsis.
    const AXIS_LIMIT_ROW: usize = 11;

    /// Get the default format configuration
    pub fn default() FormatConfig {
        return FormatConfig{
            .axis_collapse_limit = AXIS_LIMIT_STACKED,
            .axis_collapse_limit_next_last = AXIS_LIMIT_COL,
            .axis_collapse_limit_last = AXIS_LIMIT_ROW,
        };
    }

    /// Get the collapse limit for the given axis index from the end.
    pub fn collapseLimit(self: Self, axis_rindex: usize) usize {
        switch (axis_rindex) {
            0 => return self.axis_collapse_limit_last,
            1 => return self.axis_collapse_limit_next_last,
            else => return self.axis_collapse_limit,
        }
    }
};

pub fn Formatter(comptime T: type) type {
    const element_type = elements_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for formatting: " ++ @typeName(T));

    return struct {
        /// The writer to output to
        writer: *std.io.Writer,
        /// The format configuration
        fmt_cfg: FormatConfig,
        /// The total rank of the array being printed
        total_array_rank: usize,
        /// The current depth in the recursive printing
        current_depth: usize,

        const Self = @This();

        const ELLIPSIS = "...";

        /// Entry point to print the given array view
        pub fn print(
            array: *const ConstArrayView(T),
            writer: *std.io.Writer,
        ) std.io.Writer.Error!void {
            const fmt_cfg = FormatConfig.default();
            var formatter = Self{
                .writer = writer,
                .fmt_cfg = fmt_cfg,
                .total_array_rank = array.dims.len,
                .current_depth = 0,
            };
            try formatter.printArrayInner(array);
        }

        /// Internal recursive function to print the array view
        fn printArrayInner(self: *Self, array_view: *const ConstArrayView(T)) std.io.Writer.Error!void {
            const current_rank = array_view.dims.len;

            // Check if array is empty
            if (std.mem.indexOfScalar(usize, array_view.dims, 0) != null) {
                // Just print [[...]]
                for (0..current_rank) |_| {
                    try self.writer.writeAll("[");
                }
                for (0..current_rank) |_| {
                    try self.writer.writeAll("]");
                }
                return;
            }

            switch (current_rank) {
                0 => {
                    // Scalar case
                    try self.printScalar(array_view.atUnchecked(&[_]usize{}).*);
                },
                1 => {
                    // One-dimensional case
                    try self.writer.writeAll("[");
                    try self.printWithOverflow(
                        array_view,
                        self.fmt_cfg.collapseLimit(0),
                        printSeparator,
                        printElemOneDim,
                    );
                    try self.writer.writeAll("]");
                },
                else => {
                    try self.writer.writeAll("[");
                    const limit = self.fmt_cfg.collapseLimit(self.total_array_rank - self.current_depth - 1);
                    self.current_depth += 1;
                    try self.printWithOverflow(
                        array_view,
                        limit,
                        printSeparator,
                        printSubArray,
                    );
                    self.current_depth -= 1;
                    try self.writer.writeAll("]");
                },
            }
        }

        /// Print a single scalar element
        fn printScalar(
            self: *Self,
            scalar: T,
        ) std.io.Writer.Error!void {
            switch (element_type) {
                .Bool => {
                    const bool_value = scalar != 0;
                    const val = if (bool_value) " True" else "False";
                    try self.writer.print("{s}", .{val});
                },
                .Int8, .Int16, .Int32, .Int64, .UInt8, .UInt16, .UInt32, .UInt64 => {
                    try self.writer.print("{d:>8}", .{scalar});
                },
                .Float32, .Float64, .Float128 => {
                    // Check if it's effectively an integer (within epsilon)
                    const is_int = @abs(scalar - @round(scalar)) < 1e-10;
                    if (is_int and @abs(scalar) < 1e10) {
                        try self.writer.print("{d:>8.1}", .{scalar});
                    } else {
                        try self.writer.print("{d:>13.8}", .{scalar});
                    }
                },
                .Complex64, .Complex128 => {
                    const sign: u8 = if (scalar.im >= 0) '+' else '-';
                    try self.writer.print("{d:.4}{c}{d:.4}j", .{ scalar.re, sign, @abs(scalar.im) });
                },
            }
        }

        /// Print separator between elements based on current rank
        fn printSeparator(
            self: *Self,
            current_rank: usize,
        ) std.io.Writer.Error!void {
            std.debug.assert(current_rank > 0);

            if (current_rank == 1) {
                try self.writer.writeAll(", ");
            } else {
                // Fill blank lines
                for (0..current_rank - 1) |_| {
                    try self.writer.writeAll("\n");
                }
                // Fill indentation spaces
                for (0..self.current_depth) |_| {
                    try self.writer.writeAll(" ");
                }
            }
        }

        /// Print a single element in a one-dimensional array
        fn printElemOneDim(
            self: *Self,
            array_view: *const ConstArrayView(T),
            elem_idx: usize,
        ) std.io.Writer.Error!void {
            std.debug.assert(array_view.dims.len == 1);
            std.debug.assert(elem_idx < array_view.dims[0]);
            try self.printScalar(array_view.atUnchecked(&[_]usize{elem_idx}).*);
        }

        fn printSubArray(
            self: *Self,
            array_view: *const ConstArrayView(T),
            elem_idx: usize,
        ) std.io.Writer.Error!void {
            std.debug.assert(array_view.dims.len >= 2);
            std.debug.assert(elem_idx < array_view.dims[0]);

            // Manually create a sub-array view to avoid allocations
            const new_dims = array_view.dims[1..];
            const new_strides = array_view.strides[1..];
            const new_data_ptr_single = pointer_mod.ptrFromOffset(
                T,
                array_view.data_ptr,
                // elem_idx is guaranteed to be less than dim size, which means less than isize max
                @as(isize, @intCast(elem_idx)) * array_view.strides[0],
            );
            const new_data_ptr: [*]const T = @ptrCast(new_data_ptr_single);

            try self.printArrayInner(&ConstArrayView(T){
                .data_ptr = new_data_ptr,
                .dims = new_dims,
                .strides = new_strides,
            });
        }

        /// Print elements with overflow handling
        /// array_view: the current array view to print
        /// limit: maximum number of elements to print before overflowing
        /// print_sep: function to print separator between elements
        /// print_elem: function to print a single element at given index
        fn printWithOverflow(
            self: *Self,
            array_view: *const ConstArrayView(T),
            limit: usize,
            print_sep: fn (self: *Self, current_rank: usize) std.io.Writer.Error!void,
            print_elem: fn (self: *Self, array_view: *const ConstArrayView(T), index: usize) std.io.Writer.Error!void,
        ) std.io.Writer.Error!void {
            const dim_size = array_view.dims[0];
            if (dim_size == 0) {
                return;
            }

            const rank = array_view.dims.len;

            if (dim_size <= limit) {
                // Print all elements
                try print_elem(self, array_view, 0);
                for (1..dim_size) |i| {
                    try print_sep(self, rank);
                    try print_elem(self, array_view, i);
                }
            } else {
                const head_count = limit / 2;
                const tail_count = limit - head_count;

                // Print head elements
                try print_elem(self, array_view, 0);
                for (1..head_count) |i| {
                    try print_sep(self, rank);
                    try print_elem(self, array_view, i);
                }

                // Print ellipsis
                try print_sep(self, rank);
                try self.writer.writeAll(ELLIPSIS);

                // Print tail elements
                for (tail_count..dim_size) |i| {
                    try print_sep(self, rank);
                    try print_elem(self, array_view, i);
                }
            }
        }
    };
}
