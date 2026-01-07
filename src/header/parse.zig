const std = @import("std");
const lex = @import("lex.zig");
const read = @import("../read.zig");
const log = std.log.scoped(.npy_parser);
const Token = lex.Token;

/// The Abstract Syntax Tree for the parsed header content.
/// Right now we only support maps, tuples, and literals.
const Ast = union(enum) {
    Map: std.StringHashMapUnmanaged(Ast),
    Literal: lex.Literal,
    Tuple: std.ArrayList(usize),

    const Self = @This();

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        switch (self) {
            .Map => |map| {
                var it = map.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit(allocator);
                }
                var mut_map = map;
                mut_map.deinit(allocator);
            },
            .Tuple => |tuple| {
                var mut_tuple = tuple;
                mut_tuple.deinit(allocator);
            },
            .Literal => {},
        }
    }
};

const ParserError = error{
    EmptyInput,
    LiteralFollowedByMoreTokens,
    MisplacedToken,
    InvalidKey,
    MissingColonAfterKey,
    MissingCommaAfterValue,
    InvalidValue,
    InvalidTupleElement,
    MissingTrailingComma,
    InvalidSyntax,
};

pub const BuildError = lex.LexerError || ParserError || std.mem.Allocator.Error;

pub const Parser = struct {
    const Self = @This();

    lexer: lex.NpyHeaderLexer,

    /// Initializes the parser with the given header buffer and encoding.
    /// Input buffer must outlive the parser.
    /// Parameters:
    /// header_buffer - The buffer containing the header content.
    /// header_encoding - The encoding type of the header.
    pub fn init(header_buffer: []const u8, header_encoding: read.HeaderEncoding) Self {
        return .{
            .lexer = lex.NpyHeaderLexer.init(header_buffer, header_encoding),
        };
    }

    /// Parse a map. The opening brace has already been consumed.
    /// The map keys are strings, and values can be literals, tuples, or nested maps.
    fn parseMap(self: *Self, allocator: std.mem.Allocator) BuildError!std.StringHashMapUnmanaged(Ast) {
        var map = std.StringHashMapUnmanaged(Ast).empty;
        errdefer map.deinit(allocator);

        const State = union(enum) {
            /// Beginning state (at the opening brace)
            Start,
            /// At a key
            Key: []const u8,
            /// At a value
            Value,
            /// At the closing brace
            Final,
        };

        // Initialize state machine
        var state: State = .Start;
        while (state != .Final) {
            const token = try self.lexer.advance();
            switch (state) {
                .Start => switch (token) {
                    // Empty map
                    .RBrace => state = .Final,
                    .Literal => |k|
                    // Expect string literal as key
                    switch (k) {
                        .String => |s| {
                            state = .{ .Key = s };
                        },
                        else => return ParserError.InvalidKey,
                    },
                    else => return ParserError.InvalidKey,
                },
                .Key => {
                    // Expect colon after key
                    if (token != .Colon) {
                        return ParserError.MissingColonAfterKey;
                    }
                    // Expect value after colon
                    switch (try self.lexer.advance()) {
                        // Literal value
                        .Literal => |literal| {
                            try map.put(allocator, state.Key, .{ .Literal = literal });
                        },
                        .LParen => {
                            // Parse tuple value
                            const tuple = try self.parseTuple(allocator);
                            try map.put(allocator, state.Key, .{ .Tuple = tuple });
                        },
                        .LBrace => {
                            // Parse nested map value
                            const nested_map = try self.parseMap(allocator);
                            try map.put(allocator, state.Key, .{ .Map = nested_map });
                        },
                        else => return ParserError.InvalidValue,
                    }
                    state = .Value;
                },
                .Value => switch (token) {
                    // Comma can only be followed by another key or closing brace
                    .Comma => switch (try self.lexer.advance()) {
                        // Expect another key
                        .Literal => |k| {
                            // Expect string literal as key
                            switch (k) {
                                .String => |s| {
                                    state = .{ .Key = s };
                                },
                                else => return ParserError.InvalidKey,
                            }
                        },
                        .RBrace => state = .Final,
                        else => return ParserError.InvalidKey,
                    },
                    .RBrace => state = .Final,
                    else => return ParserError.MissingCommaAfterValue,
                },
                .Final => unreachable,
            }
        }
        return map;
    }

    /// Parse a tuple. The opening parenthesis has already been consumed.
    /// The tuple can only contain number literals.
    fn parseTuple(self: *Self, allocator: std.mem.Allocator) BuildError!std.ArrayList(usize) {
        var list = std.ArrayList(usize).empty;
        errdefer list.deinit(allocator);

        const State = enum {
            /// Beginning state (at the opening parenthesis)
            Start,
            /// At a comma
            Comma,
            /// At a number literal
            Literal,
            /// At the closing parenthesis
            Final,
        };

        // Initialize state machine
        var state: State = .Start;
        while (state != .Final) {
            const token = try self.lexer.advance();
            switch (state) {
                .Start => {
                    switch (token) {
                        // Empty tuple
                        .RParen => state = .Final,
                        .Literal => |literal| {
                            switch (literal) {
                                .Number => |n| {
                                    try list.append(allocator, n);
                                    state = .Literal;
                                },
                                else => return ParserError.InvalidTupleElement,
                            }
                        },
                        else => return ParserError.InvalidTupleElement,
                    }
                },
                .Literal => {
                    switch (token) {
                        .Comma => state = .Comma,
                        .RParen => switch (list.items.len) {
                            // Single-element tuple must have a trailing comma before closing parenthesis
                            1 => return ParserError.MissingTrailingComma,
                            // Multi-element tuple can close directly
                            else => state = .Final,
                        },
                        else => return ParserError.InvalidSyntax,
                    }
                },
                .Comma => {
                    switch (token) {
                        .Literal => |literal| {
                            switch (literal) {
                                .Number => |n| {
                                    try list.append(allocator, n);
                                    state = .Literal;
                                },
                                else => return ParserError.InvalidTupleElement,
                            }
                        },
                        .RParen => state = .Final,
                        else => return ParserError.InvalidSyntax,
                    }
                },
                .Final => unreachable,
            }
        }

        return list;
    }

    /// Parse the header content into an AST.
    pub fn parse(self: *Self, allocator: std.mem.Allocator) BuildError!Ast {
        const token = try self.lexer.advance();
        switch (token) {
            .EOF => return ParserError.EmptyInput,
            .LBrace => {
                return .{ .Map = try self.parseMap(allocator) };
            },
            .LParen => {
                return .{ .Tuple = try self.parseTuple(allocator) };
            },
            .RBrace, .RParen, .Colon, .Comma => return ParserError.MisplacedToken,
            .Literal => |literal| {
                // No more tokens should follow a literal
                if (try self.lexer.peek() != .EOF) {
                    return ParserError.LiteralFollowedByMoreTokens;
                }
                return .{ .Literal = literal };
            },
        }
    }
};

test "parse empty map" {
    const input = "{}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    try std.testing.expectEqual(@as(usize, 0), ast.Map.count());
}

test "parse map with string literal" {
    const input = "{'descr': '<f8'}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const descr = ast.Map.get("descr").?;
    try std.testing.expectEqual(Ast.Literal, @as(std.meta.Tag(Ast), descr));
    try std.testing.expectEqualStrings("<f8", descr.Literal.String);
}

test "parse map with boolean literal" {
    const input = "{'fortran_order': False}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const fortran = ast.Map.get("fortran_order").?;
    try std.testing.expectEqual(Ast.Literal, @as(std.meta.Tag(Ast), fortran));
    try std.testing.expectEqual(false, fortran.Literal.Boolean);
}

test "parse map with empty tuple" {
    const input = "{'shape': ()}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const shape = ast.Map.get("shape").?;
    try std.testing.expectEqual(Ast.Tuple, @as(std.meta.Tag(Ast), shape));
    try std.testing.expectEqual(@as(usize, 0), shape.Tuple.items.len);
}

test "parse map with single-element tuple" {
    const input = "{'shape': (5,)}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const shape = ast.Map.get("shape").?;
    try std.testing.expectEqual(Ast.Tuple, @as(std.meta.Tag(Ast), shape));
    try std.testing.expectEqual(@as(usize, 1), shape.Tuple.items.len);
    try std.testing.expectEqual(@as(usize, 5), shape.Tuple.items[0]);
}

test "parse map with multi-element tuple" {
    const input = "{'shape': (3, 4, 5)}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const shape = ast.Map.get("shape").?;
    try std.testing.expectEqual(Ast.Tuple, @as(std.meta.Tag(Ast), shape));
    try std.testing.expectEqual(@as(usize, 3), shape.Tuple.items.len);
    try std.testing.expectEqual(@as(usize, 3), shape.Tuple.items[0]);
    try std.testing.expectEqual(@as(usize, 4), shape.Tuple.items[1]);
    try std.testing.expectEqual(@as(usize, 5), shape.Tuple.items[2]);
}

test "parse map with trailing comma" {
    const input = "{'descr': '<f8', 'fortran_order': False,}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    try std.testing.expectEqual(@as(usize, 2), ast.Map.count());
}

test "parse complete npy header" {
    const input = "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4)}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    try std.testing.expectEqual(@as(usize, 3), ast.Map.count());
}

test "parse nested map" {
    const input = "{'outer': {'inner': 'value'}}";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), ast));
    const outer = ast.Map.get("outer").?;
    try std.testing.expectEqual(Ast.Map, @as(std.meta.Tag(Ast), outer));
    const inner = outer.Map.get("inner").?;
    try std.testing.expectEqualStrings("value", inner.Literal.String);
}

test "parse top-level tuple" {
    const input = "(1, 2, 3)";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Tuple, @as(std.meta.Tag(Ast), ast));
    try std.testing.expectEqual(@as(usize, 3), ast.Tuple.items.len);
}

test "parse top-level literal" {
    const input = "'standalone'";
    var parser = Parser.init(input, .Ascii);
    const ast = try parser.parse(std.testing.allocator);
    defer ast.deinit(std.testing.allocator);

    try std.testing.expectEqual(Ast.Literal, @as(std.meta.Tag(Ast), ast));
    try std.testing.expectEqualStrings("standalone", ast.Literal.String);
}

test "error on empty input" {
    const input = "";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.EmptyInput, parser.parse(std.testing.allocator));
}

test "error on literal followed by more tokens" {
    const input = "'value' 123";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.LiteralFollowedByMoreTokens, parser.parse(std.testing.allocator));
}

test "error on misplaced closing brace" {
    const input = "}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MisplacedToken, parser.parse(std.testing.allocator));
}

test "error on misplaced closing paren" {
    const input = ")";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MisplacedToken, parser.parse(std.testing.allocator));
}

test "error on misplaced colon" {
    const input = ":";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MisplacedToken, parser.parse(std.testing.allocator));
}

test "error on misplaced comma" {
    const input = ",";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MisplacedToken, parser.parse(std.testing.allocator));
}

test "error on invalid key (number)" {
    const input = "{123: 'value'}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidKey, parser.parse(std.testing.allocator));
}

test "error on invalid key (boolean)" {
    const input = "{True: 'value'}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidKey, parser.parse(std.testing.allocator));
}

test "error on missing colon after key" {
    const input = "{'key' 'value'}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MissingColonAfterKey, parser.parse(std.testing.allocator));
}

test "error on missing comma after value" {
    const input = "{'key1': 'value1' 'key2': 'value2'}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MissingCommaAfterValue, parser.parse(std.testing.allocator));
}

test "error on invalid value (closing brace)" {
    const input = "{'key': }";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidValue, parser.parse(std.testing.allocator));
}

test "error on invalid tuple element (string)" {
    const input = "('string',)";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidTupleElement, parser.parse(std.testing.allocator));
}

test "error on invalid tuple element (boolean)" {
    const input = "(True,)";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidTupleElement, parser.parse(std.testing.allocator));
}

test "error on missing trailing comma in single-element tuple" {
    const input = "(5)";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.MissingTrailingComma, parser.parse(std.testing.allocator));
}

test "error on invalid syntax in tuple (double comma)" {
    const input = "(1,, 2)";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidSyntax, parser.parse(std.testing.allocator));
}

test "error on invalid key after comma" {
    const input = "{'key1': 'value1', 123}";
    var parser = Parser.init(input, .Ascii);
    try std.testing.expectError(ParserError.InvalidKey, parser.parse(std.testing.allocator));
}
