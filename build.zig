const std = @import("std");

const examples: [4][]const u8 = [_][]const u8{
    "endian",
    "file_io",
    "from_mmap",
    "from_reader",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("znpy", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    inline for (examples) |example_name| {
        const demo_exe = b.addExecutable(.{
            .name = example_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path("examples/" ++ example_name ++ ".zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "znpy", .module = mod },
                },
            }),
        });

        const run_cmd = b.addRunArtifact(demo_exe);
        const run_step = b.step(example_name, "Run the " ++ example_name ++ " example");
        run_step.dependOn(&run_cmd.step);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}
