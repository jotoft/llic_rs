Prepare and tag a release for the llic_rs crate. The user may specify a version
number as $ARGUMENTS, or you should determine the next version based on semver
and the changes since the last release.

Follow these steps:

1. Determine the version to release:
   - If the user provided a version in $ARGUMENTS, use that.
   - Otherwise, inspect `git log` since the last tag (or last version bump commit)
     and decide: patch (bug fixes, perf), minor (new features, backward-compatible
     API additions), or major (breaking API changes).
   - The project follows semver strictly.

2. Pre-flight checks:
   - Run `cargo test --release` and ensure all tests pass.
   - Run `RUSTFLAGS="-C target-cpu=native" cargo clippy --release` and ensure
     there are no warnings.
   - Run `cargo fmt -- --check` and ensure formatting is clean.
   - If any check fails, fix the issues before proceeding.

3. Update CHANGELOG.md:
   - Move the [Unreleased] section contents into a new version heading with
     today's date.
   - Categorize changes under Added/Changed/Deprecated/Removed/Fixed/Security
     per Keep a Changelog conventions.
   - Add a fresh empty [Unreleased] section at the top.

4. Update version in Cargo.toml to the new version.

5. Create a commit with message: "Release v{version}"
   - Stage only Cargo.toml, Cargo.lock (if it changed), and CHANGELOG.md.

6. Create a git tag: `v{version}`

7. Print a summary of what was done and remind the user to push with:
   ```
   git push && git push --tags
   ```
