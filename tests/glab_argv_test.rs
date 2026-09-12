//! Child-argv tests for `rtk glab`: which argument becomes the MR/issue identifier, and where
//! the user's `--` and rtk's own `-R`/`-g` end up. Stubs `glab` on PATH with a script that
//! records its argv, so every assertion is on what glab actually received.

#![cfg(unix)]

use std::process::Command;

fn shell_quote(path: &std::path::Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "'\\''"))
}

/// Runs `rtk <args>` against a `glab` stub and returns the argv the stub received.
fn glab_argv(args: &[&str]) -> Vec<String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let argv_file = dir.path().join("argv.txt");
    let stub_path = dir.path().join("glab");

    std::fs::write(
        &stub_path,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\nexit 0\n",
            shell_quote(&argv_file)
        ),
    )
    .expect("write stub");
    let mut perms = std::fs::metadata(&stub_path)
        .expect("stat stub")
        .permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&stub_path, perms).expect("chmod stub");

    let path_with_stub = format!(
        "{}:{}",
        dir.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let out = Command::new(env!("CARGO_BIN_EXE_rtk"))
        .env("PATH", path_with_stub)
        .env("LC_ALL", "C")
        .current_dir(dir.path())
        .args(args)
        .output()
        .expect("spawn rtk");

    std::fs::read_to_string(&argv_file)
        .unwrap_or_else(|e| {
            panic!(
                "read captured argv: {e}; rtk stdout={} stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            )
        })
        .lines()
        .map(str::to_string)
        .collect()
}

// ── identifier extraction: a flag's value is never the MR/issue id ──────

#[test]
fn mr_view_page_value_is_not_the_mr_id() {
    // `-p/--page` takes a value on `glab mr view`; the walker's list omitted it, so "2" was
    // hoisted out of the flag and re-emitted as the MR identifier.
    let argv = glab_argv(&["glab", "mr", "view", "--page", "2"]);
    assert_eq!(argv, vec!["mr", "view", "-F", "json", "--page", "2"]);
}

#[test]
fn mr_view_jq_expression_is_not_the_mr_id() {
    let argv = glab_argv(&["glab", "mr", "view", "--jq", ".iid"]);
    assert_eq!(argv, vec!["mr", "view", "-F", "json", "--jq", ".iid"]);
}

#[test]
fn issue_view_per_page_value_is_not_the_issue_id() {
    let argv = glab_argv(&["glab", "issue", "view", "-P", "50"]);
    assert_eq!(argv, vec!["issue", "view", "-F", "json", "-P", "50"]);
}

#[test]
fn mr_view_keeps_an_explicit_id_ahead_of_a_valued_flag() {
    let argv = glab_argv(&["glab", "mr", "view", "--page", "2", "42"]);
    assert_eq!(argv, vec!["mr", "view", "42", "-F", "json", "--page", "2"]);
}

// ── `--` handling ──────────────────────────────────────────────────────

#[test]
fn glab_double_dash_before_subcommand_is_forwarded_verbatim() {
    // clap eats a `--` sitting at the head of the trailing region. Restoring it over `args`
    // alone lands one token off and duplicates the subcommand (`glab mr mr view 42`).
    let argv = glab_argv(&["glab", "--", "mr", "view", "42"]);
    assert_eq!(argv, vec!["--", "mr", "view", "42"]);
}

#[test]
fn glab_double_dash_inside_trailing_region_is_preserved() {
    let argv = glab_argv(&["glab", "api", "--", "projects/1", "--paginate"]);
    assert_eq!(argv, vec!["api", "--", "projects/1", "--paginate"]);
}

#[test]
fn glab_repo_flag_is_injected_before_the_boundary() {
    // glab reads everything past `--` as a positional, so an appended `-R` would arrive as two
    // extra arguments rather than as the repo flag.
    let argv = glab_argv(&["glab", "-R", "o/r", "mr", "diff", "--", "42"]);
    assert_eq!(argv, vec!["mr", "diff", "-R", "o/r", "--", "42"]);
}
