//! `--` handling for `rtk gh`. clap carves the `subcommand` positional out of the same trailing
//! region as `args`, so restoring the stripped `--` over `args` alone shifts the region one token
//! and duplicates the subcommand. The one `--` clap strips is rtk's own option terminator —
//! `trailing_var_arg` keeps every later one — so it must not reach gh. Stubs `gh` on PATH with a
//! script that records its argv.

#![cfg(unix)]

use std::process::Command;

fn shell_quote(path: &std::path::Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "'\\''"))
}

/// Runs `rtk <args>` against a `gh` stub and returns the argv the stub received.
fn gh_argv(args: &[&str]) -> Vec<String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let argv_file = dir.path().join("argv.txt");
    let stub_path = dir.path().join("gh");

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

#[test]
fn gh_double_dash_before_subcommand_is_dropped() {
    // gh 2.46 and 2.100 both answer `gh -- pr view 42` with `unknown command "pr" for "gh"`,
    // and the subcommand must reach gh once, not twice.
    let argv = gh_argv(&["gh", "--", "pr", "view", "42"]);
    assert_eq!(argv[..3], ["pr", "view", "42"]);
    assert!(
        !argv.iter().any(|a| a == "--"),
        "rtk's own terminator must not reach gh: {argv:?}"
    );
}

#[test]
fn gh_double_dash_after_subcommand_is_dropped() {
    let argv = gh_argv(&["gh", "api", "--", "repos/o/r"]);
    assert_eq!(argv, vec!["api", "repos/o/r"]);
}

#[test]
fn gh_double_dash_after_subcommand_is_dropped_before_a_flag() {
    // gh reads everything past `--` as a positional, so this spelling is `accepts 1 arg(s),
    // received 3` on gh 2.46 and 2.100 while `gh api repos/o/r --jq .name` works.
    let argv = gh_argv(&["gh", "api", "--", "repos/o/r", "--jq", ".name"]);
    assert_eq!(argv, vec!["api", "repos/o/r", "--jq", ".name"]);
}

#[test]
fn gh_double_dash_inside_trailing_region_is_preserved() {
    let argv = gh_argv(&["gh", "api", "repos/o/r", "--", "--jq", ".name"]);
    assert_eq!(argv, vec!["api", "repos/o/r", "--", "--jq", ".name"]);
}

#[test]
fn gh_double_dash_inside_a_filtered_subcommand_is_preserved() {
    let argv = gh_argv(&["gh", "pr", "list", "--", "--state", "open"]);
    assert_eq!(
        argv[argv.len() - 3..],
        ["--", "--state", "open"],
        "an interior -- is the user's own and reaches gh: {argv:?}"
    );
}
