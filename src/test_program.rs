//! Stand-in programs the tests write and run.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// `script` (a shell script, its `#!` line first) as the executable at
/// `path`, ready to run. A file just written cannot run ("Text file busy")
/// while a process another test forked in the meantime still holds it open
/// for writing; it can once those have gone on, and no later fork holds it.
/// So the script answers `--probe` at once, doing nothing, and is run so
/// until it starts.
pub fn write(path: &Path, script: &str) {
    let (shebang, body) = script.split_once('\n').unwrap_or((script, ""));
    std::fs::write(
        path,
        format!("{shebang}\n[ \"$1\" = --probe ] && exit 0\n{body}"),
    )
    .unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    for _ in 0..200 {
        match std::process::Command::new(path).arg("--probe").status() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            ran => {
                assert!(
                    ran.unwrap().success(),
                    "{} answers no --probe",
                    path.display()
                );
                return;
            }
        }
    }
    panic!("{} stayed busy", path.display());
}
