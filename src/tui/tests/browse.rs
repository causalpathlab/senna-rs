use super::*;

/// Text files, several at once when `many`; `.zarr` stores too.
struct Txt {
    many: bool,
}

/// Text files, refusing any named `no.txt`.
struct Picky;

impl Wanted for Picky {
    type About = ();

    fn header(&self) -> Header {
        Txt { many: false }.header()
    }

    fn file(&self, _path: &Path, name: &str) -> Option<()> {
        name.ends_with(".txt").then_some(())
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        "".into()
    }

    fn refuse(&self, name: &str, (): &()) -> Option<String> {
        (name == "no.txt").then(|| format!("{name} will not do"))
    }
}

impl Wanted for Txt {
    type About = String;

    fn header(&self) -> Header {
        Header {
            title: "Text".into(),
            notes: Vec::new(),
            what: "text files",
            star: None,
            verb: "take",
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        name.ends_with(".txt").then(|| size_of(path))
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<String> {
        Some(String::new())
    }

    fn describe<'a>(&self, about: &'a String) -> std::borrow::Cow<'a, str> {
        std::borrow::Cow::Borrowed(about)
    }

    fn many(&self) -> bool {
        self.many
    }
}

fn write(dir: &Path, name: &str) {
    std::fs::write(dir.join(name), "").unwrap();
}

fn key(c: KeyCode) -> KeyEvent {
    KeyEvent::new(c, KeyModifiers::NONE)
}

/// The files a key took; none when it took none.
fn taken(o: Outcome) -> Vec<PathBuf> {
    match o {
        Outcome::Chosen(c) => c.files(),
        _ => Vec::new(),
    }
}

fn names<W: Wanted>(b: &Browser<W>) -> Vec<String> {
    b.shown().iter().map(|e| e.name().to_string()).collect()
}

#[test]
fn folders_come_before_wanted_files_and_stores_are_files() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(d, "b.txt");
    write(d, "a.txt");
    write(d, "skip.csv");
    std::fs::create_dir(d.join("sub")).unwrap();
    std::fs::create_dir(d.join("s.zarr")).unwrap();
    std::fs::create_dir(d.join(".hidden")).unwrap();
    let mut b = Browser::open(d.to_path_buf(), Txt { many: false }, None);
    assert_eq!(names(&b), ["..", "sub", "a.txt", "b.txt", "s.zarr"]);
    // Hidden entries show only when asked for with a leading `.`.
    b.filter.push('.');
    assert!(names(&b).contains(&".hidden".to_string()));
}

#[test]
fn enter_chooses_one_file_and_opens_folders() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    write(&dir.path().join("sub"), "a.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: false }, None);
    assert_eq!(b.current().map(Entry::name), Some("sub"));
    assert_eq!(b.key(key(KeyCode::Enter)), Outcome::Moved);
    assert_eq!(b.current().map(Entry::name), Some("a.txt"));
    // Space narrows a single-choice browser like any letter.
    assert_eq!(b.key(key(KeyCode::Char(' '))), Outcome::Moved);
    assert_eq!(b.filter, " ");
    b.key(key(KeyCode::Backspace));
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("sub/a.txt")]
    );
    b.go_up();
    assert_eq!(b.current().map(Entry::name), Some("sub"));
}

#[test]
fn marked_files_in_several_folders_are_taken_together() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt");
    std::fs::create_dir(dir.path().join("more")).unwrap();
    write(&dir.path().join("more"), "b.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: true }, None);
    assert_eq!(names(&b), ["..", "more", "a.txt"]);
    b.row = 2;
    b.key(key(KeyCode::Char(' ')));
    b.row = 1;
    assert_eq!(
        b.key(key(KeyCode::Enter)),
        Outcome::Moved,
        "a folder still opens"
    );
    assert_eq!(b.dir, dir.path().join("more"));
    b.row = 1;
    b.key(key(KeyCode::Char(' ')));
    b.row = 1;
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("a.txt"), dir.path().join("more/b.txt")]
    );
    assert!(b.marked.is_empty());
}

#[test]
fn sizes_read_in_their_unit() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f");
    std::fs::write(&p, vec![0u8; 1500]).unwrap();
    assert_eq!(size_of(&p), "1.5 kB");
    std::fs::write(&p, b"abc").unwrap();
    assert_eq!(size_of(&p), "3 B");
}

#[test]
fn a_refused_file_is_not_taken_and_the_popup_says_why() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    write(dir.path(), "yes.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Picky, Some("no.txt"));
    assert_eq!(b.key(key(KeyCode::Enter)), Outcome::Moved);
    let said: Vec<String> = b.lines(10).iter().map(ToString::to_string).collect();
    assert!(
        said.iter().any(|l| l.contains("no.txt will not do")),
        "{said:?}"
    );
    b.key(key(KeyCode::Down));
    assert!(b.refused.is_none());
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("yes.txt")]
    );
}
