use super::*;

/// Text files, several at once when `many`; `.zarr` stores too.
#[derive(Clone)]
struct Txt {
    many: bool,
}

/// Text files, refusing any named `no.txt`.
#[derive(Clone)]
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
    let said: Vec<String> = b.lines(20).iter().map(ToString::to_string).collect();
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

/// Text files that can be taken several at once, refusing `no.txt`.
#[derive(Clone)]
struct PickyMany;

impl Wanted for PickyMany {
    type About = ();

    fn header(&self) -> Header {
        Picky.header()
    }

    fn file(&self, path: &Path, name: &str) -> Option<()> {
        Picky.file(path, name)
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        "".into()
    }

    fn refuse(&self, name: &str, (): &()) -> Option<String> {
        Picky.refuse(name, &())
    }

    fn many(&self) -> bool {
        true
    }
}

#[test]
fn refused_files_are_never_marked() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    write(dir.path(), "yes.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), PickyMany, Some("no.txt"));
    b.key(key(KeyCode::Char(' ')));
    assert!(b.marked.is_empty());
    assert_eq!(b.refused.as_deref(), Some("no.txt will not do"));
    b.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
    assert_eq!(
        b.marked.iter().collect::<Vec<_>>(),
        [&dir.path().join("yes.txt")]
    );
    assert!(b.refused.is_some(), "the refused one is named");
}

#[test]
fn a_refusal_stays_through_keys_that_do_nothing() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Picky, Some("no.txt"));
    b.key(key(KeyCode::Enter));
    assert!(b.refused.is_some());
    assert_eq!(b.key(key(KeyCode::Tab)), Outcome::Ignored);
    assert!(b.refused.is_some());
}

#[test]
fn the_key_hints_fit_however_short_the_popup() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..40 {
        write(dir.path(), &format!("f{i:02}.txt"));
    }
    let b = Browser::open(dir.path().to_path_buf(), Txt { many: true }, None);
    let lines = b.lines(20);
    assert!(lines.len() + 2 <= 20, "{}", lines.len());
    assert!(lines.last().unwrap().to_string().contains("esc cancel"));
}

/// Type `text` into the browser a key at a time.
fn type_in<W: Wanted>(b: &mut Browser<W>, text: &str) {
    for c in text.chars() {
        b.key(key(KeyCode::Char(c)));
    }
}

#[test]
fn a_typed_path_is_followed_and_a_folder_it_reaches_is_entered_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().canonicalize().unwrap();
    for sub in ["data", "data2", "other"] {
        std::fs::create_dir(d.join(sub)).unwrap();
    }
    write(&d.join("data2"), "a.txt");
    let start = d.join("other");
    let mut b = Browser::open(start.clone(), Txt { many: false }, None);
    // The whole path typed as it is: each folder it passes is listed.
    let root = format!("{}/", d.display());
    type_in(&mut b, &root);
    assert_eq!(b.dir, d);
    assert_eq!(b.filter, root);
    // `data` is a folder, but `data2` begins with it too: wait.
    type_in(&mut b, "data");
    assert_eq!(b.dir, d);
    assert_eq!(names(&b), ["..", "data", "data2"]);
    // `data2` is the only one: its folder is listed at once, and what was
    // typed stays as typed.
    type_in(&mut b, "2");
    assert_eq!(b.dir, d.join("data2"));
    assert_eq!(b.filter, format!("{root}data2"));
    assert_eq!(names(&b), ["..", "a.txt"]);
    // The `/` typed next is just that: the same folder.
    type_in(&mut b, "/");
    assert_eq!(b.filter, format!("{root}data2/"));
    assert_eq!(b.dir, d.join("data2"));
    // Backspace back past the folder lists its parent again.
    b.key(key(KeyCode::Backspace));
    b.key(key(KeyCode::Backspace));
    assert_eq!(b.dir, d);
    assert_eq!(names(&b), ["..", "data", "data2"]);
    // Enter takes the file the path ends on.
    type_in(&mut b, "2/a");
    assert_eq!(taken(b.key(key(KeyCode::Enter))), [d.join("data2/a.txt")]);
}

#[test]
fn letters_without_a_leading_slash_still_narrow_names() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("data2")).unwrap();
    write(dir.path(), "a.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: false }, None);
    type_in(&mut b, "data2");
    assert_eq!(b.dir, dir.path());
    assert_eq!(names(&b), ["..", "data2"]);
}

#[test]
fn a_typed_path_that_is_not_there_stays_where_it_is() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: false }, None);
    let gone = format!("{}/nowhere/x", dir.path().display());
    type_in(&mut b, &gone);
    assert_eq!(b.dir, dir.path());
    assert!(!b.completed());
    assert_eq!(names(&b), [".."]);
}

/// Text files, each described slowly: a folder that takes a while to list.
#[derive(Clone)]
struct Slow;

impl Wanted for Slow {
    type About = ();

    fn header(&self) -> Header {
        Txt { many: false }.header()
    }

    fn file(&self, _path: &Path, name: &str) -> Option<()> {
        std::thread::sleep(std::time::Duration::from_millis(80));
        name.ends_with(".txt").then_some(())
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        std::borrow::Cow::Borrowed("")
    }
}

#[test]
fn a_slow_folder_is_listed_on_the_side_and_shown_when_done() {
    let dir = tempfile::tempdir().unwrap();
    for f in ["a.txt", "b.txt", "c.txt"] {
        write(dir.path(), f);
    }
    let mut b = Browser::open(dir.path().to_path_buf(), Slow, Some("b.txt"));
    // Too slow to wait for: the browser is up at once, listing.
    assert!(b.listing().is_some());
    assert_eq!(names(&b), [".."]);
    let lines: Vec<String> = b.lines(30).iter().map(ToString::to_string).collect();
    assert!(
        lines.iter().any(|l| l.contains("listing this folder")),
        "{lines:?}"
    );
    while !b.poll() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(b.listing().is_none());
    assert_eq!(names(&b), ["..", "a.txt", "b.txt", "c.txt"]);
    // The cursor goes where it was asked, once the folder is there.
    assert_eq!(b.current().map(Entry::name), Some("b.txt"));
}
