//! What `senna view` has senna compute for a run: its layouts and
//! clusterings, missing ones on open and chosen ones from the `r` menu. One
//! place builds the commands, so both run the same thing.

use std::process::{Command, Stdio};

/// Leiden resolutions the menu steps through for a clustering, 1 (the
/// clustering default) in the middle; higher gives more clusters.
pub(crate) const RESOLUTIONS: [&str; 9] = ["0.2", "0.3", "0.5", "0.7", "1", "1.5", "2", "3", "5"];

/// One thing senna computes for a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    CellLayout,
    FeatureLayout,
    /// Cells and co-embedded features in one layout: the `joint` method.
    JointLayout,
    CellClusters,
    FeatureClusters,
}

impl Step {
    pub const ALL: [Step; 5] = [
        Step::CellLayout,
        Step::FeatureLayout,
        Step::JointLayout,
        Step::CellClusters,
        Step::FeatureClusters,
    ];

    /// The layout that made a map of `kind` from method `method`: features
    /// on a cell map come with the cell layout, both with the joint one.
    pub fn on_screen(kind: crate::view::SpaceKind, method: &str) -> Self {
        match Self::for_method(method) {
            Step::CellLayout if kind == crate::view::SpaceKind::Features => Step::FeatureLayout,
            step => step,
        }
    }

    /// The layout that makes a cell map of method `method`.
    pub fn for_method(method: &str) -> Self {
        if method == "joint" {
            Step::JointLayout
        } else {
            Step::CellLayout
        }
    }

    fn is_clustering(self) -> bool {
        matches!(self, Step::CellClusters | Step::FeatureClusters)
    }

    /// What the step's one setting can be: a layout's methods (the default
    /// first; only umap and phate lay out features), a clustering's
    /// resolutions, nothing for the joint layout.
    pub fn settings(self) -> &'static [&'static str] {
        match self {
            Step::CellLayout => &["umap", "phate", "tsne"],
            Step::FeatureLayout => &["umap", "phate"],
            Step::CellClusters | Step::FeatureClusters => &RESOLUTIONS,
            Step::JointLayout => &[],
        }
    }

    /// Where the menu starts: resolution 1 for a clustering, else the
    /// method's default.
    fn default_setting(self) -> &'static str {
        if self.is_clustering() {
            "1"
        } else {
            self.settings().first().copied().unwrap_or("")
        }
    }

    /// A setting as said on screen: a method as it is, a resolution named;
    /// none for no setting.
    pub fn say(self, setting: &str) -> Option<String> {
        match setting {
            "" => None,
            r if self.is_clustering() => Some(format!("resolution {r}")),
            m => Some(m.to_string()),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Step::CellLayout => "cell layout",
            Step::FeatureLayout => "feature layout",
            Step::JointLayout => "cells + features (joint)",
            Step::CellClusters => "cell clusters",
            Step::FeatureClusters => "feature clusters",
        }
    }

    /// senna's arguments for this step on `t` with `setting`: a layout's
    /// method, or a clustering's Leiden resolution (empty: its default).
    pub fn argv(self, t: &Target, setting: &str) -> Vec<String> {
        let (from, out) = (t.from.as_str(), t.out.as_str());
        let v: Vec<&str> = if self.is_clustering() {
            let mut v = vec!["clustering", "--from", from, "-m", "leiden", "-o", out];
            match (self, &t.latent) {
                (Step::CellClusters, Some(latent)) => v.extend(["--latent", latent]),
                (Step::FeatureClusters, _) => v.extend(["--target", "features"]),
                _ => {}
            }
            if !setting.is_empty() {
                v.extend(["--resolution", setting]);
            }
            v
        } else {
            // The joint layout is umap over cells and features together.
            let (method, extra): (&str, &[&str]) = match self {
                Step::FeatureLayout => (setting, &["--target", "features"]),
                Step::JointLayout => ("umap", &["--joint"]),
                _ => (setting, &[]),
            };
            [
                &["layout", method][..],
                extra,
                &["--from", from, "--out", out],
            ]
            .concat()
        };
        v.into_iter().map(String::from).collect()
    }
}

/// A run to compute for: its manifest, where outputs go, and what it has.
#[derive(Clone, Debug)]
pub(crate) struct Target {
    pub from: String,
    pub out: String,
    /// The table cells are clustered on, when the run has cells.
    pub latent: Option<String>,
    pub has_cells: bool,
    pub has_features: bool,
    /// Features co-embedded in the cells' space, which a joint layout needs.
    pub has_coembedding: bool,
}

impl Target {
    /// The run whose manifest is `from`; outputs go beside it.
    pub fn load(from: &str) -> anyhow::Result<Self> {
        let (m, dir) = senna::run_manifest::RunManifest::load(std::path::Path::new(from))?;
        Ok(Self::from_manifest(from, &m, &dir))
    }

    /// The run of manifest `m` at `from`, read from directory `dir`.
    pub fn from_manifest(
        from: &str,
        m: &senna::run_manifest::RunManifest,
        dir: &std::path::Path,
    ) -> Self {
        let has_cells = m.kind.has_cells();
        let latent = m.outputs.geometry_latent().filter(|_| has_cells).map(|l| {
            senna::run_manifest::resolve(dir, l)
                .to_string_lossy()
                .into_owned()
        });
        Self {
            from: from.into(),
            out: senna::run_manifest::derive_out_prefix(from),
            latent,
            has_cells,
            has_features: m.outputs.feature_embedding.is_some(),
            has_coembedding: m.kind.cell_space() == senna::run_manifest::CellSpace::Embedding
                && m.outputs.feature_coembedding.is_some(),
        }
    }

    /// The steps this run can take.
    pub fn offered(&self) -> Vec<Step> {
        Step::ALL
            .into_iter()
            .filter(|s| match s {
                Step::CellLayout => self.has_cells,
                Step::CellClusters => self.latent.is_some(),
                Step::FeatureLayout | Step::FeatureClusters => self.has_features,
                Step::JointLayout => self.has_coembedding,
            })
            .collect()
    }
}

/// The `r` menu: the steps offered, which are chosen, and their methods.
pub(crate) struct Menu {
    pub items: Vec<Item>,
    /// The item under the cursor.
    pub at: usize,
}

pub(crate) struct Item {
    pub step: Step,
    pub on: bool,
    /// Index of the chosen one of `step.settings()`.
    pub setting: usize,
}

impl Item {
    pub fn setting(&self) -> &'static str {
        self.step
            .settings()
            .get(self.setting)
            .copied()
            .unwrap_or("")
    }
}

impl Menu {
    /// The steps `t` offers, with `on_screen` chosen and under the cursor,
    /// laid out with `method` when it can (else its default); clusterings
    /// start at resolution 1.
    pub fn new(t: &Target, on_screen: Step, method: &str) -> Self {
        let items: Vec<Item> = t
            .offered()
            .into_iter()
            .map(|step| {
                let wanted = if step == on_screen && !step.is_clustering() {
                    method
                } else {
                    step.default_setting()
                };
                let setting = step
                    .settings()
                    .iter()
                    .position(|s| *s == wanted)
                    .unwrap_or(0);
                Item {
                    step,
                    on: step == on_screen,
                    setting,
                }
            })
            .collect();
        let at = items.iter().position(|i| i.on).unwrap_or(0);
        Self { items, at }
    }

    /// Move the cursor by `d`, wrapping.
    pub fn step(&mut self, d: isize) {
        let n = self.items.len().max(1) as isize;
        self.at = (self.at as isize + d).rem_euclid(n) as usize;
    }

    pub fn toggle(&mut self) {
        if let Some(i) = self.items.get_mut(self.at) {
            i.on = !i.on;
        }
    }

    /// The next (`d` = 1) or previous setting of the item under the cursor.
    pub fn step_setting(&mut self, d: isize) {
        if let Some(i) = self.items.get_mut(self.at) {
            let n = i.step.settings().len() as isize;
            if n > 0 {
                i.setting = (i.setting as isize + d).rem_euclid(n) as usize;
            }
        }
    }

    /// What enter would run, in order.
    pub fn chosen(&self) -> Vec<(Step, &'static str)> {
        self.items
            .iter()
            .filter(|i| i.on)
            .map(|i| (i.step, i.setting()))
            .collect()
    }
}

/// Stops a running recompute: the step running is killed, and none after
/// it starts.
#[derive(Default)]
pub(crate) struct Stopper {
    stopped: std::sync::atomic::AtomicBool,
    child: std::sync::Mutex<Option<std::process::Child>>,
}

impl Stopper {
    pub fn stop(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Ok(mut c) = self.child.lock() {
            if let Some(c) = c.as_mut() {
                let _ = c.kill();
            }
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Run each chosen step as `senna …` in turn, its latest log line in
/// `progress`. Stops at the first that fails, with that step and its last
/// line, or when `stopper` is stopped.
pub(crate) fn run(
    t: &Target,
    chosen: &[(Step, &str)],
    progress: &std::sync::Mutex<String>,
    stopper: &Stopper,
) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let commands = chosen
        .iter()
        .map(|&(step, setting)| {
            let mut c = Command::new(&exe);
            c.args(step.argv(t, setting));
            (step.label(), c)
        })
        .collect();
    run_commands(commands, progress, stopper)
}

/// Run `commands` in turn, each named for `progress`; see `run`.
fn run_commands(
    commands: Vec<(&str, Command)>,
    progress: &std::sync::Mutex<String>,
    stopper: &Stopper,
) -> Result<(), String> {
    let say = |text: String| {
        if let Ok(mut p) = progress.lock() {
            *p = text;
        }
    };
    for (label, mut command) in commands {
        if stopper.is_stopped() {
            return Err("stopped".into());
        }
        say(format!("{label}…"));
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot run senna: {e}"))?;
        let log = child.stderr.take();
        // Where `stop` can reach it; stopped meanwhile, it goes at once.
        if let Ok(mut c) = stopper.child.lock() {
            *c = Some(child);
        }
        if stopper.is_stopped() {
            stopper.stop();
        }
        let last = crate::view::decide::follow_log(log, |line| say(format!("{label}: {line}")));
        let status = stopper
            .child
            .lock()
            .ok()
            .and_then(|mut c| c.take())
            .map(|mut c| c.wait());
        if stopper.is_stopped() {
            return Err("stopped".into());
        }
        match status {
            Some(Ok(s)) if s.success() => {}
            Some(Err(e)) => return Err(format!("{label} failed: {e}")),
            _ => {
                let why = last.strip_prefix("Error: ").unwrap_or(&last);
                return Err(format!("{label} failed: {why}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(latent: bool) -> Target {
        Target {
            from: "run.senna.json".into(),
            out: "run".into(),
            latent: latent.then(|| "./run.latent.parquet".into()),
            has_cells: true,
            has_features: true,
            has_coembedding: false,
        }
    }

    fn with_coembedding() -> Target {
        Target {
            has_coembedding: true,
            ..target(true)
        }
    }

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn each_step_runs_what_opening_a_run_would() {
        let t = target(true);
        assert_eq!(
            Step::CellLayout.argv(&t, "phate"),
            argv(&[
                "layout",
                "phate",
                "--from",
                "run.senna.json",
                "--out",
                "run"
            ])
        );
        assert_eq!(
            Step::FeatureLayout.argv(&t, "umap"),
            argv(&[
                "layout",
                "umap",
                "--target",
                "features",
                "--from",
                "run.senna.json",
                "--out",
                "run"
            ])
        );
        assert_eq!(
            Step::CellClusters.argv(&t, ""),
            argv(&[
                "clustering",
                "--from",
                "run.senna.json",
                "-m",
                "leiden",
                "-o",
                "run",
                "--latent",
                "./run.latent.parquet"
            ])
        );
        assert_eq!(
            Step::FeatureClusters.argv(&t, ""),
            argv(&[
                "clustering",
                "--from",
                "run.senna.json",
                "-m",
                "leiden",
                "-o",
                "run",
                "--target",
                "features"
            ])
        );
    }

    #[test]
    fn a_run_offers_only_what_it_can_compute() {
        assert_eq!(
            target(true).offered(),
            vec![
                Step::CellLayout,
                Step::FeatureLayout,
                Step::CellClusters,
                Step::FeatureClusters,
            ]
        );
        // No latent: nothing to cluster cells on.
        assert!(!target(false).offered().contains(&Step::CellClusters));
        let cells_only = Target {
            has_features: false,
            ..target(true)
        };
        assert_eq!(
            cells_only.offered(),
            vec![Step::CellLayout, Step::CellClusters]
        );
        let features_only = Target {
            has_cells: false,
            latent: None,
            ..target(true)
        };
        assert_eq!(
            features_only.offered(),
            vec![Step::FeatureLayout, Step::FeatureClusters]
        );
    }

    #[test]
    fn layouts_set_their_method_and_clusterings_their_resolution() {
        let t = target(true);
        assert_eq!(Step::CellLayout.settings(), ["umap", "phate", "tsne"]);
        assert_eq!(Step::FeatureLayout.settings(), ["umap", "phate"]);
        assert_eq!(Step::CellClusters.settings(), RESOLUTIONS);
        assert_eq!(Step::FeatureClusters.settings(), RESOLUTIONS);
        // A chosen resolution goes to leiden; none (opening a run) leaves
        // clustering's own default.
        let with = Step::FeatureClusters.argv(&t, "2");
        assert_eq!(&with[with.len() - 2..], &["--resolution", "2"]);
        assert!(!Step::FeatureClusters
            .argv(&t, "")
            .contains(&"--resolution".to_string()));
    }

    #[test]
    fn the_menu_starts_clusterings_at_resolution_one_and_steps_it() {
        let mut m = Menu::new(&target(true), Step::CellLayout, "umap");
        m.step(2); // cell clusters
        m.toggle();
        assert_eq!(m.chosen()[1], (Step::CellClusters, "1"));
        m.step_setting(1);
        m.step_setting(1);
        assert_eq!(m.chosen()[1], (Step::CellClusters, "2"));
        m.step_setting(-3);
        assert_eq!(m.chosen()[1], (Step::CellClusters, "0.7"));
    }

    #[test]
    fn a_run_with_a_coembedding_also_offers_cells_and_features_laid_out_together() {
        let t = with_coembedding();
        assert_eq!(
            t.offered(),
            vec![
                Step::CellLayout,
                Step::FeatureLayout,
                Step::JointLayout,
                Step::CellClusters,
                Step::FeatureClusters,
            ]
        );
        assert!(!target(true).offered().contains(&Step::JointLayout));
        // One line of its own, no method to pick: umap over cells and
        // features together, recorded as the `joint` method.
        assert!(Step::JointLayout.settings().is_empty());
        assert_eq!(
            Step::JointLayout.argv(&t, ""),
            argv(&[
                "layout",
                "umap",
                "--joint",
                "--from",
                "run.senna.json",
                "--out",
                "run"
            ])
        );
        let m = Menu::new(&t, Step::JointLayout, "joint");
        assert_eq!(m.chosen(), vec![(Step::JointLayout, "")]);
        assert_eq!(m.at, 2);
    }

    #[test]
    fn the_map_on_screen_names_the_step_to_redo() {
        use crate::view::SpaceKind;
        assert_eq!(Step::on_screen(SpaceKind::Cells, "umap"), Step::CellLayout);
        assert_eq!(
            Step::on_screen(SpaceKind::FeaturesOnCells, "phate"),
            Step::CellLayout
        );
        assert_eq!(
            Step::on_screen(SpaceKind::Features, "umap"),
            Step::FeatureLayout
        );
        assert_eq!(
            Step::on_screen(SpaceKind::Cells, "joint"),
            Step::JointLayout
        );
        assert_eq!(
            Step::on_screen(SpaceKind::FeaturesOnCells, "joint"),
            Step::JointLayout
        );
    }

    #[test]
    fn the_menu_starts_on_the_layout_on_screen_with_its_method() {
        let m = Menu::new(&target(true), Step::FeatureLayout, "phate");
        let chosen: Vec<(Step, &str)> = m.chosen();
        assert_eq!(chosen, vec![(Step::FeatureLayout, "phate")]);
        assert_eq!(m.at, 1);
        // A method the step cannot use falls back to its first.
        let m = Menu::new(&target(true), Step::FeatureLayout, "tsne");
        assert_eq!(m.chosen(), vec![(Step::FeatureLayout, "umap")]);
    }

    #[test]
    fn the_menu_toggles_moves_and_changes_methods() {
        let mut m = Menu::new(&target(true), Step::CellLayout, "umap");
        m.step_setting(1);
        m.step(2);
        m.toggle();
        m.step_setting(1); // a clustering's resolution: 1 to 1.5
        m.step(-3); // wraps to the last item
        m.toggle();
        assert_eq!(
            m.chosen(),
            vec![
                (Step::CellLayout, "phate"),
                (Step::CellClusters, "1.5"),
                (Step::FeatureClusters, "1"),
            ]
        );
        m.step(-1);
        m.step(-1);
        m.step(-1);
        m.toggle();
        assert!(m.chosen().iter().all(|(s, _)| *s != Step::CellLayout));
    }

    #[test]
    fn a_missing_joint_map_is_redone_as_a_joint_layout() {
        // What opening a run recomputes for its current method.
        let t = with_coembedding();
        let step = Step::for_method("joint");
        assert_eq!(step, Step::JointLayout);
        assert_eq!(
            step.argv(&t, "joint"),
            argv(&[
                "layout",
                "umap",
                "--joint",
                "--from",
                "run.senna.json",
                "--out",
                "run"
            ])
        );
        assert_eq!(Step::for_method("phate"), Step::CellLayout);
    }

    #[test]
    fn a_stopped_run_kills_its_step_and_starts_no_more() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("second-ran");
        let mut slow = Command::new("sleep");
        slow.arg("30");
        let mut next = Command::new("touch");
        next.arg(&marker);
        let stopper = std::sync::Arc::new(Stopper::default());
        let progress = std::sync::Mutex::new(String::new());
        let s = stopper.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            s.stop();
        });
        let started = std::time::Instant::now();
        let out = run_commands(vec![("slow", slow), ("next", next)], &progress, &stopper);
        assert_eq!(out, Err("stopped".into()));
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
        assert!(!marker.exists());
    }

    #[test]
    fn a_run_left_alone_runs_every_step() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("ran");
        let mut touch = Command::new("touch");
        touch.arg(&marker);
        let progress = std::sync::Mutex::new(String::new());
        let out = run_commands(vec![("touch", touch)], &progress, &Stopper::default());
        assert_eq!(out, Ok(()));
        assert!(marker.exists());
    }
}
