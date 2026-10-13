//! The visible units' modality blocks. A multiome load names every feature
//! `{name}/{modality}`; each modality is its own multinomial over its own
//! features at its own depth. A plain load is one block.
//!
//! The machine keeps every block's features contiguous: [`Blocks::perm`]
//! takes the machine's feature order to the data's row order.

use legume_numeric::candle::candle_core::{Result, Tensor};
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct Blocks {
    /// Every block's name (`""` for a plain load).
    pub names: Vec<Box<str>>,
    /// Every block's features, in the machine's order.
    pub ranges: Vec<Range<usize>>,
    /// The machine's feature `i` is the data's row `perm[i]`.
    pub perm: Vec<usize>,
    /// Every machine feature's symbol (the name before `/{modality}`).
    pub symbol: Vec<u32>,
    pub symbols: Vec<Box<str>>,
}

impl Blocks {
    /// Blocks read off the data's row names. Rows split into blocks only when
    /// every row has exactly one `/` (the multiome grammar); otherwise the
    /// whole axis is one block.
    pub fn from_names(rows: &[Box<str>]) -> Self {
        let modalities = senna::multiome_layout::trained_modalities(rows);
        let block_of = |r: &str| -> usize {
            r.rsplit_once('/')
                .and_then(|(_, m)| modalities.iter().position(|x| x.as_ref() == m))
                .unwrap_or(0)
        };
        let n_blocks = modalities.len().max(1);
        let mut perm = Vec::with_capacity(rows.len());
        let mut ranges = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let lb = perm.len();
            perm.extend((0..rows.len()).filter(|&i| block_of(&rows[i]) == b));
            ranges.push(lb..perm.len());
        }
        let names = if modalities.is_empty() {
            vec!["".into()]
        } else {
            modalities
        };
        let multi = names.len() > 1;
        let mut symbols: Vec<Box<str>> = Vec::new();
        let mut index = rustc_hash::FxHashMap::default();
        let symbol = perm
            .iter()
            .map(|&i| {
                let r = rows[i].as_ref();
                let s = if multi {
                    r.rsplit_once('/').map_or(r, |(s, _)| s)
                } else {
                    r
                };
                *index.entry(Box::<str>::from(s)).or_insert_with(|| {
                    symbols.push(s.into());
                    (symbols.len() - 1) as u32
                })
            })
            .collect();
        Self {
            names,
            ranges,
            perm,
            symbol,
            symbols,
        }
    }

    /// A data-ordered `[F × N]` table as machine-ordered rows `[N × F]`.
    pub fn machine_rows(&self, t_fn: &Tensor) -> Result<Tensor> {
        let perm = Tensor::from_iter(self.perm.iter().map(|&i| i as u32), t_fn.device())?;
        t_fn.contiguous()?.index_select(&perm, 0)?.t()?.contiguous()
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn features(&self) -> usize {
        self.perm.len()
    }

    /// The machine's feature names, in its order.
    pub fn feature_names(&self, rows: &[Box<str>]) -> Vec<Box<str>> {
        self.perm.iter().map(|&i| rows[i].clone()).collect()
    }
}
