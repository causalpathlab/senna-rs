//! Unit tests for [`super`]: sample-id suffix inference.

use super::*;

/// Gene files carry `_count`; the sample id is the name without it.
#[test]
fn gene_file_default_strips_count() {
    assert_eq!(&*strip_any_suffix("s1_count", &[COUNT_SUFFIX]), "s1");
    assert_eq!(&*strip_any_suffix("s1", &[COUNT_SUFFIX]), "s1");
}
