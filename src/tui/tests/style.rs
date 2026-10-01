use super::*;

/// WCAG contrast ratio between two sRGB colours.
fn contrast(a: [u8; 3], b: [u8; 3]) -> f32 {
    let linear = |c: u8| {
        let c = f32::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = |[r, g, b]: [u8; 3]| 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
    let (x, y) = (lum(a), lum(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

#[test]
fn menu_text_and_hints_read_clearly_on_the_page() {
    assert!(
        contrast(TEXT, BACKGROUND) >= 7.0,
        "{}",
        contrast(TEXT, BACKGROUND)
    );
    assert!(
        contrast(HINT, BACKGROUND) >= 4.5,
        "{}",
        contrast(HINT, BACKGROUND)
    );
    // Hints stay a step below the text they explain.
    assert!(contrast(HINT, BACKGROUND) < contrast(TEXT, BACKGROUND));
    // The old hint grey did not.
    assert!(contrast(MUTED, BACKGROUND) < 2.0);
}
