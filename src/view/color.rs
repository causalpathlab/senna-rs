//! Colour for `senna view`: an OKLCH categorical palette tuned for a light
//! background, and the linear-light ↔ sRGB conversions the renderer blends in.
//!
//! Every colour inside the renderer is linear RGB in `[0, 1]`; conversion to
//! 8-bit sRGB happens once per pixel at composite time.

pub type Rgb = [f32; 3];

/// Page background: a warm light gray.
pub const BACKGROUND: [u8; 3] = [246, 245, 242];
/// Points with no group, and points outside the focused group.
pub const MUTED: [u8; 3] = [200, 199, 196];
/// Status-line text.
pub const INK: [u8; 3] = [92, 92, 92];

#[must_use]
pub fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[must_use]
pub fn linear_rgb(c: [u8; 3]) -> Rgb {
    c.map(srgb_to_linear)
}

/// Linear `[0, 1]` → 8-bit sRGB through a 4096-entry table, built once.
pub struct Encoder {
    lut: Vec<u8>,
}

impl Encoder {
    const N: usize = 4096;

    #[must_use]
    pub fn new() -> Self {
        let lut = (0..Self::N)
            .map(|i| {
                let v = i as f32 / (Self::N - 1) as f32;
                let s = if v <= 0.003_130_8 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (s * 255.0).round().clamp(0.0, 255.0) as u8
            })
            .collect();
        Self { lut }
    }

    #[inline]
    #[must_use]
    pub fn encode(&self, v: f32) -> u8 {
        let i = (v.clamp(0.0, 1.0) * (Self::N - 1) as f32) as usize;
        self.lut[i]
    }
}

/// OKLCH (lightness, chroma, hue in degrees) → linear sRGB, clamped.
#[must_use]
pub fn oklch(l: f32, c: f32, h_deg: f32) -> Rgb {
    let h = h_deg.to_radians();
    let (a, b) = (c * h.cos(), c * h.sin());
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l3, m3, s3) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_7 * l3 - 3.307_711_6 * m3 + 0.230_969_94 * s3,
        -1.268_438 * l3 + 2.609_757_4 * m3 - 0.341_319_38 * s3,
        -0.004_196_086_3 * l3 - 0.703_418_6 * m3 + 1.707_614_7 * s3,
    ]
    .map(|v| v.clamp(0.0, 1.0))
}

/// Hue of category `i`: golden-angle steps, so neighbouring ids never get
/// neighbouring hues however many categories there are.
fn hue(i: usize) -> f32 {
    (25.0 + i as f32 * 137.507_76) % 360.0
}

/// Point colour of category `i`. Equal lightness keeps any one group from
/// shouting; past a dozen groups lightness alternates to buy separation.
#[must_use]
pub fn category(i: usize, n: usize) -> Rgb {
    let l = if n <= 12 {
        0.64
    } else {
        [0.64, 0.76, 0.52][i % 3]
    };
    oklch(l, 0.15, hue(i))
}

/// Label colour of category `i`: the same hue, darker, so text stays legible
/// on the light page.
#[must_use]
pub fn category_ink(i: usize) -> Rgb {
    oklch(0.42, 0.12, hue(i))
}

/// Sequential ramp for feature activity, light to deep, one hue family
/// (cool blue into violet) so it reads as "more" on a light page.
#[must_use]
pub fn activity_ramp(n: usize) -> Vec<Rgb> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            oklch(0.88 - 0.52 * t, 0.04 + 0.13 * t, 235.0 + 75.0 * t)
        })
        .collect()
}

/// Ink for a highlighted single feature.
#[must_use]
pub fn highlight_ink() -> Rgb {
    oklch(0.36, 0.17, 310.0)
}
