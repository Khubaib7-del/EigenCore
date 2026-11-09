# EigenCore brand kit v2

Flat v2 identity. Spec: `docs/design/DESIGN_HANDBOOK.md`. SVGs are the source; PNG and ICO files are exports of them.

| File | Use |
| --- | --- |
| `eigencore-banner-animated.svg` | README header (animation plays once, respects reduced motion) |
| `eigencore-banner.svg` / `.png` | Static banner |
| `eigencore-logo.svg` / `.png` | Full mark, transparent background, 48 px and up |
| `eigencore-mark-small.svg`, `favicon.*` | Small mark (no traces or axis), 16-32 px |
| `eigencore-lockup-dark.*` | Mark + wordmark for **dark** grounds ("Core" in indigo) |
| `eigencore-lockup-light.*` | Mark + wordmark for **light** grounds (word all ink) |
| `eigencore-wordmark-light.svg` / `-ink.svg` | Wordmark alone (outlined Space Grotesk Bold) |

Palette: graphite `#1F2226`, ink `#1B1E22`, stone `#ECE9E2`, die `#424B5E` / rim `#6A7489` / shadow `#15181E`, core `#5B6CF0` / deep `#2E3A9E` / glint `#C9CFFF`, core-ice `#8C98FF`. Signals: teal `#1F6B66` / `#4FB3A9`, amber `#E0A84A` (dark only), clay `#B04E2C` / `#E07A5A`.

Regenerate the raster files from the SVGs (Python, `pip install cairosvg`, ImageMagick):

```bash
python - <<'PY'
import cairosvg
cairosvg.svg2png(url="eigencore-banner.svg", write_to="eigencore-banner.png", output_width=1600)
cairosvg.svg2png(url="eigencore-logo.svg", write_to="eigencore-logo.png", output_width=512)
for n in ("dark", "light"):
    cairosvg.svg2png(url=f"eigencore-lockup-{n}.svg", write_to=f"eigencore-lockup-{n}.png", output_height=512)
for s in (16, 32, 512):
    cairosvg.svg2png(url="favicon.svg", write_to=f"favicon-{s}.png", output_width=s, output_height=s)
cairosvg.svg2png(url="favicon.svg", write_to="favicon-apple-touch.png", output_width=180, output_height=180, background_color="#1F2226")
PY
magick favicon-512.png -define icon:auto-resize=48,32,16 favicon.ico
```

> **Transition note (2026-11-09):** the SVG files here are v2. The PNG and ICO files in this folder are still the v1 exports until the regenerated files are committed.
