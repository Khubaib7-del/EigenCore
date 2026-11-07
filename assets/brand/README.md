# EigenCore brand kit

Provided brand assets. Use the animated banner where SVG motion is supported; the static PNG is the fallback for contexts that disable animation.

| File | Use |
| --- | --- |
| `eigencore-banner-animated.svg` | README header banner (animated) |
| `eigencore-banner.svg` / `eigencore-banner.png` | Static banner fallback |
| `eigencore-logo.svg` / `eigencore-logo.png` | Square logo mark, transparent background |
| `eigencore-lockup-dark.*` / `eigencore-lockup-light.*` | Logo + wordmark lockups |
| `favicon.*` | Browser/docs-site icons |

For an HTML docs site, add (paths shown relative to the repository root):

```html
<link rel="icon" href="assets/brand/favicon.svg" type="image/svg+xml">
<link rel="icon" href="assets/brand/favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="assets/brand/favicon-apple-touch.png">
```

Palette: near-black `#07090D` / `#0E1218`, indigo-blue `#3C4FB8`, slate `#323A4A` / `#5E6982`, ice `#E2E7FF` / `#93A5FF`.
