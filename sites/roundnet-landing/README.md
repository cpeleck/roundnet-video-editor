# Roundnet landing page

Static marketing site for the Roundnet Rally Editor. The palette, wordmark,
net symbol, and italic serif accent follow the existing editor interface.

## Preview

From this directory:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:4173. No package installation or build step is required.

## Content and interactions

- `dist/index.html`: landing-page copy, illustrative court/editor preview,
  workflow, export commitments, installation links, and FAQs.
- `dist/styles.css`: responsive layout, brand palette, and reduced-motion support.
- `dist/app.js`: sample clip/match-highlight selection and mobile navigation.
- `dist/LICENSE.txt`: the same standard MIT License as the main project.

The preview is explicitly illustrative; it does not process or play footage.
Source and installation buttons link to the real project repository. This
page does not advertise an unavailable installer or collect payments. No
analytics, external font requests, account flow, or forms are included.

Site identity and static hosting configuration are in `.openai/hosting.json`.
The Site is private by default. Review it before changing its audience for a
public launch. Publish the main project's license and current setup docs to
GitHub before that launch so the repository matches the landing-page promise.
