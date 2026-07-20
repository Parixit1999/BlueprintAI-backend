# Rendering & evidence

`app/services/rendering.py` + `render_service.py`. Produces the PNG the frontend
highlights on.

- `GET /files/{id}/render?page=N` → `{url (presigned), extents, page}`.
- Renderers: DXF via ezdxf drawing add-on + matplotlib (white background); PDF
  page rasterized via PyMuPDF; image passed through. Each returns
  `(png_bytes, extents)` where `extents = [xmin, ymin, xmax, ymax]` describes the
  coordinate space of the image in the same y-up convention the extractors use.
- **Lazy + cached**: rendered on first request, stored in object storage, and the
  metadata cached on `files.render` (`{pages: {N: {s3_key, extents}}}`), so files
  uploaded before this feature existed still work.
- The frontend maps a region's bbox to image percentages using `extents`
  (y-flipped) to draw the highlight box.
