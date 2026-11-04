# The training pictures

The training pictures are anime illustrations from [Danbooru](https://danbooru.donmai.us). This folder
does **not** contain them. It only holds the list [`danbooru.json`](danbooru.json), and
`blur-detector fetch` downloads the listed pictures when you need them.

**Copyright.** Every picture belongs to its artist. The repository only stores the list (post ids,
artists, sources and checksums) and does not distribute the pictures. Models trained on them are for
demonstrating this project only. The repository's MIT licence covers the code, not these pictures.

| | Pictures | What they are |
|---|---|---|
| `train/nonblur` | 103 | listed posts with `"split": "train"`, saved as `danbooru-<post id>.png` |
| `test/nonblur` | 20 | listed posts with `"split": "test"`, never used for training |
| `train/blur`, `test/blur` | 103 + 20 | blurred copies made by `generate` |
| `demo.png` | 1 | a scene drawn by code (no download needed), the default picture of `run` and `predict` |

## How the list was made

1. **Search.** [`src/detector/tools/pick_danbooru.py`](../src/detector/tools/pick_danbooru.py) searched
   Danbooru's best-scored posts with `rating:g order:score` (general audience only).
2. **Tag filter.** It dropped posts with any tag in `DANBOORU_EXCLUDED_TAGS`
   ([`config.py`](../src/detector/blur_detection/config.py)): anything not fully all-ages, AI-generated
   pictures, comics and text-heavy pictures, photos, 3D, sketches, monochrome, and pictures that are blurry
   on purpose (depth of field, bokeh, motion blur). It also dropped posts without exactly one artist tag,
   pictures under 800 px or longer than 2 : 1, and more than two pictures from the same artist.
3. **Eye check.** A person then looked at the 356 candidates and kept 128. They dropped anything
   with any doubt: suggestive poses or clothes, romance, blood or menace, alcohol, real people, brand
   logos, lots of text, comic panels, and two pictures whose artists ask not to use them for AI or
   without permission. One candidate never made it onto a review sheet, so it was not looked at and is
   not listed. Of the 128, five square ones are the web demo's pictures and stay out of this list.
4. **Split.** Of the other 123, 20 went to `test` and 103 to `train`. The test pictures are the 20
   best-scored ones of a first, smaller list (60 training pictures), and they stayed the same when the
   list grew.

Each entry records the post id, Danbooru's MD5 of the original, the MD5 of the downloaded "large"
picture, the artist, the copyright, the original source and the rating.

## Download

```bash
docker compose run --rm detector fetch                 # 123 pictures, one request per second
docker compose run --rm detector generate --seed=1     # the 123 blurred copies
```

`fetch` takes about two minutes (125 requests), and it is polite on purpose:

- It names itself in the User-Agent, with this repository's address.
- It sends at most one request per second and backs off when Danbooru answers "429 Too Many Requests".
- It stops at once if a bot check answers instead of Danbooru, instead of trying to get past it.

Pictures already in the folder are not downloaded again. Every download is checked against the MD5 in
the list, shrunk so that its longer side is at most 512 px, and saved as PNG. A post that Danbooru
deleted, changed or no longer rates general audience is skipped, and the summary lists it. `.gitignore`
keeps the downloaded files and their blurred copies out of git.

## How well the model does on them

Trained with the web demo's settings, the model gets 35 to 36 of the 40 test pictures (20 sharp, 20
blurred) right and 92 to 94% of the heat map windows. The pictures it misses are the most lightly blurred
copies (Gaussian kernel 5, lens radius 5) of finely drawn art, and two very soft paintings. So the web
demo publishes a model at 85% test accuracy or better (34 of 40). That threshold was 90% until the
project owner lowered it on 2026-11-04, because the Danbooru-trained model judges the heat map windows
far better (92% instead of 64% for the model trained on drawn scenes) while it missed 90% by one
picture in CI.

## Use your own pictures instead

Put sharp pictures (`.png`, `.jpg`, `.jpeg`) in `train/nonblur` and `test/nonblur`, and run `generate`
to make the blurred halves. Training reads every picture in `<split>/blur` and `<split>/nonblur`,
downloaded or not. Other files here, like this README, `danbooru.json` or `demo.png`, are ignored.
Without a network, `blur-detector draw` makes a data set of cartoon scenes drawn by code instead.

History: before 2026-11-04 this folder held 61 character portraits from *Blue Archive* (still in the
git history up to commit `5f698fe`). After that it held 161 drawn scenes, until commit `f1327cd`.
