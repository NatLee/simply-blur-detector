# The training pictures

The pictures here are this project's own original illustrations, under the repository's
[MIT licence](../LICENSE) like the code. They are drawn in an anime style and show all kinds of
things: people at work and at play, landscapes, rooms, food, animals, towns, machines, plants and night
scenes. The line work ranges from thick outlines with flat colours to very fine, detailed drawing, because
fine lines are the hard case for a blur detector.

| | Pictures | What they are |
|---|---|---|
| `train/nonblur` | 100 | sharp pictures for training, `picture-001.jpg` to `picture-100.jpg` |
| `test/nonblur` | 20 | sharp pictures kept out of training, `picture-101.jpg` to `picture-120.jpg` |
| `train/blur`, `test/blur` | 100 + 20 | a blurred copy of each, made by `generate --seed=1` |
| `demo.png` | 1 | the default picture of `run` and `predict`; also one of the web demo's pictures |

The training and test pictures are JPEGs (quality 90) whose longer side is 512 pixels, the size that the
blur strengths of `generate` were chosen for. Three in five are square, the rest landscape (3 : 2) or
portrait (2 : 3). The test pictures were set aside before training, one in every six, so they cover the
same range of subjects and styles.

[`pictures.json`](pictures.json) lists every picture of the project (these, the five of the web demo and
the one on the cover) with what it is for, its size and the SHA-256 of the file. A test checks that the
files and the list agree, and that nothing unlisted sits in the training folders.

## Make the blurred copies again

```bash
docker compose run --rm detector generate --seed=1
```

The blurred copies are in the repository already. `generate` gives every sharp picture a lens, Gaussian
or motion blur of random strength; with `--seed=1` it repeats the committed ones (on another CPU a few
pixels may differ by one).

## How well the model does on them

Trained with the web demo's settings (20 epochs, 2 crops, seed 1), the model got 39 of the 40 test
pictures (20 sharp, 20 blurred) and 568 of their 600 heat map windows (94.7%) right in a local run. The
model card at the bottom of the web demo shows the numbers of the run that trained the published model,
which is published only at 85% test accuracy or better (34 of 40).

Models trained on the earlier data sets, on the same 40 test pictures:

| Model trained on | Test pictures | Heat map windows |
|---|---|---|
| these pictures | 39 / 40 (97.5%) | 568 / 600 (94.7%) |
| the anime pictures from Danbooru used before (the published model of 2026-11-04) | 35 / 40 (87.5%) | 593 / 600 (98.8%) |
| cartoon scenes drawn by code (`draw`) | 34 / 40 (85.0%) | 410 / 600 (68.3%) |

## Use your own pictures instead

Put sharp pictures (`.png`, `.jpg`, `.jpeg`) in `train/nonblur` and `test/nonblur`, and run `generate`
to make the blurred halves. Training reads every picture directly in `<split>/blur` and `<split>/nonblur`;
other files here, like this README, `pictures.json` or `demo.png`, and other folders are ignored.

`blur-detector draw` makes a data set of cartoon scenes drawn by code, which needs nothing but the
package. It also writes a `demo.png`, so give it a folder of its own, for example
`draw --data_path=/data/drawn` and then `train --data_path=/data/drawn`.

History: until 2026-11-04 this folder held 61 character portraits from *Blue Archive* (still in the git
history up to commit `5f698fe`), then 161 scenes drawn by code (up to `f1327cd`), then a list of anime
pictures from Danbooru that a `fetch` command downloaded (up to `9eca321`).
