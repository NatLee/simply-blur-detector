# The built-in data set

Every picture in this folder is drawn by code in this repository: cartoon scenes with skies, cities,
hills, robots, flowers, signs and badges, made of OpenCV shapes and its built-in Hershey font
([`src/detector/blur_detection/scenes.py`](../src/detector/blur_detection/scenes.py)). No photos,
no third-party artwork. The pictures are covered by the repository's [MIT licence](../LICENSE).

| Folder | Pictures | What they are |
|---|---|---|
| `train/nonblur` | 60 | sharp scenes `scene-000.png` … `scene-059.png` |
| `train/blur` | 60 | the same scenes, blurred |
| `test/nonblur` | 20 | sharp scenes `scene-060.png` … `scene-079.png`, never used for training |
| `test/blur` | 20 | the same scenes, blurred |
| `demo.png` | 1 | one more scene, the default picture of `blur-detector run` and `predict` |

Each picture is 384 to 512 pixels wide and high. Every blurred copy got one blur from
[blur-generator](https://github.com/NatLee/blur-generator), picked at random: a lens blur with a radius
of 5, 7, 9 or 11 px, a Gaussian blur with a 5 to 9 px kernel, or a motion blur 20 to 80 px long.

## Make it again

The two commands below rebuild this folder: the drawn pictures bit for bit, the blurred copies to
within one brightness level (the blurs use floating point, which a few CPUs round differently). A test
checks that they still do.

```bash
docker compose run --rm detector draw                 # the 81 sharp pictures (seed 2023)
docker compose run --rm detector generate --seed=1    # the 80 blurred copies
```

`draw --seed=N` gives a different set of scenes, and `--train` / `--test` change how many.

## Use your own pictures instead

Put sharp pictures (`.png`, `.jpg`, `.jpeg`) in `train/nonblur` and `test/nonblur`, delete the
drawn ones you do not want, and run `generate` to make the blurred halves. Training only reads
`<split>/blur` and `<split>/nonblur`, so other files here (like this README or `demo.png`) are ignored.

Until 2026-11-04 this folder held character art from the game *Blue Archive*. It was replaced
because its licence was unclear. It is still in the git history, up to commit `5f698fe`.
