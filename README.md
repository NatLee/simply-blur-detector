<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLee/simply-blur-detector/main/docs/images/hero-dark.png">
    <img src="https://raw.githubusercontent.com/NatLee/simply-blur-detector/main/docs/images/hero-light.png" alt="simply-blur-detector: Is it blurry? A small Keras CNN, trained on pictures blurred with blur-generator, against the classic Laplacian variance, window by window, in your browser. Commands: docker compose up, natlee.github.io/simply-blur-detector. Flow: fetch, blur-generator, train, ONNX, browser. The picture is the web demo's heat map on Danbooru post 9999999 by tetuw, whose right half was blurred: the left half is marked sharp, the right half blurry.">
  </picture>
</p>

<p align="center">
  <a href="https://natlee.github.io/simply-blur-detector/"><img src="https://img.shields.io/badge/demo-try%20it%20in%20your%20browser-c9f24b?labelColor=0d0e11" alt="Live demo"></a>
  <a href="https://github.com/NatLee/simply-blur-detector/actions/workflows/ci.yml"><img src="https://github.com/NatLee/simply-blur-detector/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/NatLee/simply-blur-detector/actions/workflows/pages.yml"><img src="https://github.com/NatLee/simply-blur-detector/actions/workflows/pages.yml/badge.svg" alt="Pages"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776ab" alt="Python 3.10 to 3.13">
  <img src="https://img.shields.io/badge/TensorFlow-2.21%20%C2%B7%20Keras%203-ff6f00" alt="TensorFlow 2.21, Keras 3">
  <img src="https://img.shields.io/badge/docker-amd64%20%7C%20arm64-2496ed" alt="Docker image for amd64 and arm64">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/NatLee/simply-blur-detector" alt="License: MIT"></a>
</p>

A small convolutional neural network (CNN) that tells sharp pictures from blurry ones. You train it on a
folder of sharp pictures: [blur-generator](https://github.com/NatLee/blur-generator) makes a lens,
Gaussian or motion-blurred copy of each one, and the network learns the difference. The repository
comes with a list of 123 general-audience anime illustrations from [Danbooru](https://danbooru.donmai.us)
that one command downloads, or you can bring your own pictures. Everything runs in Docker on x86 and
ARM machines, and the trained model is also published as a
**[web demo](https://natlee.github.io/simply-blur-detector/)**: drop a picture and it shows the
network's verdict, a window-by-window blur heat map and the classic Laplacian variance side by side,
all in your browser.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NatLee/simply-blur-detector/main/docs/images/demo-dark.png">
    <img src="https://raw.githubusercontent.com/NatLee/simply-blur-detector/main/docs/images/demo-light.png" alt="The web demo on Danbooru post 10468701 by channel (caststation): the picture with the CNN heat map over it, the credit under it, a dial with the CNN's probability that the picture is blurry, the Laplacian variance on a log scale with its threshold, and the cards of the anime pictures." width="820">
  </picture>
</p>

## How it works

```mermaid
flowchart TB
  list["data/danbooru.json<br/>123 checked posts"] -->|"blur-detector fetch"| sharp
  own["or your own pictures"] -.-> sharp
  sharp["Sharp pictures<br/>data/*/nonblur"] -->|"blur-detector generate"| bg["blur-generator<br/>lens · Gaussian · motion"]
  bg --> blurred["Blurred copies<br/>data/*/blur"]
  sharp --> train
  blurred --> train["blur-detector train<br/>Keras CNN"]
  train --> model["model/model.h5"]
  model -->|"blur-detector predict"| figure["Comparison figure<br/>outputs/results.png"]
  model -->|"blur-detector export"| onnx["ONNX model + model card<br/>src/web/public/model"]
  onnx --> web["Web demo<br/>ONNX Runtime Web on GitHub Pages"]
```

Every picture is prepared the same way for training, prediction and the web demo: each colour
channel's histogram is equalized, the picture is resized to 256 × 256 and scaled to 0–1. The network is
three convolution + max-pooling blocks and two dense layers (about 100k weights); it answers with the
probability of "Clear" and "Blur".

## Quick start

You need [Docker](https://docs.docker.com/get-docker/) with Compose. The same image builds on x86-64
and on ARM64 (Apple silicon included).

```bash
git clone https://github.com/NatLee/simply-blur-detector.git
cd simply-blur-detector
docker compose run --rm detector fetch                # download the 123 listed pictures (about 2 min)
docker compose run --rm detector generate --seed=1    # blurred copies of them
docker compose up
```

`fetch` downloads the pictures listed in `data/danbooru.json` ([what they are, and their copyright](data/README.md)).
`docker compose up` then trains the network on them for 5 epochs, saves it to `model/model.h5` and draws
`outputs/results.png`. If `model/model.h5` already exists, training continues from it. Five epochs is a
quick first run; the web demo's model trains for longer and with crops, see [The web demo](#the-web-demo).
To run every step from scratch (download what is missing, new blurred copies, a new model, the figure and
the web demo's model):

```bash
./scripts/fast-run.sh        # Windows: scripts\fast-run.bat
```

## Commands

Inside the container the command line is `blur-detector`; with Compose, put
`docker compose run --rm detector` in front. `blur-detector <command> --help` lists every flag.

| Command | What it does |
|---|---|
| `fetch` | Downloads the Danbooru pictures listed in `data/danbooru.json` into `data/*/nonblur`: one request per second, MD5-checked, only the ones not there yet. Deleted or re-rated posts are skipped and listed. |
| `draw` | Draws a data set from code instead, for working offline: 60 training and 20 test cartoon scenes plus `data/demo.png`. `--seed` picks other scenes, `--train` / `--test` change the counts. |
| `generate` | Writes a blurred copy of every picture in `data/*/nonblur` to the `blur` folder next to it: a lens, Gaussian or motion blur of random strength (`--seed` makes it repeatable). |
| `train` | Trains on `data/{train,test}/{blur,nonblur}` and saves `model/model.h5`. `--epochs` (5), `--crops=N` (also train on N random crops of every picture, for the heat map), `--seed`, `--model_path=None` to start over. |
| `predict` | Prints P(blur) of one picture, then blurs it more and more and draws a 4 × 4 grid of predictions. `--img_path`, `--blur_mode` (`lens`, `gaussian`, `motion`), `--save_fig_path`. |
| `export` | Exports the model for the web demo: an ONNX file checked against Keras, and a model card with the accuracy and package versions. |
| `run` | Trains, saves and draws the figure in one go: the default command, with the same flags as the old `main.py` (`--train_model`, `--save_model`, `--demo`, `--demo_img_path`, `--demo_blur_mode`, ...). |

Paths are inside the container, where Compose mounts `./data` on `/data`, `./model` on `/model`,
`./outputs` on `/outputs` (the working folder) and `src/web/public` on `/web`. In Git Bash on Windows,
put `MSYS_NO_PATHCONV=1` in front of commands with such paths, otherwise Git Bash turns `/web/...` into
a Windows path (PowerShell and cmd leave them alone).

For example, ask the model about the demo picture (a cartoon scene drawn by code, `data/demo.png`):

```console
$ docker compose run --rm detector predict --img_path=/data/demo.png --blur_mode=lens --save_fig_path=/outputs/demo-lens.png
INFO | Blur Detection Model Loaded.
INFO | /data/demo.png: Clear (P(blur) = 0.000)
INFO | tile 0,0: Clear (P(blur) = 0.043)
...
INFO | Saved the comparison figure to /outputs/demo-lens.png
0.0004406454972922802
```

The last line is the return value, P(blur), so a script can use it (log lines shortened here).

<p align="center">
  <img src="docs/images/demo-lens.png" alt="A 4 by 4 grid of the demo picture with a lens blur that grows from the top left to the bottom right; each tile is labelled Clear or Blur with the probability of blur" width="440">
</p>

## Train on your own pictures

Put sharp pictures (`.png`, `.jpg`, `.jpeg`) in `data/train/nonblur` and `data/test/nonblur` (with or
without the downloaded ones), then let blur-generator make the blurry half and train:

```text
data
├── train
│   ├── nonblur   your sharp pictures
│   └── blur      written by `generate`
└── test
    ├── nonblur
    └── blur
```

```bash
docker compose run --rm detector generate
docker compose run --rm detector train --model_path=None --epochs=20 --crops=2
docker compose run --rm detector export
```

Files elsewhere under `data/` are ignored, so `data/demo.png` (the default picture of `run` and
`predict`, drawn by code), `data/danbooru.json` and `data/README.md` can sit next to the folders.

## Use it from Python

```python
from blur_detection import DetectionModel, get_data, prepare_image, predict_blur_img
from blur_detection.images import read_image

detector = DetectionModel()                      # untrained; DetectionModel('/model/model.h5') loads one
train_ds, test_ds = get_data('/data')            # tf.data datasets of prepared pictures
detector.train(train_ds, test_ds, epochs=5)
detector.save('/model/model.h5')                 # .h5 or .keras

probabilities, labels = detector.predict(prepare_image(read_image('/data/demo.png')))
print(labels[0], float(probabilities[0][1]))     # e.g. "Clear 7.65e-07" after this short training

predict_blur_img(detector, '/data/demo.png', blur_mode='lens', save_fig_path='/outputs/results.png')
```

`DetectionModel` also loads `.h5` weight files saved by the original TensorFlow 2.7 version of this
project.

## The web demo

The demo runs the same network with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), so
your pictures never leave the browser. It opens with one of five anime pictures from
[Danbooru](https://danbooru.donmai.us) and shows:

- the CNN's P(blur) for the whole picture, and the Laplacian variance (a blur score that needs no
  training: low variance means few sharp edges) with an adjustable threshold;
- a heat map from either detector over square windows half as wide as the picture's shorter side
  (the smallest crop the network saw in training);
- a stress test that blurs the picture step by step and plots how both scores react.

About the Danbooru pictures on the page:

- **Picked and checked by hand.** The page shows only the five general-audience, square posts in
  [`samples/danbooru.json`](src/web/public/samples/danbooru.json), each one looked at by a person before
  it was listed (see [Image credits](#image-credits)). They are not among the training pictures. The page
  asks Danbooru for the post, checks again that it is rated general (`rating:g`) and skips to the next
  listed one if it is gone.
- **Loaded at run time, always credited.** Your browser loads the picture straight from Danbooru's image
  server. The credit under it names the artist and links to the Danbooru post and the original
  source. Nothing of these pictures is stored in this repository or on the site.
- **Polite.** It makes one API request and one picture request per picture, and the current picture
  stays for the browser session.
- **Your own picture when Danbooru is away.** The page has no pictures of its own. When Danbooru cannot be
  reached, or the list cannot be read, it says why and asks you to drop, paste or choose a picture; the
  detectors, the heat map and the stress test all work with it. (Automated browsers do not get Danbooru's
  CORS headers, so they always see this.)

GitHub Actions builds it ([`pages.yml`](.github/workflows/pages.yml)): it downloads the listed pictures
(cached between runs, so Danbooru is asked again only when the list changes), trains the network on them
(EPOCHS, CROPS and SEED at the top of the workflow), refuses to publish a model below 90% test accuracy,
exports it, checks in Node.js that the page's JavaScript gives the same P(blur) as Python for
`data/demo.png`, runs the page in Chrome with Danbooru answered by the test, and deploys
`src/web/public`. The model card at the bottom of the page links to the run that trained it.

To try the page on your own machine, export a model and serve the folder:

```bash
docker compose run --rm detector export
docker compose run --rm -p 8000:8000 --entrypoint python detector -m http.server 8000 -d /web
# open http://localhost:8000
```

## GPU

On a Linux machine with an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), the
`detector-gpu` service builds the image with TensorFlow's CUDA libraries and hands the GPU to it:

```bash
docker compose run --rm detector-gpu train
```

## Development

```bash
pip install -e "src/detector[test]"      # Python 3.10-3.13
cd src/detector && pytest                # data, model, figure, command line, ONNX export
cd src/web && npm ci && npm test         # OpenCV ports, heat map, browser pipeline (needs an exported model)
```

Or in Docker: `docker build --build-arg EXTRAS=test -t simply-blur-detector:test .`, then run
`pytest` in it with `src` mounted. CI runs both suites, and builds and runs the Docker image on amd64
and arm64.

The project is laid out as `src/detector` (the Python package `blur_detection` and its tests) and
`src/web` (the demo page in `public/`, its tests and the tool that makes its OpenCV test fixture).

## The training pictures

The training pictures are 123 general-audience anime illustrations from Danbooru: 103 for training and 20
for testing. They were chosen from Danbooru's best-scored `rating:g` posts and filtered by tags. Then a
person looked at every one of the 356 candidates and kept 128: these 123 and the five of the web demo. [`data/README.md`](data/README.md) explains
how, and `data/danbooru.json` lists them with their artists and sources.

**Their copyright belongs to their artists.** This repository only stores the list and does not
distribute the pictures: `fetch` downloads them from Danbooru when you need them. Models trained on them
are for demonstrating this project only. The MIT licence of this repository covers the code, not these
pictures.

History: until November 2026 the folder held character art from *Blue Archive* (licence unclear), then
for a while cartoon scenes drawn by code. `blur-detector draw` still makes those for offline use.

## Image credits

The cover at the top, the demo screenshot below it and the anime cards of the web demo show these
Danbooru posts, all rated general audience. The cover and the screenshots are in this repository
(`docs/images/`); the demo loads its five pictures from Danbooru when you open it.

| Where | Post | Artist | Size | Copyright |
|---|---|---|---|---|
| cover, social preview (right half blurred by blur-generator) | [#9999999](https://danbooru.donmai.us/posts/9999999) | tetuw | 1600 × 1600 | arknights |
| demo: Window seat, and the demo screenshot | [#10468701](https://danbooru.donmai.us/posts/10468701) | channel_(caststation) | 2484 × 2484 | vocaloid |
| demo: Heart glasses | [#6446709](https://danbooru.donmai.us/posts/6446709) | gya_(144) | 1000 × 1000 | genshin_impact |
| demo: Boxing match | [#5045433](https://danbooru.donmai.us/posts/5045433) | hakuzumu | 2500 × 2500 | genshin_impact |
| demo: Small talk | [#5260352](https://danbooru.donmai.us/posts/5260352) | omniformblue | 1600 × 1600 | persona_4 |
| demo: Corner store | [#6648778](https://danbooru.donmai.us/posts/6648778) | kaisunstreehole | 2048 × 2010 | baldur's_gate_3 |

The pictures belong to their artists and are shown with credit, for demonstration only; the MIT licence
does not cover them. `data/demo.png` and the comparison figure above are drawn by code in this
repository.

## License

[MIT](LICENSE) © 2023 Nat Lee, for the code and the pictures it draws. The Danbooru pictures are not
covered (see [The training pictures](#the-training-pictures) and [Image credits](#image-credits)).

## Author

Nat Lee
