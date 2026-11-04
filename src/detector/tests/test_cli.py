import inspect
import subprocess
import sys

import pytest

from blur_detection import cli
from blur_detection.images import read_image, write_image

from .helpers import sharp_image

#: Flags and defaults of the original main.py execute(); `run` must keep every one of them.
ORIGINAL_MAIN_FLAGS = {
    'model_path': '/model/model.h5',
    'model_save_path': '/model/model.h5',
    'data_path': '/data',
    'train_model': True,
    'epochs': 5,
    'save_model': True,
    'demo': True,
    # Was /data/Mika.png. The user had the game artwork removed from data/ on 2026-11-04, and the
    # drawn demo picture took its place.
    'demo_img_path': '/data/demo.png',
    'demo_blur_mode': 'lens',
}


def defaults(func):
    return {name: p.default for name, p in inspect.signature(func).parameters.items()}


def test_run_keeps_the_flags_and_defaults_of_main_py():
    flags = defaults(cli.run)
    assert {name: flags[name] for name in ORIGINAL_MAIN_FLAGS} == ORIGINAL_MAIN_FLAGS
    assert set(flags) - set(ORIGINAL_MAIN_FLAGS) == {'seed'}  # the only new flag, off by default
    assert flags['seed'] is None


def test_every_step_has_a_command():
    assert list(cli.COMMANDS) == ['fetch', 'draw', 'generate', 'train', 'predict', 'export', 'run']


@pytest.mark.parametrize('args', [['--help'], ['train', '--help']])
def test_help_works_from_the_command_line(args):
    out = subprocess.run([sys.executable, '-m', 'blur_detection', *args], capture_output=True, text=True,
                         env={'PAGER': 'cat', 'TERM': 'dumb', 'PATH': ''}, timeout=300)
    assert out.returncode == 0, out.stderr
    assert 'train' in out.stdout + out.stderr  # fire prints help to stderr


def test_generate_train_predict_from_end_to_end(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / 'model' / 'model.h5'
    cli.draw(data_path=str(data_dir), train=2, test=1, seed=3)  # adds drawn pictures next to the fixture's
    cli.generate(data_path=str(data_dir), seed=1)
    cli.train(data_path=str(data_dir), model_path=None, model_save_path=str(model), epochs=1, seed=1)
    assert model.is_file()

    demo = write_image(tmp_path / 'demo.png', sharp_image(9, size=(90, 90)))
    p_blur = cli.predict(img_path=str(demo), model_path=str(model), blur_mode='gaussian',
                         save_fig_path=str(tmp_path / 'figure.png'))
    assert 0 <= p_blur <= 1
    assert read_image(tmp_path / 'figure.png') is not None


def test_run_trains_saves_and_writes_results_png_in_the_working_folder(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / 'model.h5'
    demo = write_image(tmp_path / 'demo.png', sharp_image(9, size=(90, 90)))
    cli.run(model_path=str(model), model_save_path=str(model), data_path=str(data_dir), epochs=1,
            demo_img_path=str(demo), demo_blur_mode='gaussian', seed=2)
    assert model.is_file()
    assert (tmp_path / 'results.png').is_file()
