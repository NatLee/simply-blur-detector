@echo off
:: Run every step from scratch, all in Docker: new blurred images, a freshly trained model,
:: the comparison figure and the model of the web demo.
setlocal
cd /d "%~dp0.."
if exist model\model.h5 del model\model.h5
docker compose build || exit /b 1
docker compose run --rm detector generate || exit /b 1
docker compose run --rm detector run || exit /b 1
docker compose run --rm detector export || exit /b 1
echo Done: model\model.h5, outputs\results.png and src\web\public\model\
