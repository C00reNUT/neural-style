@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set content_path=results\collages
set styles_path=styles

for %%f in (%content_path%\*.jpg) do (
	set ARGS=--mode crop --in %%f --styles-path %styles_path%
	echo python build_collage.py !ARGS!
	python build_collage.py !ARGS!
)
