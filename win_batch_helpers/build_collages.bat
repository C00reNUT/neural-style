@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set stylized_path=results\collages
set styles_path=styles
set content_path=contents

for %%f in (%stylized_path%\*.jpg) do (
	set ARGS=--mode crop --in %%f --styles-path %styles_path%\ --content-path %content_path%\
	echo python build_collage.py !ARGS!
	python build_collage.py !ARGS!
)
