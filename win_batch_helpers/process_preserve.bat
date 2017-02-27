@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set stylized_path=results\preserve_colors
set content_path=contents

for %%f in (%stylized_path%\*.jpg) do (
	set ARGS= --mode yuv --in %%f --content-path %content_path%\
	echo python luma_transfer.py !ARGS!
	python luma_transfer.py !ARGS!
)
