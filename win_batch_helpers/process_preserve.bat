@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set content_path=results\preserve_colors

for %%f in (%content_path%\*.jpg) do (
	set ARGS= --mode yuv --in %%f
	echo python luma_transfer.py !ARGS!
	python luma_transfer.py !ARGS!
)
