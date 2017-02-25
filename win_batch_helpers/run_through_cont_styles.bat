@echo off
REM this is needed to enable variable expansion within the FOR loop
REM delayed expansions should be used as !ARGS! instead of %ARGS%
SETLOCAL ENABLEDELAYEDEXPANSION

set content_list=content01.jpg content02.jpg
set styles_list=style01.jpg style02.jpg

set total_count=0
for %%a in (%content_list%) do (
    for %%b in (%styles_list%) do (
	set /a total_count+=1
	)
)

set count=0
for %%a in (%content_list%) do (
    for %%b in (%styles_list%) do (
		set /a count+=1
		TITLE !count!/!total_count! %%a %%b
	
		set ARGS=--content %%a --styles %%b
		echo python neural_style.py !ARGS!
		REM python neural_style.py !ARGS!
	)
)

TITLE Processing finished!
