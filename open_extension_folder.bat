@echo off
title ALETHEX Chrome Extension Loader Helper (v1.2 Clean)
echo ====================================================================
echo   ALETHEX Universal AI Memory Guard v1.2 (Clean Fresh Build)
echo ====================================================================
echo.
echo [QUICK INSTRUCTIONS]
echo  1. In Chrome, go to chrome://extensions
echo  2. If an older ALETHEX card exists, click "Remove"
echo  3. Click "Load unpacked" (top left)
echo  4. Paste this folder path into the "Folder:" bar and click "Select Folder":
echo     D:\Projects\Notes\Text Analytics Project\alethex-guard-v1.2
echo.
echo Copying fresh v1.2 folder path to clipboard...
powershell -Command "Set-Clipboard -Value 'D:\Projects\Notes\Text Analytics Project\alethex-guard-v1.2'"
echo Copied to clipboard!
echo.
echo Opening Windows Explorer with fresh v1.2 folder highlighted...
explorer.exe /select,"%~dp0alethex-guard-v1.2"
pause
