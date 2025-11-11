@echo off
setlocal
set "REPO_DIR=%~dp0"
set "PYTHONPATH=%REPO_DIR%;%REPO_DIR%src;%REPO_DIR%scripts;%PYTHONPATH%"
python -m pytest %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

