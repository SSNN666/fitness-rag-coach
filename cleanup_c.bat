@echo off
echo Cleaning C drive...

taskkill /f /im "Docker Desktop.exe" 2>nul
taskkill /f /im "com.docker.backend.exe" 2>nul
taskkill /f /im "wsl.exe" 2>nul

del /f /q "C:\Users\SSNN\AppData\Local\Docker\backend.error.json" 2>nul
del /f /q "C:\Users\SSNN\AppData\Local\Docker\backend.error.*" 2>nul

rmdir /s /q "C:\Users\SSNN\AppData\Local\Temp\claude" 2>nul

echo Done!
pause
