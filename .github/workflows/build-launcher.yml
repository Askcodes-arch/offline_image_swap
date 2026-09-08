name: Build Launcher

on:
  workflow_dispatch:
  push:
    paths:
      - "launcher/launcher.c"
      - ".github/workflows/build-launcher.yml"
    branches:
      - main

jobs:
  build:
    runs-on: windows-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build launcher
        shell: powershell
        run: |
          gcc launcher\launcher.c `
            -o OfflineImageSwap.exe `
            -mwindows `
            -s

      - name: Verify launcher
        shell: powershell
        run: |
          if (!(Test-Path "OfflineImageSwap.exe")) {
              Write-Error "Launcher EXE was not created."
              exit 1
          }

          Write-Host "Launcher successfully created."

      - name: Upload launcher
        uses: actions/upload-artifact@v4
        with:
          name: OfflineImageSwap-Launcher
          path: OfflineImageSwap.exe
          if-no-files-found: error
