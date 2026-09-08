#include <windows.h>
#include <stdio.h>

int WINAPI WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nCmdShow
)
{
    char baseDir[MAX_PATH];
    char pythonPath[MAX_PATH];
    char scriptPath[MAX_PATH];
    char commandLine[MAX_PATH * 2];

    GetModuleFileNameA(
        NULL,
        baseDir,
        MAX_PATH
    );

    char *lastSlash = strrchr(
        baseDir,
        '\\'
    );

    if (lastSlash != NULL)
    {
        *lastSlash = '\0';
    }

    snprintf(
        pythonPath,
        sizeof(pythonPath),
        "%s\\runtime\\python.exe",
        baseDir
    );

    snprintf(
        scriptPath,
        sizeof(scriptPath),
        "%s\\main.py",
        baseDir
    );

    if (GetFileAttributesA(pythonPath) ==
        INVALID_FILE_ATTRIBUTES)
    {
        MessageBoxA(
            NULL,
            "Python runtime was not found.\n\n"
            "Please install the runtime package first.",
            "Offline Image Swap",
            MB_OK | MB_ICONERROR
        );

        return 1;
    }

    if (GetFileAttributesA(scriptPath) ==
        INVALID_FILE_ATTRIBUTES)
    {
        MessageBoxA(
            NULL,
            "main.py was not found.",
            "Offline Image Swap",
            MB_OK | MB_ICONERROR
        );

        return 1;
    }

    snprintf(
        commandLine,
        sizeof(commandLine),
        "\"%s\" \"%s\"",
        pythonPath,
        scriptPath
    );

    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION processInfo;

    ZeroMemory(
        &startupInfo,
        sizeof(startupInfo)
    );

    ZeroMemory(
        &processInfo,
        sizeof(processInfo)
    );

    startupInfo.cb =
        sizeof(startupInfo);

    BOOL result = CreateProcessA(
        NULL,
        commandLine,
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        baseDir,
        &startupInfo,
        &processInfo
    );

    if (!result)
    {
        MessageBoxA(
            NULL,
            "Could not start the application.",
            "Offline Image Swap",
            MB_OK | MB_ICONERROR
        );

        return 1;
    }

    CloseHandle(
        processInfo.hProcess
    );

    CloseHandle(
        processInfo.hThread
    );

    return 0;
}

