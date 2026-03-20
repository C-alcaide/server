"""
run_build.py — CasparVP incremental build helper.

Usage:
    python run_build.py [target ...]

    Targets (space-separated, default: ffmpeg decklink):
        ffmpeg      rebuild ffmpeg module
        decklink    rebuild decklink module
        casparcg    rebuild + link the full server executable
        all         rebuild everything

The CORRECT way to run cmake with MSVC on this machine is to invoke everything
inside a single cmd.exe session that starts with vcvars64.bat.  Capturing env
vars via subprocess and re-injecting them does NOT work because vcvars64.bat is
a stub that calls vcvarsall.bat which sets INCLUDE/LIB through nested calls that
subprocess.run(..., shell=True) cannot see.

See BUILDING_WORKFLOW.md for the full workflow documentation.
"""

import subprocess
import sys
import os
import pathlib

VCVARS  = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
BUILD   = r'd:\Github\CasparVP\build'
LOG     = r'd:\Github\CasparVP\build_out.txt'

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['ffmpeg', 'decklink']
    target_args = ' '.join(f'--target {t}' for t in targets)

    # Build the cmake command
    cmake_cmd = f'cmake --build "{BUILD}" {target_args}'

    # Chain inside a single cmd session so vcvars64 environment is active.
    # Use 'call' (not 'cmd /c') because shell=True already spawns cmd.exe /c;
    # using 'cmd /c' would create a child cmd that sets env vars and then exits
    # before cmake runs, so the INCLUDE/LIB vars would be lost.
    full_cmd = f'call "{VCVARS}" && {cmake_cmd}'

    print(f'Building targets: {targets}')
    print(f'Command: {cmake_cmd}')
    print('-' * 60)

    proc = subprocess.run(full_cmd, capture_output=True, text=True, shell=True)
    output = proc.stdout + proc.stderr

    # Save log
    pathlib.Path(LOG).write_text(output + f'\nEXIT_CODE={proc.returncode}\n', encoding='utf-8')

    # Print last 60 lines
    lines = output.splitlines()
    print('\n'.join(lines[-60:]))
    print('-' * 60)
    print(f'Exit code: {proc.returncode}  |  Log: {LOG}')

    if proc.returncode != 0:
        sys.exit(proc.returncode)

if __name__ == '__main__':
    main()
