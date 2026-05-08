vcvars = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmake  = r'C:\Program Files\CMake\bin\cmake.exe'
build  = r'd:\Github\CasparVP\build'

lines = [
    'call "%s"' % vcvars,
    '"%s" --build %s --target casparcg' % (cmake, build),
]
with open(r'D:\Github\CasparVP\build_now.bat', 'w', encoding='ascii') as f:
    f.write('\r\n'.join(lines) + '\r\n')
print('build_now.bat rewritten OK')
