import sys

def remove_tests(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # split at standalone text
    split_str = '# ─── Standalone test'
    if split_str in content:
        content = content.split(split_str)[0]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

remove_tests(r'D:\Github\CasparVP\src\modules\cuda_prores\CMakeLists.txt')
remove_tests(r'D:\Github\CasparVP\src\modules\cuda_notchlc\CMakeLists.txt')
