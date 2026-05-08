import os

file_path = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\ffmpeg_producer.cpp"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

import re

# resolve ffmpeg_producer
content = re.sub(
    r"<<<<<<< HEAD\s+(\{0, 2\}, pingpong, speed)\s+=======\s+(.*?, growing)\s+>>>>>>> server_local/MAV-VMX-Replay",
    r"{0, 2}, pingpong, speed, growing)",
    content,
    flags=re.MULTILINE | re.DOTALL
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

file_path_av = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\av_producer.cpp"
with open(file_path_av, "r", encoding="utf-8") as f:
    content_av = f.read()

# Replace block 1
content_av = re.sub(
    r"<<<<<<< HEAD\n(#ifdef _WIN32\n#include <d3d11\.h>\n#endif)\n=======\n(#include <chrono>\n#include <sstream>\n#include <iomanip>)\n>>>>>>> server_local/MAV-VMX-Replay",
    r"\1\n\2",
    content_av,
    flags=re.MULTILINE
)

# Replace block 2
content_av = re.sub(
    r"<<<<<<< HEAD\n(\s*std::atomic<bool>\s*pingpong_.*?)\n=======\n(\s*bool\s*growing_.*?)\n\s*std::atomic<double>\s*speed_.*?>>>>>>> server_local/MAV-VMX-Replay",
    r"\1\n\2",
    content_av,
    flags=re.MULTILINE
)

# Replace block 3
content_av = re.sub(
    r"<<<<<<< HEAD\n(\s*if \(seek != AV_NOPTS_VALUE\) \{\n\s*try \{.*?\}\n\s*\})\n=======\n\s*(if \(seek != AV_NOPTS_VALUE\) \{.*?continue;\n\s*\})\n>>>>>>> server_local/MAV-VMX-Replay",
    r"\1",
    content_av,
    flags=re.MULTILINE | re.DOTALL
)

# Replace block 4
content_av = re.sub(
    r"<<<<<<< HEAD\n(\s*if \(\(buffer_eof_\) && !frame_flush_\) \{.*?return core::draw_frame::still\(frame_\);\n\s*\})\n=======\n(\s*if \(\(buffer_eof_ \|\| growing_\) && !frame_flush_\) \{.*?return core::draw_frame::still\(frame_\);\n\s*\})\n>>>>>>> server_local/MAV-VMX-Replay",
    r"\2", # take the growing_ line along with proper end behaviour
    content_av,
    flags=re.MULTILINE | re.DOTALL
)

with open(file_path_av, "w", encoding="utf-8") as f:
    f.write(content_av)

print("Merged correctly")
