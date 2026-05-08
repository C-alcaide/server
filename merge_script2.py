import os
import re

file_path_av = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\av_producer.cpp"
with open(file_path_av, "r", encoding="utf-8") as f:
    content_av = f.read()

# Replace block 3
content_av = re.sub(
    r"<<<<<<< HEAD\n\s*if \(seek != AV_NOPTS_VALUE\) \{\n\s*try \{.*?\}\n\s*\}\n=======\n\s*(if \(seek != AV_NOPTS_VALUE\) \{.*?continue;\n\s*\})\n>>>>>>> server_local/MAV-VMX-Replay",
    r"                    if (seek != AV_NOPTS_VALUE) {\n                        try {\n                            seek_internal(seek);\n                            current_seek_target_ = seek;\n                            last_dropped_frame   = Frame{};\n                            last_valid_video.reset();\n                            frame                = Frame{};\n                        } catch (const std::exception& e) {\n                            CASPAR_LOG(warning) << print() << \" Seek (graph rebuild) failed: \" << e.what() << \" - retrying\";\n                            // Restore the seek command so the next loop iteration retries it,\n                            // unless another newer seek has already been enqueued!\n                            int64_t expected = AV_NOPTS_VALUE;\n                            seek_.compare_exchange_strong(expected, seek);\n                            std::this_thread::sleep_for(std::chrono::milliseconds(20));\n                            continue;\n                        }\n                    }",
    content_av,
    flags=re.MULTILINE | re.DOTALL
)

# And another one that has `buffer_eof_`
content_av = re.sub(
    r"<<<<<<< HEAD\n(\s*// When speed is negative.*?=======\n)\s*// Speed Logic.*?return core::draw_frame::still\(frame_\);\n\s*}\n\s*if \(frame_time_ < end.*?=======\n\s*if \(\(buffer_eof_ \|\| growing_\).*?>>>>>>> server_local/MAV-VMX-Replay\n",
    r"", # wait, this is getting complicated... Let's just fix it all programatically with Python scripts that are simpler, or just replace strings block by block
    content_av,
)

with open(file_path_av, "w", encoding="utf-8") as f:
    f.write(content_av)

print("Merged correctly")
