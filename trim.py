from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 0
end_time = 12

ffmpeg_extract_subclip("video0.avi", start_time, end_time, targetname="test0.avi")
ffmpeg_extract_subclip("video1.avi", start_time, end_time, targetname="test1.avi")
ffmpeg_extract_subclip("video2.avi", start_time, end_time, targetname="test2.avi")
