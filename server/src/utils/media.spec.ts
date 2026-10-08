import { describe, expect, it } from 'vitest';
import { defaults } from 'src/dtos/config.dto.js';
import { VideoFrameConfig, getVideoFrameTimestamps } from 'src/utils/media.js';
import { probeStub } from 'test/fixtures/media.stub.js';

describe('getVideoFrameTimestamps', () => {
  it('should return nothing if sampling is disabled', () => {
    expect(getVideoFrameTimestamps(10_000, 0)).toEqual([]);
  });

  it('should return nothing for a video without duration', () => {
    expect(getVideoFrameTimestamps(0, 10)).toEqual([]);
  });

  it('should sample the middle of each section', () => {
    expect(getVideoFrameTimestamps(10_000, 25)).toEqual([1250, 3750, 6250, 8750]);
  });

  it('should round up the number of sections to cover the whole video', () => {
    expect(getVideoFrameTimestamps(12_000, 30)).toEqual([1500, 4500, 7500, 10_500]);
  });

  it('should sample a single frame in the middle for 100 percent', () => {
    expect(getVideoFrameTimestamps(10_000, 100)).toEqual([5000]);
  });
});

describe(VideoFrameConfig.name, () => {
  it('should seek to the frame and write a single jpeg to a stream', () => {
    const { videoStream, format } = probeStub.videoStream2160p;
    const config = VideoFrameConfig.create({ ...defaults.ffmpeg, targetResolution: '1440' });

    const command = config.getFrameCommand(videoStream, 61_234, format);

    expect(command.inputOptions).toEqual(['-ss', '61.234', '-sws_flags', 'accurate_rnd+full_chroma_int']);
    expect(command.outputOptions).toEqual(
      expect.arrayContaining(['-frames:v', '1', '-f', 'image2pipe', '-c:v', 'mjpeg', '-q:v', '2']),
    );
    expect(command.outputOptions).not.toContain('-skip_frame');
    const filters = command.outputOptions[command.outputOptions.indexOf('-vf') + 1];
    expect(filters).toContain('scale=-2:1440');
    expect(filters).not.toContain('thumbnail');
    expect(command.twoPass).toBe(false);
  });
});
