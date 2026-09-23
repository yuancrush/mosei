"""读取问题1单个样本特征的最小示例。"""
from pathlib import Path
import numpy as np

sample_path = next((Path(__file__).parent / "features").glob("*.npz"))
data = np.load(sample_path)
print("sample_id:", data["sample_id"].item())
print("word-level:", data["word_text"].shape, data["word_audio"].shape, data["word_vision"].shape)
print("aligned-50:", data["text"].shape, data["audio"].shape, data["vision"].shape)
print("durations:", data["video_duration_s"].item(), data["text_effective_duration_s"].item(),
      data["audio_effective_duration_s"].item(), data["visual_effective_duration_s"].item())
print("alignment reliability:", data["alignment_reliability"].item(),
      "usable=", data["time_mapping_usable"].item())
print("first words and intervals:")
for word, interval in zip(data["words"][:5], data["word_intervals"][:5]):
    print(str(word), interval.tolist())
