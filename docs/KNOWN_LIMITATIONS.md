# Known Limitations

This list describes current boundaries and evidence gaps; it is not a claim that model performance or resource usage is acceptable for every classroom or device.

- **No real-world identity:** A detection is a behavior-class bounding box in one frame. A history row is a tracker's assigned behavior record in a processed frame. Tracker IDs are per-video association labels and are not verified unique students or identities across sessions.
- **Model quality is context-dependent:** Lighting, camera angle, occlusion, classroom layout, and visual similarity among behaviors can affect detections. The software does not promise a measured accuracy level. No retraining or threshold changes are made by normal app operation.
- **Memory ceiling is not established:** The approved long-video run measured sampled process-tree RSS and available GPU memory, but no numerical acceptance threshold has been approved. Do not claim bounded memory from those samples.
- **Hardware variability:** Camera indices, Windows camera permissions, DirectShow availability, GPU memory, codecs, and encoder support vary by host. Mocked tests do not establish physical-device behavior.
- **Video codec and audio:** Accepted filename extensions do not guarantee decoder compatibility. OpenCV must be able to decode the content. Annotated output is video-only and does not preserve source audio. H.264 playback depends on available encoders/decoders.
- **Local storage and retention:** Uploaded source video, annotated output, CSVs, and reports remain in unique run folders until an operator removes them outside the app. Uploads can consume substantial disk space; the app does not enforce a total storage quota or automatic retention schedule.
- **Local-only security model:** Flask binds to `127.0.0.1` and validates Host, Origin, and session CSRF tokens for mutations. It has no account system or remote-user authorization and is not intended for LAN/public exposure.
- **MediaPipe experiments are separate:** `feature_extraction.py` reuses `src/features.py`, while `features_test.py` remains a standalone visualization. None is integrated into the YOLO behavior application.
- **Clean-clone setup is not verified:** The documented tests were run in the project's existing Windows Python 3.12 environment with local model/data assets. A new installation on another operating system or CPU-only machine has not been verified here.
- **Version metadata:** `src.__version__` reads the root `VERSION` file. If that file is absent in a source checkout, the package reports `0+unknown`; release packaging must include the root version file.
- **License metadata:** No project license is declared in the inspected repository. Do not assume reuse terms.
