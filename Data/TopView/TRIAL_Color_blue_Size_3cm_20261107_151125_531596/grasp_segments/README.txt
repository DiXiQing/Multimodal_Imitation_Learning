Primary distance: Euclidean landmark 5 to object center; verified against x/y.
Calibration from this settings.json: None mm/px (None means uncalibrated; plots and primary fields are px, no assumed ratio).
Batch-specific boundaries use aperture rise above 12 px and sustained departure of 4 px. Closure onset is retrospectively confirmed with 5 stable frames, and up to 10 subsequent hold frames are retained.
Start: 10 frames before departure; end: 10 frames after the retrospectively confirmed onset of the closure plateau. Raw CSV unchanged; per-trial distance-aperture plots use distance / first recorded distance, aperture stays in mm (or px if uncalibrated); no time normalization or smoothing. Actual t_camera minus segment first timestamp.
Original CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).
All candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.
All CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.

Trial folders use first-frame Asia/Tokyo timestamps YYYYMMDD_HHMMSS, without fractions or suffixes. Unflagged segments are at top level. All flagged candidates are retained separately under review_segments/; flags do not by themselves establish actual grasp failure. summary.json lists relative_folder for each candidate.
