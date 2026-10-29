Primary distance: Euclidean landmark 5 to object center; verified against x/y.
Calibration from this settings.json: 1.4285714285714286 mm/px. Approximate scale at cube top-face height.
Detection excludes near events shorter than 15 frames from confirmed grasp exports; they are listed separately without deleting source data. Detection uses raw px: near <60, merge gaps <=8 frames; distant closed rest: distance >240 and aperture <15; backtrack opening to local rest baseline +3 px aperture / -4 px distance, then backtrack until pre-action 10-frame distance range <=4 px and aperture range <=3 px.
Closure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <50 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.
Start: 10 frames before departure; end: 10 frames after five-frame closure confirmation. No smoothing or normalization. Actual t_camera minus segment first timestamp.
Original CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).
All candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.
All CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.

Exceptions: segment 26 has no confirmed closure/10-frame hold due to missing detection; retained for review, not a confirmed complete grasp. Eight flagged segments: 13,20,26,27,29,31,33,35. See missing_or_suspicious.csv and detection_review.json. Calibration is copied from this recording (1.4285714285714286 mm/px); no scale inferred from the folder name or object size.
