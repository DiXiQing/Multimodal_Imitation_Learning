Primary distance: Euclidean landmark 5 to object center; verified against x/y.
Calibration from this settings.json: 1.5 mm/px. Approximate scale at cube top-face height.
Detection uses raw px: near <55, merge gaps <=8 frames; distant closed rest: distance >230 and aperture <15; backtrack opening to local rest baseline +3 px aperture / -4 px distance.
Closure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <45 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.
Start: 10 frames before departure; end: 10 frames after five-frame closure confirmation. No smoothing or normalization. Actual t_camera minus segment first timestamp.
Original CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).
All candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.
All CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.
