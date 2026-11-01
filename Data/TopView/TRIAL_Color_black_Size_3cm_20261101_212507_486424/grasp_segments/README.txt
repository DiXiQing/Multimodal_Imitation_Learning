Primary distance: Euclidean landmark 5 to object center; verified against x/y.
Calibration from this settings.json: 1.2 mm/px (None means uncalibrated; plots and primary fields are px, no assumed ratio).
Detection uses raw px: near <65, merge finite gaps <=6 frames, never cross missing intervals, lookback <=300 frames; distant closed rest: distance >80% local distant-distance 90th percentile (minimum100px) and aperture <15; Persistent onset: quiet local 10-frame baseline; 3/5 frames depart by distance -6px or aperture +5px; reject if returning within baseline distance 4px/aperture +3px for 8 frames before main opening peak; backtrack up to6 frames for smaller 2.5px departure, retain10 prior frames. No fixed-duration trimming. Flag if their distance range >8px, aperture range >4px, or aperture reaches 15px.
Closure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <58 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.
Start: 10 frames before departure; end: 10 frames after five-frame closure confirmation. Raw measurements unchanged; per-trial distance-aperture plots use initial-distance normalization, aperture remains in recorded units; no time normalization or smoothing. Actual t_camera minus segment first timestamp.
Original CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).
All candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.
All CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.

Unflagged segments are grasp_### at top level. All flagged candidates are retained separately under review_segments/grasp_###; flags do not by themselves establish actual grasp failure. summary.json lists relative_folder for each candidate.
