Primary distance: Euclidean landmark 5 to object center; verified against x/y.
Calibration from this settings.json: 1.2 mm/px (None means uncalibrated; plots and primary fields are px, no assumed ratio).
Detection uses raw px: near <60, merge finite gaps <=6 frames, never cross missing intervals, lookback <=300 frames; distant closed rest: distance >80% local distant-distance 90th percentile (minimum100px) and aperture <15; Last distant closed posture before approach, then first sustained departure (distance -4px or aperture +3px); retain 10 prior frames. Flag if their distance range >8px, aperture range >4px, or aperture reaches 15px.
Closure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <50 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.
Start: 10 frames before departure; end: 10 frames after five-frame closure confirmation. Raw CSV unchanged; per-trial distance-aperture plots use distance / first recorded distance, aperture stays in mm (or px if uncalibrated); no time normalization or smoothing. Actual t_camera minus segment first timestamp.
Original CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).
All candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.
All CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.

Unflagged segments are grasp_### at top level. All flagged candidates are retained separately under review_segments/grasp_###; flags do not by themselves establish actual grasp failure. summary.json lists relative_folder for each candidate.
