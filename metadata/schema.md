# Metadata schema

`train_pairs.csv` and `test_pairs.csv` describe the final benchmark split. `sample_id` is zero-based within each split and is the join key for predictions; use `split` plus `sample_id` if combining the files. There are 5,105 training rows from 408 sites and 2,015 test rows from 183 sites. The splits have no site overlap. These are metadata only: each image path names a file held outside this repository.

| Field | Meaning |
| --- | --- |
| `sample_id`, `split` | Row number within the released split; train or test |
| `site_id`, `location_index` | Site identifier and prior-state image location index |
| `before`, `after` | Relative references to restricted prior- and designed-state Street View crops |
| `control_edge`, `control_depth`, `control_segmentation`, `control_scribble` | Relative references to image-derived condition maps, also restricted |
| `text` | Image-level description used in the study |
| `before_capture_month_year`, `after_capture_month_year` | Capture month and year |
| `before_pano_id`, `after_pano_id` | Source panorama identifiers |
| `before_lat`, `before_lon`, `after_lat`, `after_lon` | Source panorama coordinates |
| `*_heading`, `*_pitch`, `*_roll`, `*_capture_time` | Raw panorama capture metadata |

The capture fields describe the source panoramas. Heading, pitch, and roll are raw panorama values, not the orientation of the rendered crop. All before and after panorama identifiers in the released split resolved to capture metadata.

The image paths and identifiers are released as study metadata. Their inclusion does not grant permission to redistribute the associated imagery. See `DATA_AVAILABILITY.md`.
