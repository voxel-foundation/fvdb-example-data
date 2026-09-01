# fvdb-example-data
Example data that complements the examples, demos and lessons in fVDB

## Contents

- `meshes/` — individual triangle meshes (Stanford scans, car meshes) used across the fVDB examples.
- `meshes/gso_shoes/` — the 254-model "Shoe" category of [Scanned Objects by Google Research](https://app.gazebosim.org/GoogleResearch),
  decimated to a 10k-face budget for voxelization workloads (generative examples: shape VAE / completion).
  Used under [CC-BY 4.0](http://creativecommons.org/licenses/by/4.0/); per-model attribution, source URLs,
  and face counts are recorded in `meshes/gso_shoes/ATTRIBUTION.json`. The subset is reproducible with
  `tools/curate_gso_shoes.py`.
