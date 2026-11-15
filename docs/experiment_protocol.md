# PnAL manuscript experiment protocol

## Main experiments

The corrected manuscript reports eight testbed and agent-count configurations. Main comparisons include PnAL, IPPO, MAPPO, H-QMIX, and CS-MADDPG. Each reported mean and sample standard deviation is based on training seeds 0–2. Test outcomes use 200 common test episodes per seed. Main outcomes are task success, collision incidence, off-road incidence, timeout incidence, and return.

Use `data/paper/main_comparisons.csv` for the corrected Table 3 and Table 4 summaries. `data/paper/controlled_pathways.csv` maps PnAL and IPPO baselines, IPPO+Shield, and IPPO+Full to the controlled comparisons. `data/paper/frozen_component_ablations.csv` records frozen-policy online-component interventions; these should be kept separate from retrained component-configuration ablations.

## Configuration mapping

The exact main-run settings are captured in `data/paper/run_protocol_map.csv` and in the runner's `SETTINGS`. For `delta_temp_rule="paper_max"`, an admitted template can contribute its stored threshold via `max(delta_t, delta_temp)`. For `"none"`, its optional threshold contribution is zero. The separate `delta_temp_when_matched` flag controls whether matched but rejected templates also contribute that threshold. The existing results use their saved run configurations without substituting a different protocol.

The `pnal_wo_semantic` preset sets `use_semantic=False` and `use_calibration=False`: semantic queries and template-threshold contributions are both disabled. The `pnal_wo_semantic_wo_shield` preset also disables SafetyShield while retaining coordination. The code includes this factorial condition, but no numerical result for it is listed among the supplied completed experiments.

## Result handling

All included CSV values are transcriptions of the supplied corrected manuscript or completed supplementary summaries. The `frozen_component_ablations.csv` file uses the completed three-seed Roundabout N=4 summary (45.0% success, 49.5% collisions) for `no_shield`, superseding the earlier two-seed interim figure.

Reconstruction of means, standard deviations, and per-episode tests from raw observations requires the original checkpoint, configuration, and episode outcome files. They are held in the author's source run folders and are not represented by the manuscript-level CSVs.

## Saved run directories

- `pnal_results_table3_intersection`: Intersection N=2,3,5 main PnAL runs.
- `pnal_results_v1`: archived Intersection N=4 attributed PnAL runs.
- `pnal_results_table4`: Roundabout and Highway main runs.
- `pnal_results_roundabout_attribution`: completed Roundabout frozen-policy comparisons.
- `pnal_results_supplement`: supplemental evaluation tables.

Calibration, validation, training, and test use separately reserved environment-seed ranges, as defined by `SeedConfig`. Reported threshold sweeps use 100 validation episodes per training seed.
