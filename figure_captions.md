# Publication Figure Captions

Figure 1. DSF-MAC experimental workflow from EEG recordings and seizure annotations to eligibility audit, patient-specific seizure-level leave-one-out cross-validation, validation-only alarm-policy selection, and event-level testing.

Figure 2. Temporal labeling protocol. The main protocol uses a 30 min seizure occurrence period (SOP) and a 5 min seizure prediction horizon (SPH). Alarms inside the SPH are not counted as valid pre-warning alarms.

Figure 3. DSF-MAC model framework. Multi-scale 5 s, 7 s, and 9 s EEG windows are scored by the DSF risk estimator, fused using validation-selected weights, and converted to event-level alarms by a causal alarm controller.

Figure 4. Illustrative causal alarm sequence. Window risk scores are accumulated into causal evidence; alarms in remote interictal time count as false alarms, while the first valid preictal alarm counts as the event hit and determines warning time.

Figure 5. Validation event sensitivity versus FAR/h operating curves for CHB-MIT and SeizeIT2. Transparent points denote validation-only candidate thresholds and alarm-duration settings; solid lines show Pareto frontiers; black stars mark the final held-out test operating points selected without using test labels.

Figure 6. Performance-efficiency comparison on the same 170 CHB-MIT held-out seizure folds. Panel A plots parameter count against event sensitivity, and Panel B plots parameter count against mean fold FAR/h. Dashed lines connect the prespecified progressive stages; DSF-MAC is shown as the sensitivity-oriented reference. Stage 3 approaches DSF-MAC sensitivity with fewer parameters and a lower FAR/h point estimate, but paired differences are not statistically significant.
