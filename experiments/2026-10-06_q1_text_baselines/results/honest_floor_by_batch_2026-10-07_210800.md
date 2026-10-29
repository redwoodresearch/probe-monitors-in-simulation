# Honest real-vs-real controls by machine batch (2026-10-07_210800)

Run AUROC, second real honest run vs first, 95% main-task bootstrap. All pairs: 258 tasks (pilot: fewer); same batch: 163 overlap800/overlap800; mixed: 95 base200/overlap800.

| set | split | level | outputs | control | all pairs (as reported) | same batch | mixed batch |
|---|---|---|---|---|---|---|---|
| own output | main-task folds | 0 | all | null_e1_vs_e2 | 0.595 [0.559, 0.633] | 0.468 [0.429, 0.508] | 0.787 [0.735, 0.840] |
| own output | main-task folds | 0 | all | scored_by_real_vs_sim | 0.504 [0.473, 0.532] | 0.519 [0.480, 0.559] | 0.480 [0.435, 0.527] |
| own output | main-task folds | 0 | no_editor | null_e1_vs_e2 | 0.640 [0.607, 0.679] | 0.502 [0.464, 0.535] | 0.861 [0.816, 0.906] |
| own output | main-task folds | 0 | no_editor | scored_by_real_vs_sim | 0.539 [0.499, 0.580] | 0.543 [0.500, 0.588] | 0.539 [0.470, 0.610] |
| own output | main-task folds | 1 | all | null_e1_vs_e2 | 0.516 [0.485, 0.551] | 0.478 [0.438, 0.517] | 0.576 [0.529, 0.629] |
| own output | main-task folds | 1 | all | scored_by_real_vs_sim | 0.508 [0.480, 0.534] | 0.517 [0.482, 0.552] | 0.495 [0.458, 0.534] |
| own output | main-task folds | 1 | no_editor | null_e1_vs_e2 | 0.550 [0.519, 0.587] | 0.505 [0.466, 0.541] | 0.627 [0.570, 0.688] |
| own output | main-task folds | 1 | no_editor | scored_by_real_vs_sim | 0.549 [0.516, 0.580] | 0.549 [0.510, 0.591] | 0.551 [0.508, 0.604] |
| own output | main-task folds | 2 | all | null_e1_vs_e2 | 0.524 [0.491, 0.559] | 0.492 [0.452, 0.533] | 0.572 [0.523, 0.619] |
| own output | main-task folds | 2 | all | scored_by_real_vs_sim | 0.508 [0.481, 0.535] | 0.517 [0.480, 0.552] | 0.494 [0.458, 0.535] |
| own output | main-task folds | 2 | no_editor | null_e1_vs_e2 | 0.551 [0.521, 0.585] | 0.512 [0.473, 0.548] | 0.616 [0.561, 0.678] |
| own output | main-task folds | 2 | no_editor | scored_by_real_vs_sim | 0.554 [0.522, 0.584] | 0.554 [0.515, 0.595] | 0.559 [0.516, 0.612] |
| own output | main-task folds | 3 | all | null_e1_vs_e2 | 0.524 [0.491, 0.559] | 0.493 [0.453, 0.534] | 0.572 [0.525, 0.620] |
| own output | main-task folds | 3 | all | scored_by_real_vs_sim | 0.510 [0.483, 0.535] | 0.517 [0.482, 0.552] | 0.498 [0.461, 0.538] |
| own output | main-task folds | 3 | no_editor | null_e1_vs_e2 | 0.551 [0.519, 0.585] | 0.512 [0.472, 0.547] | 0.616 [0.561, 0.677] |
| own output | main-task folds | 3 | no_editor | scored_by_real_vs_sim | 0.561 [0.528, 0.592] | 0.554 [0.514, 0.598] | 0.573 [0.527, 0.624] |
| own output | main-task folds | 4 | all | null_e1_vs_e2 | 0.519 [0.486, 0.556] | 0.491 [0.450, 0.533] | 0.562 [0.514, 0.611] |
| own output | main-task folds | 4 | all | scored_by_real_vs_sim | 0.527 [0.494, 0.561] | 0.548 [0.506, 0.590] | 0.495 [0.446, 0.547] |
| own output | main-task folds | 4 | no_editor | null_e1_vs_e2 | 0.541 [0.507, 0.576] | 0.502 [0.463, 0.541] | 0.606 [0.548, 0.669] |
| own output | main-task folds | 4 | no_editor | scored_by_real_vs_sim | 0.549 [0.516, 0.580] | 0.553 [0.513, 0.594] | 0.549 [0.496, 0.600] |
| own output | main-task folds | 5 | all | null_e1_vs_e2 | 0.519 [0.486, 0.556] | 0.492 [0.451, 0.535] | 0.562 [0.514, 0.612] |
| own output | main-task folds | 5 | all | scored_by_real_vs_sim | 0.530 [0.497, 0.566] | 0.552 [0.509, 0.594] | 0.494 [0.445, 0.547] |
| own output | main-task folds | 5 | no_editor | null_e1_vs_e2 | 0.541 [0.507, 0.576] | 0.502 [0.463, 0.541] | 0.606 [0.548, 0.669] |
| own output | main-task folds | 5 | no_editor | scored_by_real_vs_sim | 0.549 [0.516, 0.580] | 0.553 [0.513, 0.594] | 0.549 [0.496, 0.600] |
| own output | main-task folds | 6 | all | null_e1_vs_e2 | 0.519 [0.486, 0.557] | 0.492 [0.451, 0.534] | 0.562 [0.514, 0.612] |
| own output | main-task folds | 6 | all | scored_by_real_vs_sim | 0.531 [0.498, 0.567] | 0.553 [0.510, 0.595] | 0.496 [0.446, 0.550] |
| own output | main-task folds | 6 | no_editor | null_e1_vs_e2 | 0.541 [0.507, 0.576] | 0.502 [0.463, 0.541] | 0.606 [0.548, 0.669] |
| own output | main-task folds | 6 | no_editor | scored_by_real_vs_sim | 0.549 [0.516, 0.580] | 0.553 [0.513, 0.594] | 0.549 [0.496, 0.600] |
| own output | main-task folds | 7 | all | null_e1_vs_e2 | 0.511 [0.479, 0.547] | 0.486 [0.441, 0.526] | 0.550 [0.504, 0.602] |
| own output | main-task folds | 7 | all | scored_by_real_vs_sim | 0.540 [0.509, 0.571] | 0.546 [0.504, 0.586] | 0.530 [0.487, 0.579] |
| own output | main-task folds | 7 | no_editor | null_e1_vs_e2 | 0.541 [0.507, 0.576] | 0.502 [0.463, 0.541] | 0.606 [0.548, 0.669] |
| own output | main-task folds | 7 | no_editor | scored_by_real_vs_sim | 0.549 [0.516, 0.580] | 0.553 [0.513, 0.594] | 0.549 [0.496, 0.600] |
| own output | held-out environments | 0 | all | null_e1_vs_e2 | 0.595 [0.563, 0.628] | 0.487 [0.452, 0.521] | 0.777 [0.725, 0.828] |
| own output | held-out environments | 0 | all | scored_by_real_vs_sim | 0.494 [0.464, 0.521] | 0.505 [0.470, 0.542] | 0.471 [0.428, 0.516] |
| own output | held-out environments | 0 | no_editor | null_e1_vs_e2 | 0.642 [0.605, 0.678] | 0.496 [0.461, 0.529] | 0.872 [0.822, 0.916] |
| own output | held-out environments | 0 | no_editor | scored_by_real_vs_sim | 0.507 [0.472, 0.539] | 0.510 [0.479, 0.542] | 0.505 [0.441, 0.573] |
| own output | held-out environments | 1 | all | null_e1_vs_e2 | 0.519 [0.491, 0.547] | 0.491 [0.456, 0.525] | 0.563 [0.513, 0.614] |
| own output | held-out environments | 1 | all | scored_by_real_vs_sim | 0.501 [0.475, 0.526] | 0.504 [0.468, 0.538] | 0.493 [0.455, 0.535] |
| own output | held-out environments | 1 | no_editor | null_e1_vs_e2 | 0.556 [0.526, 0.588] | 0.504 [0.471, 0.537] | 0.643 [0.593, 0.693] |
| own output | held-out environments | 1 | no_editor | scored_by_real_vs_sim | 0.543 [0.509, 0.576] | 0.514 [0.478, 0.551] | 0.593 [0.533, 0.657] |
| own output | held-out environments | 2 | all | null_e1_vs_e2 | 0.525 [0.498, 0.552] | 0.500 [0.466, 0.532] | 0.566 [0.521, 0.615] |
| own output | held-out environments | 2 | all | scored_by_real_vs_sim | 0.502 [0.476, 0.528] | 0.507 [0.471, 0.541] | 0.496 [0.457, 0.537] |
| own output | held-out environments | 2 | no_editor | null_e1_vs_e2 | 0.558 [0.526, 0.591] | 0.505 [0.468, 0.540] | 0.642 [0.594, 0.690] |
| own output | held-out environments | 2 | no_editor | scored_by_real_vs_sim | 0.548 [0.515, 0.581] | 0.517 [0.479, 0.554] | 0.602 [0.544, 0.664] |
| own output | held-out environments | 3 | all | null_e1_vs_e2 | 0.525 [0.498, 0.553] | 0.501 [0.468, 0.533] | 0.568 [0.522, 0.616] |
| own output | held-out environments | 3 | all | scored_by_real_vs_sim | 0.505 [0.478, 0.531] | 0.508 [0.472, 0.542] | 0.499 [0.461, 0.540] |
| own output | held-out environments | 3 | no_editor | null_e1_vs_e2 | 0.558 [0.526, 0.591] | 0.505 [0.467, 0.539] | 0.643 [0.595, 0.691] |
| own output | held-out environments | 3 | no_editor | scored_by_real_vs_sim | 0.560 [0.524, 0.594] | 0.523 [0.482, 0.565] | 0.621 [0.562, 0.679] |
| own output | held-out environments | 4 | all | null_e1_vs_e2 | 0.523 [0.495, 0.552] | 0.498 [0.465, 0.531] | 0.566 [0.518, 0.621] |
| own output | held-out environments | 4 | all | scored_by_real_vs_sim | 0.519 [0.487, 0.553] | 0.525 [0.482, 0.568] | 0.510 [0.460, 0.564] |
| own output | held-out environments | 4 | no_editor | null_e1_vs_e2 | 0.553 [0.521, 0.584] | 0.502 [0.465, 0.536] | 0.639 [0.589, 0.691] |
| own output | held-out environments | 4 | no_editor | scored_by_real_vs_sim | 0.543 [0.511, 0.577] | 0.509 [0.468, 0.550] | 0.599 [0.542, 0.657] |
| own output | held-out environments | 5 | all | null_e1_vs_e2 | 0.523 [0.495, 0.552] | 0.499 [0.465, 0.532] | 0.564 [0.516, 0.618] |
| own output | held-out environments | 5 | all | scored_by_real_vs_sim | 0.520 [0.489, 0.556] | 0.526 [0.483, 0.571] | 0.509 [0.457, 0.561] |
| own output | held-out environments | 5 | no_editor | null_e1_vs_e2 | 0.553 [0.521, 0.584] | 0.502 [0.465, 0.536] | 0.639 [0.589, 0.691] |
| own output | held-out environments | 5 | no_editor | scored_by_real_vs_sim | 0.543 [0.511, 0.577] | 0.509 [0.468, 0.550] | 0.599 [0.542, 0.657] |
| own output | held-out environments | 6 | all | null_e1_vs_e2 | 0.523 [0.494, 0.552] | 0.499 [0.465, 0.532] | 0.565 [0.517, 0.619] |
| own output | held-out environments | 6 | all | scored_by_real_vs_sim | 0.521 [0.490, 0.557] | 0.527 [0.485, 0.570] | 0.510 [0.459, 0.562] |
| own output | held-out environments | 6 | no_editor | null_e1_vs_e2 | 0.553 [0.521, 0.584] | 0.502 [0.465, 0.536] | 0.639 [0.589, 0.691] |
| own output | held-out environments | 6 | no_editor | scored_by_real_vs_sim | 0.543 [0.511, 0.577] | 0.509 [0.468, 0.550] | 0.599 [0.542, 0.657] |
| own output | held-out environments | 7 | all | null_e1_vs_e2 | 0.512 [0.488, 0.538] | 0.490 [0.461, 0.519] | 0.550 [0.510, 0.598] |
| own output | held-out environments | 7 | all | scored_by_real_vs_sim | 0.530 [0.507, 0.555] | 0.520 [0.491, 0.548] | 0.547 [0.508, 0.588] |
| own output | held-out environments | 7 | no_editor | null_e1_vs_e2 | 0.553 [0.521, 0.584] | 0.502 [0.465, 0.536] | 0.639 [0.589, 0.691] |
| own output | held-out environments | 7 | no_editor | scored_by_real_vs_sim | 0.543 [0.511, 0.577] | 0.509 [0.468, 0.550] | 0.599 [0.542, 0.657] |
| running mean | main-task folds | 0 | all | null_e1_vs_e2 | 0.636 [0.599, 0.678] | 0.473 [0.433, 0.512] | 0.886 [0.843, 0.926] |
| running mean | main-task folds | 0 | all | scored_by_real_vs_sim | 0.535 [0.504, 0.565] | 0.537 [0.503, 0.574] | 0.535 [0.474, 0.598] |
| running mean | main-task folds | 0 | no_editor | null_e1_vs_e2 | 0.653 [0.617, 0.694] | 0.499 [0.464, 0.532] | 0.914 [0.875, 0.949] |
| running mean | main-task folds | 0 | no_editor | scored_by_real_vs_sim | 0.545 [0.506, 0.584] | 0.514 [0.472, 0.553] | 0.597 [0.526, 0.674] |
| running mean | main-task folds | 1 | all | null_e1_vs_e2 | 0.552 [0.517, 0.590] | 0.474 [0.435, 0.516] | 0.670 [0.616, 0.726] |
| running mean | main-task folds | 1 | all | scored_by_real_vs_sim | 0.531 [0.506, 0.555] | 0.534 [0.503, 0.567] | 0.524 [0.480, 0.571] |
| running mean | main-task folds | 1 | no_editor | null_e1_vs_e2 | 0.562 [0.529, 0.598] | 0.481 [0.444, 0.516] | 0.688 [0.636, 0.743] |
| running mean | main-task folds | 1 | no_editor | scored_by_real_vs_sim | 0.577 [0.545, 0.609] | 0.528 [0.492, 0.563] | 0.650 [0.601, 0.708] |
| running mean | main-task folds | 2 | all | null_e1_vs_e2 | 0.559 [0.525, 0.596] | 0.486 [0.448, 0.529] | 0.663 [0.612, 0.716] |
| running mean | main-task folds | 2 | all | scored_by_real_vs_sim | 0.531 [0.505, 0.555] | 0.534 [0.503, 0.567] | 0.521 [0.479, 0.566] |
| running mean | main-task folds | 2 | no_editor | null_e1_vs_e2 | 0.566 [0.533, 0.600] | 0.495 [0.460, 0.528] | 0.680 [0.629, 0.736] |
| running mean | main-task folds | 2 | no_editor | scored_by_real_vs_sim | 0.582 [0.548, 0.613] | 0.534 [0.495, 0.570] | 0.657 [0.608, 0.716] |
| running mean | main-task folds | 3 | all | null_e1_vs_e2 | 0.559 [0.525, 0.596] | 0.486 [0.448, 0.529] | 0.663 [0.613, 0.716] |
| running mean | main-task folds | 3 | all | scored_by_real_vs_sim | 0.532 [0.506, 0.556] | 0.534 [0.503, 0.568] | 0.524 [0.481, 0.568] |
| running mean | main-task folds | 3 | no_editor | null_e1_vs_e2 | 0.565 [0.532, 0.600] | 0.494 [0.459, 0.528] | 0.680 [0.628, 0.735] |
| running mean | main-task folds | 3 | no_editor | scored_by_real_vs_sim | 0.584 [0.549, 0.616] | 0.534 [0.497, 0.571] | 0.660 [0.612, 0.719] |
| running mean | main-task folds | 4 | all | null_e1_vs_e2 | 0.559 [0.524, 0.597] | 0.491 [0.451, 0.532] | 0.662 [0.608, 0.716] |
| running mean | main-task folds | 4 | all | scored_by_real_vs_sim | 0.563 [0.532, 0.594] | 0.553 [0.516, 0.588] | 0.579 [0.531, 0.630] |
| running mean | main-task folds | 4 | no_editor | null_e1_vs_e2 | 0.558 [0.525, 0.592] | 0.486 [0.451, 0.521] | 0.677 [0.625, 0.736] |
| running mean | main-task folds | 4 | no_editor | scored_by_real_vs_sim | 0.581 [0.549, 0.611] | 0.539 [0.504, 0.576] | 0.647 [0.594, 0.708] |
| running mean | main-task folds | 5 | all | null_e1_vs_e2 | 0.559 [0.524, 0.597] | 0.493 [0.453, 0.533] | 0.662 [0.608, 0.716] |
| running mean | main-task folds | 5 | all | scored_by_real_vs_sim | 0.569 [0.538, 0.602] | 0.558 [0.520, 0.595] | 0.586 [0.535, 0.639] |
| running mean | main-task folds | 5 | no_editor | null_e1_vs_e2 | 0.558 [0.525, 0.592] | 0.486 [0.451, 0.521] | 0.677 [0.625, 0.736] |
| running mean | main-task folds | 5 | no_editor | scored_by_real_vs_sim | 0.581 [0.549, 0.611] | 0.539 [0.504, 0.576] | 0.647 [0.594, 0.708] |
| running mean | main-task folds | 6 | all | null_e1_vs_e2 | 0.559 [0.524, 0.598] | 0.493 [0.453, 0.533] | 0.662 [0.609, 0.716] |
| running mean | main-task folds | 6 | all | scored_by_real_vs_sim | 0.569 [0.538, 0.603] | 0.557 [0.518, 0.594] | 0.588 [0.538, 0.642] |
| running mean | main-task folds | 6 | no_editor | null_e1_vs_e2 | 0.558 [0.525, 0.592] | 0.486 [0.451, 0.521] | 0.677 [0.625, 0.736] |
| running mean | main-task folds | 6 | no_editor | scored_by_real_vs_sim | 0.581 [0.549, 0.611] | 0.539 [0.504, 0.576] | 0.647 [0.594, 0.708] |
| running mean | main-task folds | 7 | all | null_e1_vs_e2 | 0.551 [0.515, 0.589] | 0.493 [0.456, 0.534] | 0.648 [0.595, 0.706] |
| running mean | main-task folds | 7 | all | scored_by_real_vs_sim | 0.578 [0.547, 0.609] | 0.551 [0.511, 0.591] | 0.624 [0.578, 0.681] |
| running mean | main-task folds | 7 | no_editor | null_e1_vs_e2 | 0.558 [0.525, 0.592] | 0.486 [0.451, 0.521] | 0.677 [0.625, 0.736] |
| running mean | main-task folds | 7 | no_editor | scored_by_real_vs_sim | 0.581 [0.549, 0.611] | 0.539 [0.504, 0.576] | 0.647 [0.594, 0.708] |
| running mean | held-out environments | 0 | all | null_e1_vs_e2 | 0.629 [0.596, 0.666] | 0.484 [0.449, 0.517] | 0.866 [0.820, 0.909] |
| running mean | held-out environments | 0 | all | scored_by_real_vs_sim | 0.522 [0.491, 0.551] | 0.522 [0.488, 0.558] | 0.524 [0.466, 0.577] |
| running mean | held-out environments | 0 | no_editor | null_e1_vs_e2 | 0.661 [0.627, 0.700] | 0.511 [0.481, 0.541] | 0.909 [0.870, 0.944] |
| running mean | held-out environments | 0 | no_editor | scored_by_real_vs_sim | 0.520 [0.482, 0.555] | 0.496 [0.464, 0.528] | 0.565 [0.497, 0.636] |
| running mean | held-out environments | 1 | all | null_e1_vs_e2 | 0.548 [0.523, 0.578] | 0.499 [0.466, 0.529] | 0.635 [0.590, 0.682] |
| running mean | held-out environments | 1 | all | scored_by_real_vs_sim | 0.522 [0.497, 0.547] | 0.514 [0.484, 0.545] | 0.529 [0.486, 0.571] |
| running mean | held-out environments | 1 | no_editor | null_e1_vs_e2 | 0.591 [0.563, 0.622] | 0.521 [0.493, 0.551] | 0.715 [0.666, 0.770] |
| running mean | held-out environments | 1 | no_editor | scored_by_real_vs_sim | 0.580 [0.549, 0.613] | 0.507 [0.475, 0.538] | 0.711 [0.656, 0.767] |
| running mean | held-out environments | 2 | all | null_e1_vs_e2 | 0.554 [0.527, 0.583] | 0.502 [0.469, 0.532] | 0.640 [0.595, 0.684] |
| running mean | held-out environments | 2 | all | scored_by_real_vs_sim | 0.521 [0.497, 0.547] | 0.518 [0.486, 0.551] | 0.527 [0.484, 0.570] |
| running mean | held-out environments | 2 | no_editor | null_e1_vs_e2 | 0.599 [0.569, 0.632] | 0.531 [0.501, 0.562] | 0.719 [0.670, 0.774] |
| running mean | held-out environments | 2 | no_editor | scored_by_real_vs_sim | 0.578 [0.544, 0.612] | 0.501 [0.467, 0.532] | 0.715 [0.658, 0.775] |
| running mean | held-out environments | 3 | all | null_e1_vs_e2 | 0.555 [0.527, 0.585] | 0.504 [0.471, 0.534] | 0.642 [0.598, 0.686] |
| running mean | held-out environments | 3 | all | scored_by_real_vs_sim | 0.523 [0.498, 0.549] | 0.519 [0.487, 0.552] | 0.530 [0.486, 0.573] |
| running mean | held-out environments | 3 | no_editor | null_e1_vs_e2 | 0.599 [0.569, 0.632] | 0.531 [0.502, 0.561] | 0.719 [0.668, 0.775] |
| running mean | held-out environments | 3 | no_editor | scored_by_real_vs_sim | 0.584 [0.550, 0.620] | 0.502 [0.466, 0.536] | 0.731 [0.677, 0.789] |
| running mean | held-out environments | 4 | all | null_e1_vs_e2 | 0.551 [0.525, 0.580] | 0.496 [0.465, 0.527] | 0.643 [0.593, 0.693] |
| running mean | held-out environments | 4 | all | scored_by_real_vs_sim | 0.542 [0.513, 0.574] | 0.504 [0.465, 0.541] | 0.608 [0.560, 0.658] |
| running mean | held-out environments | 4 | no_editor | null_e1_vs_e2 | 0.593 [0.563, 0.626] | 0.515 [0.488, 0.542] | 0.725 [0.673, 0.783] |
| running mean | held-out environments | 4 | no_editor | scored_by_real_vs_sim | 0.575 [0.540, 0.610] | 0.495 [0.460, 0.526] | 0.725 [0.667, 0.785] |
| running mean | held-out environments | 5 | all | null_e1_vs_e2 | 0.552 [0.525, 0.580] | 0.497 [0.465, 0.526] | 0.642 [0.594, 0.692] |
| running mean | held-out environments | 5 | all | scored_by_real_vs_sim | 0.541 [0.513, 0.573] | 0.503 [0.463, 0.540] | 0.605 [0.560, 0.654] |
| running mean | held-out environments | 5 | no_editor | null_e1_vs_e2 | 0.593 [0.563, 0.626] | 0.515 [0.488, 0.542] | 0.725 [0.673, 0.783] |
| running mean | held-out environments | 5 | no_editor | scored_by_real_vs_sim | 0.575 [0.540, 0.610] | 0.495 [0.460, 0.526] | 0.725 [0.667, 0.785] |
| running mean | held-out environments | 6 | all | null_e1_vs_e2 | 0.551 [0.525, 0.580] | 0.496 [0.464, 0.526] | 0.643 [0.593, 0.693] |
| running mean | held-out environments | 6 | all | scored_by_real_vs_sim | 0.543 [0.515, 0.575] | 0.505 [0.465, 0.542] | 0.608 [0.562, 0.657] |
| running mean | held-out environments | 6 | no_editor | null_e1_vs_e2 | 0.593 [0.563, 0.626] | 0.515 [0.488, 0.542] | 0.725 [0.673, 0.783] |
| running mean | held-out environments | 6 | no_editor | scored_by_real_vs_sim | 0.575 [0.540, 0.610] | 0.495 [0.460, 0.526] | 0.725 [0.667, 0.785] |
| running mean | held-out environments | 7 | all | null_e1_vs_e2 | 0.546 [0.522, 0.572] | 0.492 [0.465, 0.517] | 0.635 [0.594, 0.682] |
| running mean | held-out environments | 7 | all | scored_by_real_vs_sim | 0.547 [0.525, 0.572] | 0.500 [0.476, 0.523] | 0.631 [0.587, 0.681] |
| running mean | held-out environments | 7 | no_editor | null_e1_vs_e2 | 0.593 [0.563, 0.626] | 0.515 [0.488, 0.542] | 0.725 [0.673, 0.783] |
| running mean | held-out environments | 7 | no_editor | scored_by_real_vs_sim | 0.575 [0.540, 0.610] | 0.495 [0.460, 0.526] | 0.725 [0.667, 0.785] |
| pilot (20 tasks held out) | main-task folds | 0 | all | null_e1_vs_e2 | 0.594 [0.558, 0.632] | 0.464 [0.423, 0.504] | 0.790 [0.738, 0.845] |
| pilot (20 tasks held out) | main-task folds | 0 | all | scored_by_real_vs_sim | 0.504 [0.473, 0.532] | 0.518 [0.480, 0.557] | 0.481 [0.439, 0.527] |
| pilot (20 tasks held out) | main-task folds | 0 | no_editor | null_e1_vs_e2 | 0.640 [0.606, 0.679] | 0.501 [0.462, 0.536] | 0.859 [0.814, 0.904] |
| pilot (20 tasks held out) | main-task folds | 0 | no_editor | scored_by_real_vs_sim | 0.540 [0.500, 0.580] | 0.545 [0.502, 0.590] | 0.537 [0.471, 0.607] |
| pilot (20 tasks held out) | main-task folds | 7 | all | null_e1_vs_e2 | 0.515 [0.482, 0.552] | 0.488 [0.442, 0.530] | 0.558 [0.511, 0.611] |
| pilot (20 tasks held out) | main-task folds | 7 | all | scored_by_real_vs_sim | 0.541 [0.509, 0.572] | 0.543 [0.501, 0.583] | 0.539 [0.495, 0.589] |
| pilot (20 tasks held out) | main-task folds | 7 | no_editor | null_e1_vs_e2 | 0.546 [0.512, 0.582] | 0.509 [0.467, 0.546] | 0.612 [0.556, 0.675] |
| pilot (20 tasks held out) | main-task folds | 7 | no_editor | scored_by_real_vs_sim | 0.553 [0.521, 0.584] | 0.559 [0.519, 0.600] | 0.547 [0.494, 0.598] |

## Real honest vs simulated honest, by the first real run's batch (own output)

| split | level | outputs | all cells | same batch (overlap800 real) | base200 real |
|---|---|---|---|---|---|
| main-task folds | 0 | all | 0.973 [0.963, 0.983] | 0.978 [0.969, 0.987] | 0.954 [0.921, 0.978] |
| main-task folds | 0 | no_editor | 0.929 [0.914, 0.944] | 0.932 [0.915, 0.948] | 0.921 [0.883, 0.954] |
| main-task folds | 1 | all | 0.932 [0.915, 0.948] | 0.937 [0.919, 0.953] | 0.911 [0.866, 0.946] |
| main-task folds | 1 | no_editor | 0.654 [0.629, 0.680] | 0.649 [0.622, 0.678] | 0.673 [0.626, 0.726] |
| main-task folds | 2 | all | 0.930 [0.913, 0.947] | 0.935 [0.917, 0.952] | 0.909 [0.865, 0.944] |
| main-task folds | 2 | no_editor | 0.651 [0.624, 0.676] | 0.645 [0.616, 0.676] | 0.670 [0.623, 0.724] |
| main-task folds | 3 | all | 0.922 [0.905, 0.940] | 0.928 [0.909, 0.946] | 0.899 [0.852, 0.937] |
| main-task folds | 3 | no_editor | 0.607 [0.584, 0.633] | 0.600 [0.575, 0.630] | 0.634 [0.589, 0.688] |
| main-task folds | 4 | all | 0.814 [0.788, 0.841] | 0.825 [0.795, 0.853] | 0.776 [0.714, 0.830] |
| main-task folds | 4 | no_editor | 0.580 [0.557, 0.606] | 0.578 [0.552, 0.608] | 0.591 [0.543, 0.646] |
| main-task folds | 5 | all | 0.782 [0.754, 0.809] | 0.792 [0.761, 0.822] | 0.744 [0.683, 0.801] |
| main-task folds | 5 | no_editor | 0.580 [0.557, 0.606] | 0.578 [0.552, 0.608] | 0.591 [0.543, 0.646] |
| main-task folds | 6 | all | 0.777 [0.749, 0.804] | 0.787 [0.755, 0.816] | 0.742 [0.681, 0.798] |
| main-task folds | 6 | no_editor | 0.580 [0.557, 0.606] | 0.578 [0.552, 0.608] | 0.591 [0.543, 0.646] |
| main-task folds | 7 | all | 0.582 [0.557, 0.606] | 0.578 [0.550, 0.608] | 0.594 [0.554, 0.639] |
| main-task folds | 7 | no_editor | 0.580 [0.557, 0.606] | 0.578 [0.552, 0.608] | 0.591 [0.543, 0.646] |
| held-out environments | 0 | all | 0.968 [0.957, 0.979] | 0.973 [0.962, 0.983] | 0.951 [0.920, 0.977] |
| held-out environments | 0 | no_editor | 0.953 [0.941, 0.964] | 0.953 [0.940, 0.966] | 0.952 [0.922, 0.976] |
| held-out environments | 1 | all | 0.927 [0.910, 0.944] | 0.930 [0.911, 0.947] | 0.916 [0.874, 0.951] |
| held-out environments | 1 | no_editor | 0.672 [0.641, 0.699] | 0.647 [0.613, 0.681] | 0.764 [0.708, 0.817] |
| held-out environments | 2 | all | 0.928 [0.911, 0.945] | 0.932 [0.912, 0.949] | 0.913 [0.871, 0.949] |
| held-out environments | 2 | no_editor | 0.665 [0.636, 0.694] | 0.643 [0.611, 0.680] | 0.742 [0.689, 0.792] |
| held-out environments | 3 | all | 0.920 [0.902, 0.938] | 0.923 [0.903, 0.942] | 0.905 [0.859, 0.942] |
| held-out environments | 3 | no_editor | 0.604 [0.575, 0.633] | 0.579 [0.548, 0.613] | 0.692 [0.644, 0.743] |
| held-out environments | 4 | all | 0.812 [0.787, 0.836] | 0.823 [0.794, 0.850] | 0.769 [0.712, 0.821] |
| held-out environments | 4 | no_editor | 0.561 [0.534, 0.587] | 0.537 [0.509, 0.567] | 0.649 [0.601, 0.705] |
| held-out environments | 5 | all | 0.779 [0.754, 0.804] | 0.790 [0.761, 0.819] | 0.739 [0.682, 0.788] |
| held-out environments | 5 | no_editor | 0.561 [0.534, 0.587] | 0.537 [0.509, 0.567] | 0.649 [0.601, 0.705] |
| held-out environments | 6 | all | 0.773 [0.747, 0.798] | 0.783 [0.754, 0.813] | 0.736 [0.679, 0.786] |
| held-out environments | 6 | no_editor | 0.561 [0.534, 0.587] | 0.537 [0.509, 0.567] | 0.649 [0.601, 0.705] |
| held-out environments | 7 | all | 0.562 [0.542, 0.582] | 0.546 [0.522, 0.570] | 0.621 [0.578, 0.664] |
| held-out environments | 7 | no_editor | 0.561 [0.534, 0.587] | 0.537 [0.509, 0.567] | 0.649 [0.601, 0.705] |