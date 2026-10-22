//! SV2-C2c (W13 lane `finitum-data`): region partitions of tensor boxes, per-region datum
//! values, regionwise two-point diffusion, serde round trips of the FV receipts, and the W12
//! identities of uniform data pinned unchanged.
use std::collections::BTreeMap;
use std::sync::Arc;

use finitum::{
    FinitumError, FvCfl, FvConservationReceipt, FvDatumValue, FvDiffusion, FvFaceAveraging,
    FvRealization, FvRealizationReceipt, FvRegionDiffusion, FvTopology, FvTopologyInvariants,
    PlanarCoverageReport, RegionMap, RegionTagId, TensorAxis, TensorBox, TensorInterface,
    TensorRegion, TensorRegionPartition,
};
use methodus::{DaeOperator, EvaluationContext};
use quantitas::UnitRegistry;
use scientia::{
    ConservationLawProgram, ConservationLawSystem, FvMethodIntent, GradientScheme, LimiterFamily,
    NumericalFluxIntent, PositivityIntent, ReconstructionIntent, ReconstructionVariables,
    SemanticModule, SourceTreatment, compile_conservation_law_system, compile_semantics,
    lower_conservation_law,
};

const SCALAR: &str = include_str!("fixtures/fv-scalar.res");
const MUSCL: &str = include_str!("fixtures/fv-muscl.res");
const REGIONS: &str = include_str!("fixtures/fv-regions.res");

fn module(source: &str) -> SemanticModule {
    compile_semantics(source, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic
}

fn law(
    source: &str,
    model: &str,
    numerical_flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
) -> (ConservationLawSystem, Arc<ConservationLawProgram>) {
    let module = module(source);
    let system = compile_conservation_law_system(&module, model, "balance", "u").unwrap();
    let intent = FvMethodIntent {
        numerical_flux,
        reconstruction,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = lower_conservation_law(&module, &system, &intent, &[]).unwrap();
    (system, Arc::new(program))
}

fn regions(system: &ConservationLawSystem) -> RegionMap {
    let mut map = RegionMap::new();
    for boundary in &system.boundaries {
        let (scientia::BoundaryFluxSemantics::GhostState { region, .. }
        | scientia::BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tag = match boundary.region_name() {
            "left" => "x_min",
            "right" => "x_max",
            "bottom" => "y_min",
            "top" => "y_max",
            other => panic!("unexpected boundary {other}"),
        };
        map.insert(*region, [RegionTagId::new(tag)]);
    }
    map
}

fn uniform(data: &[(&str, f64)]) -> BTreeMap<String, FvDatumValue> {
    data.iter()
        .map(|(name, value)| (name.to_string(), FvDatumValue::Uniform(*value)))
        .collect()
}

fn per_region(values: &[(&str, f64)]) -> FvDatumValue {
    FvDatumValue::PerRegion(
        values
            .iter()
            .map(|(tag, value)| (RegionTagId::new(*tag), *value))
            .collect(),
    )
}

fn tags(values: &[(&str, f64)]) -> BTreeMap<RegionTagId, f64> {
    values
        .iter()
        .map(|(tag, value)| (RegionTagId::new(*tag), *value))
        .collect()
}

fn bind(
    source: &str,
    model: &str,
    flux: NumericalFluxIntent,
    topology: &Arc<FvTopology>,
    data: &BTreeMap<String, FvDatumValue>,
) -> Result<FvRealization, FinitumError> {
    let (system, program) = law(source, model, flux, ReconstructionIntent::Constant);
    FvRealization::bind(&system, program, topology.clone(), &regions(&system), data)
}

fn line(cells: usize, periodic: bool) -> Arc<FvTopology> {
    let axis = TensorAxis::uniform(0.0, 1.0, cells);
    let axis = if periodic { axis.periodic() } else { axis };
    Arc::new(FvTopology::from_tensor_box(&TensorBox { axes: vec![axis] }).unwrap())
}

/// `[0, 1]` split at `x = 0.5` into `left` and `right` with interface `contact`.
fn split_line(cells: usize, periodic: bool) -> (TensorBox, Arc<FvTopology>) {
    let axis = TensorAxis::uniform(0.0, 1.0, cells);
    let axis = if periodic { axis.periodic() } else { axis };
    let spec = TensorBox { axes: vec![axis] };
    let middle = spec.node_index_at(0, 0.5).unwrap();
    let partition = TensorRegionPartition {
        regions: vec![
            TensorRegion {
                tag: RegionTagId::new("left"),
                cells: vec![[0, middle]],
            },
            TensorRegion {
                tag: RegionTagId::new("right"),
                cells: vec![[middle, cells]],
            },
        ],
        interfaces: vec![TensorInterface {
            tag: RegionTagId::new("contact"),
            between: [RegionTagId::new("left"), RegionTagId::new("right")],
        }],
    };
    let topology = FvTopology::from_partitioned_tensor_box(&spec, &partition).unwrap();
    (spec, Arc::new(topology))
}

/// `[0, 2] × [−1, 1]`, 4 × 3 cells, optionally periodic in both axes.
fn plane_spec(periodic: bool) -> TensorBox {
    let wrap = |axis: TensorAxis| if periodic { axis.periodic() } else { axis };
    TensorBox {
        axes: vec![
            wrap(TensorAxis::uniform(0.0, 2.0, 4)),
            wrap(TensorAxis::uniform(-1.0, 1.0, 3)),
        ],
    }
}

/// The plane split by the row `j = 1`: `solid` is the bottom row, `fluid` the two above.
fn plane_partition() -> TensorRegionPartition {
    TensorRegionPartition {
        regions: vec![
            TensorRegion {
                tag: RegionTagId::new("solid"),
                cells: vec![[0, 4], [0, 1]],
            },
            TensorRegion {
                tag: RegionTagId::new("fluid"),
                cells: vec![[0, 4], [1, 3]],
            },
        ],
        interfaces: vec![TensorInterface {
            tag: RegionTagId::new("contact"),
            between: [RegionTagId::new("solid"), RegionTagId::new("fluid")],
        }],
    }
}

fn cx() -> EvaluationContext {
    EvaluationContext::reproducible()
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut seed = seed | 1;
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn code(error: &FinitumError) -> String {
    let message = error.to_string();
    message
        .split(':')
        .find(|part| part.trim_start().starts_with("FV_"))
        .map(|part| part.trim().to_string())
        .unwrap_or(message)
}

fn hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn bytes_of(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn json_hex<T: serde::Serialize>(value: &T) -> String {
    hex(&serde_json::to_vec(value).unwrap())
}

fn residual(operator: &FvRealization, state: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    let rate = vec![0.0; state.len()];
    DaeOperator::residual(operator, &cx(), 0.0, state, &rate, &mut out).unwrap();
    out
}

fn jvp(operator: &FvRealization, state: &[f64], direction: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    let zero = vec![0.0; state.len()];
    DaeOperator::jacobian_vector_product(
        operator, &cx(), 0.0, state, &zero, direction, &zero, &mut out,
    )
    .unwrap();
    out
}

/// Solve the linear steady problem `R(u) = 0` by assembling `J` from JVP columns (the
/// SV2-C2b steady path).
fn steady(operator: &FvRealization) -> Vec<f64> {
    let n = operator.dimension();
    let zero = vec![0.0; n];
    let b = residual(operator, &zero);
    let mut matrix = vec![vec![0.0; n + 1]; n];
    for column in 0..n {
        let mut e = vec![0.0; n];
        e[column] = 1.0;
        let jc = jvp(operator, &zero, &e);
        for row in 0..n {
            matrix[row][column] = jc[row];
        }
    }
    for row in 0..n {
        matrix[row][n] = -b[row];
    }
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|a, b| matrix[*a][pivot].abs().total_cmp(&matrix[*b][pivot].abs()))
            .unwrap();
        matrix.swap(pivot, best);
        let (head, tail) = matrix.split_at_mut(pivot + 1);
        let pivot_row = &head[pivot];
        for row in tail {
            let factor = row[pivot] / pivot_row[pivot];
            for (value, above) in row[pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *value -= factor * above;
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|c| matrix[row][c] * solution[c])
            .sum::<f64>();
        solution[row] = (matrix[row][n] - tail) / matrix[row][row];
    }
    solution
}

fn hand_llf(a: f64, um: f64, up: f64, n: f64) -> f64 {
    let fm = (a * um) * n;
    let fp = (a * up) * n;
    let speed = (a * n).abs();
    0.5 * (fm + fp) - (0.5 * speed.max(speed)) * (up - um)
}

#[test]
fn w12_identities_of_uniform_data_are_unchanged() {
    // Captured from W12 head 90c4789 (this lane's base) before any SV2-C2c change.
    let plane = FvTopology::from_tensor_box(&plane_spec(false)).unwrap();
    let graded = FvTopology::from_tensor_box(&TensorBox {
        axes: vec![
            TensorAxis::graded(vec![0.0, 0.1, 0.3, 0.6, 1.0]).periodic(),
            TensorAxis::uniform(0.0, 1.0, 2),
        ],
    })
    .unwrap();
    for (label, digest, expected) in [
        (
            "line 8",
            line(8, false).digest().hex.clone(),
            "dea107d97aca1228ac8455efeab9c7963002c4ad3bedb6df727d95ac643d6c22",
        ),
        (
            "periodic line 8",
            line(8, true).digest().hex.clone(),
            "bdb1abbb56ba0d9dabca9a1aba0705b1dc8269c6288e8981f7ea15c9cba51792",
        ),
        (
            "plane",
            plane.digest().hex.clone(),
            "b3c72c007200e96a5f17ea407861d959dccca36c0814e495efcfd17bf1038432",
        ),
        (
            "plane source",
            plane.source_digest().hex.clone(),
            "01d0d47577630f449e55e0f6250f0124232b1237241b406254dff53abf0d5ed2",
        ),
        (
            "graded",
            graded.digest().hex.clone(),
            "1fe4962d98cc90347afe760dc8fdcf827525f130087f621ee93338d09ebc54bf",
        ),
    ] {
        assert_eq!(digest, expected, "{label}");
    }
    assert!(!plane.has_regions() && plane.interfaces().is_empty());
    let invariants = plane.check_invariants().unwrap();
    assert_eq!((invariants.regions, invariants.interface_faces), (0, 0));

    // Uniform two-point diffusion: identity, receipt, residual and JVP bit for bit.
    let (system, program) = law(
        MUSCL,
        "DirichletAdvection1D",
        NumericalFluxIntent::Central,
        ReconstructionIntent::Constant,
    );
    let data = [("a".to_string(), 0.5)].into_iter().collect();
    let through_new = FvRealization::new(
        &system,
        program.clone(),
        line(10, false),
        &regions(&system),
        &data,
    )
    .unwrap()
    .with_diffusion(FvDiffusion { coefficient: 0.7 })
    .unwrap();
    assert_eq!(
        through_new.digest().hex,
        "cd4ccb2ea949336bd88600cdaa15bc6a914b06a09fe93a42a05953b42810fe4a"
    );
    assert_eq!(
        json_hex(&through_new.receipt()),
        "f3f70b90c300ce1008eb51736aaf7e0dcabfbb84bb8a69df03c260e7326af675"
    );
    let state = (0..10)
        .map(|i| ((i * 7 + 3) % 10) as f64 * 0.1 - 0.3)
        .collect::<Vec<_>>();
    assert_eq!(
        hex(&bytes_of(&residual(&through_new, &state))),
        "756dc800d5389e0ef93aaa5d2722639a73a0e33ea3d6ff6bfb86900c6066a011"
    );
    let direction = (0..10).map(|i| (i as f64 - 4.5) * 0.2).collect::<Vec<_>>();
    assert_eq!(
        hex(&bytes_of(&jvp(&through_new, &state, &direction))),
        "c9447cccabb2933771e04b8c7d848c8e9db798a7705d0230ad90f0484baf5293"
    );
    // `bind` with `Uniform` values is the same operator.
    let through_bind = FvRealization::bind(
        &system,
        program,
        line(10, false),
        &regions(&system),
        &uniform(&[("a", 0.5)]),
    )
    .unwrap()
    .with_diffusion(FvDiffusion { coefficient: 0.7 })
    .unwrap();
    assert_eq!(through_bind.digest(), through_new.digest());
    assert_eq!(through_bind.receipt(), through_new.receipt());
    assert_eq!(
        through_bind.data(),
        [("a".to_string(), FvDatumValue::Uniform(0.5))]
            .into_iter()
            .collect()
    );

    // First-order realizations without diffusion (the SV2-C2b pins and a 2-D one).
    for (source, model, flux, topology, data, digest, receipt) in [
        (
            SCALAR,
            "Burgers1D",
            NumericalFluxIntent::Hll,
            line(8, true),
            vec![("k", 0.3)],
            "d4e8bc9aab85db722c26899cf82c9478e9304e609725a2d966ea496b09634299",
            "b78d546e18aa3cc3ecc402a46f20e4060c7b0c8b3c5b2a1e3f36b0bec992638c",
        ),
        (
            SCALAR,
            "PeriodicAdvection2D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            Arc::new(FvTopology::from_tensor_box(&plane_spec(true)).unwrap()),
            vec![("ax", 1.0), ("ay", -0.5)],
            "2b35fae0c5bad6cf0e58774f4e329e38b5486614c0d5246d5cc54521bb02ff97",
            "ee9074d45fc7e37b836c0e4b1269d1214c3035cc305806a971067566ce01c73e",
        ),
    ] {
        let operator = bind(source, model, flux, &topology, &uniform(&data)).unwrap();
        assert_eq!(operator.digest().hex, digest, "{model}");
        assert_eq!(json_hex(&operator.receipt()), receipt, "{model}");
    }

    // The wire form of a datum value is a number or a tag map.
    assert_eq!(
        serde_json::to_string(&FvDatumValue::Uniform(0.5)).unwrap(),
        "0.5"
    );
    assert_eq!(
        serde_json::to_string(&per_region(&[("fluid", 2.0), ("solid", 1.0)])).unwrap(),
        r#"{"fluid":2.0,"solid":1.0}"#
    );
    assert_eq!(
        serde_json::from_str::<FvDatumValue>("0.5").unwrap(),
        FvDatumValue::Uniform(0.5)
    );
    assert_eq!(
        serde_json::from_str::<FvDatumValue>(r#"{"solid":1.0}"#).unwrap(),
        per_region(&[("solid", 1.0)])
    );
}

#[test]
fn region_partition_tags_cells_and_interfaces_and_keeps_the_block_layout() {
    let spec = plane_spec(false);
    let plain = FvTopology::from_tensor_box(&spec).unwrap();
    let split = FvTopology::from_partitioned_tensor_box(&spec, &plane_partition()).unwrap();
    // Same cells, faces, blocks and order; only the region columns and identities differ.
    assert_eq!(split.cell_volumes(), plain.cell_volumes());
    assert_eq!(split.cell_centroids(), plain.cell_centroids());
    assert_eq!(split.interior(), plain.interior());
    assert_eq!(split.periodic(), plain.periodic());
    assert_eq!(split.boundary(), plain.boundary());
    assert_ne!(split.digest(), plain.digest());
    assert_ne!(split.source_digest(), plain.source_digest());
    // Axis 0 fastest: cells 0..4 are the bottom row.
    let expected = (0..12)
        .map(|cell| RegionTagId::new(if cell < 4 { "solid" } else { "fluid" }))
        .collect::<Vec<_>>();
    assert_eq!(split.cell_regions(), expected.as_slice());
    assert!(split.has_regions());
    assert_eq!(
        split.region_tags().into_iter().cloned().collect::<Vec<_>>(),
        [RegionTagId::new("fluid"), RegionTagId::new("solid")]
    );
    let contact = &split.interfaces()[&RegionTagId::new("contact")];
    assert_eq!(contact.len(), 4);
    assert!(contact.periodic.is_empty());
    for face in &contact.interior {
        let (minus, plus) = (split.interior().minus[*face], split.interior().plus[*face]);
        assert_eq!((minus / 4, plus / 4), (0, 1), "face {face}");
        assert_eq!(minus % 4, plus % 4);
        assert_eq!(split.interior().normal[1][*face], 1.0);
    }
    let invariants = split.check_invariants().unwrap();
    assert_eq!((invariants.regions, invariants.interface_faces), (2, 4));
    assert_eq!(invariants.cells, 12);
    // A periodic wrap between two regions is an interface face of the periodic block.
    let (spec, ring) = split_line(6, true);
    assert_eq!(spec.node_index_at(0, 0.5).unwrap(), 3);
    assert_eq!(spec.cell_counts().unwrap(), [6]);
    let contact = &ring.interfaces()[&RegionTagId::new("contact")];
    assert_eq!((contact.interior.as_slice(), contact.periodic.as_slice()), ([2].as_slice(), [0].as_slice()));
    let invariants = ring.check_invariants().unwrap();
    assert_eq!((invariants.regions, invariants.interface_faces, invariants.periodic_faces), (2, 2, 1));
    // The partition and the topology serialize and read back.
    let partition = plane_partition();
    let text = serde_json::to_string(&partition).unwrap();
    assert_eq!(serde_json::from_str::<TensorRegionPartition>(&text).unwrap(), partition);
    let spec_text = serde_json::to_string(&plane_spec(true)).unwrap();
    assert_eq!(serde_json::from_str::<TensorBox>(&spec_text).unwrap(), plane_spec(true));
    // A node lookup never snaps.
    assert_eq!(code(&spec.node_index_at(0, 0.49).unwrap_err()), "FV_REGION_BOUNDARY_OFF_NODE");
    assert_eq!(code(&spec.node_index_at(1, 0.5).unwrap_err()), "FV_REGION_RANGE_INVALID");
}

#[test]
fn region_partition_refusals() {
    let spec = plane_spec(false);
    let region = |tag: &str, x: [usize; 2], y: [usize; 2]| TensorRegion {
        tag: RegionTagId::new(tag),
        cells: vec![x, y],
    };
    let interface = |tag: &str, a: &str, b: &str| TensorInterface {
        tag: RegionTagId::new(tag),
        between: [RegionTagId::new(a), RegionTagId::new(b)],
    };
    let contact = || interface("contact", "solid", "fluid");
    let refusal = |partition: TensorRegionPartition| {
        code(&FvTopology::from_partitioned_tensor_box(&spec, &partition).unwrap_err())
    };
    let cases: Vec<(&str, TensorRegionPartition)> = vec![
        (
            "FV_REGION_RANGE_INVALID",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 4])],
                interfaces: vec![],
            },
        ),
        (
            "FV_REGION_RANGE_INVALID",
            TensorRegionPartition {
                regions: vec![TensorRegion {
                    tag: RegionTagId::new("solid"),
                    cells: vec![[0, 4]],
                }],
                interfaces: vec![],
            },
        ),
        (
            "FV_REGION_RANGE_INVALID",
            TensorRegionPartition {
                regions: vec![region("solid", [2, 2], [0, 3])],
                interfaces: vec![],
            },
        ),
        (
            "FV_REGION_OVERLAP",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 2]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![contact()],
            },
        ),
        (
            "FV_REGION_UNCOVERED",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [2, 3])],
                interfaces: vec![contact()],
            },
        ),
        (
            "FV_REGION_TAG_DUPLICATE",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("solid", [0, 4], [1, 3])],
                interfaces: vec![],
            },
        ),
        (
            "FV_REGION_TAG_OVERLAP",
            TensorRegionPartition {
                regions: vec![region("x_min", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![interface("contact", "x_min", "fluid")],
            },
        ),
        (
            "FV_REGION_TAG_OVERLAP",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![interface("solid", "solid", "fluid")],
            },
        ),
        (
            "FV_REGION_INTERFACE_UNTAGGED",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![],
            },
        ),
        (
            "FV_REGION_INTERFACE_UNKNOWN",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![interface("contact", "solid", "gas")],
            },
        ),
        (
            "FV_REGION_INTERFACE_UNKNOWN",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![interface("contact", "solid", "solid")],
            },
        ),
        (
            "FV_REGION_INTERFACE_DUPLICATE",
            TensorRegionPartition {
                regions: vec![region("solid", [0, 4], [0, 1]), region("fluid", [0, 4], [1, 3])],
                interfaces: vec![contact(), interface("again", "fluid", "solid")],
            },
        ),
        (
            "FV_REGION_INTERFACE_EMPTY",
            TensorRegionPartition {
                regions: vec![
                    region("solid", [0, 4], [0, 1]),
                    region("fluid", [0, 4], [1, 2]),
                    region("gas", [0, 4], [2, 3]),
                ],
                interfaces: vec![
                    contact(),
                    interface("surface", "fluid", "gas"),
                    interface("never", "solid", "gas"),
                ],
            },
        ),
    ];
    for (expected, partition) in cases {
        assert_eq!(refusal(partition), expected);
    }
}

#[test]
fn two_region_steady_conduction_is_exact_with_harmonic_averaging() {
    // k = 1 on [0, 0.5], k = 4 on [0.5, 1], u(0) = 400 K, u(1) = 300 K: the series resistance is
    // 0.5/1 + 0.5/4 = 0.625, the flux 160 (W/m²-equivalent) and the interface temperature
    // 400 − 160·0.5 = 320 K. TPFA with harmonic face averaging is exact for piecewise-constant k
    // in 1-D when the jump lies on a face, so cell values equal the piecewise-linear profile at
    // the centroids to the round-off of the dense steady solve. Predeclared bound:
    // 16·n·ε·cond(J)·T_hot with cond(J) ≤ 4n²/π² for the 1-D Laplacian, i.e. 6.5·n³·ε·400.
    let (flux, interface) = (160.0, 320.0);
    let exact = |x: f64| {
        if x < 0.5 {
            400.0 - flux * x
        } else {
            interface - flux / 4.0 * (x - 0.5)
        }
    };
    let data = |extra: &[(&str, FvDatumValue)]| {
        let mut data = uniform(&[("a", 0.0), ("t_hot", 400.0), ("t_cold", 300.0)]);
        data.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
        data
    };
    let mut arithmetic_errors = Vec::new();
    for cells in [2usize, 4, 10, 16, 50] {
        let (_, topology) = split_line(cells, false);
        let conduction = |averaging: FvFaceAveraging| {
            bind(
                REGIONS,
                "Conduction1D",
                NumericalFluxIntent::Central,
                &topology,
                &data(&[]),
            )
            .unwrap()
            .with_region_diffusion(FvRegionDiffusion {
                coefficients: tags(&[("left", 1.0), ("right", 4.0)]),
                averaging,
            })
            .unwrap()
        };
        let bound = 6.5 * (cells as f64).powi(3) * f64::EPSILON * 400.0;
        let harmonic = conduction(FvFaceAveraging::Harmonic);
        let solution = steady(&harmonic);
        let centroids = topology.cell_centroids();
        let error = solution
            .iter()
            .zip(centroids)
            .map(|(u, x)| (u - exact(*x)).abs())
            .fold(0.0, f64::max);
        assert!(error <= bound, "{cells} cells: error {error:e} > {bound:e}");
        // The interface flux and temperature from the realization's own face fluxes.
        let fluxes = harmonic.face_fluxes(0.0, &solution).unwrap();
        let contact = &topology.interfaces()[&RegionTagId::new("contact")];
        let face = contact.interior[0];
        assert!((fluxes.interior[face] - flux).abs() <= bound * flux / 400.0);
        let (minus, plus) = (topology.interior().minus[face], topology.interior().plus[face]);
        let h = 1.0 / cells as f64;
        // Linear between the two centroids with the harmonic mean: T(0.5) from either side.
        let from_left = solution[minus] - flux * h / 2.0;
        let from_right = solution[plus] + flux / 4.0 * h / 2.0;
        assert!((from_left - interface).abs() <= bound && (from_right - interface).abs() <= bound);
        // Boundary fluxes balance the interior one (steady state, no source).
        assert!(fluxes.boundary.values().all(|values| values.len() == 1));
        assert!((fluxes.boundary[&RegionTagId::new("x_min")][0] + flux).abs() <= bound);
        assert!((fluxes.boundary[&RegionTagId::new("x_max")][0] - flux).abs() <= bound);
        assert!(harmonic.conservation_receipt(0.0, &solution).unwrap().balanced);
        let receipt = harmonic.receipt();
        let diffusion = receipt.region_diffusion.as_ref().unwrap();
        assert_eq!(
            (diffusion.scheme.as_str(), diffusion.averaging),
            ("two_point", FvFaceAveraging::Harmonic)
        );
        assert_eq!(diffusion.coefficients, tags(&[("left", 1.0), ("right", 4.0)]));
        assert_eq!(diffusion.max_nonorthogonality, 0.0);
        assert!(receipt.diffusion.is_none() && receipt.region_data.is_empty());
        assert_eq!(receipt.max_nonorthogonality, Some(0.0));

        // Arithmetic averaging is not exact: its interface face conducts 2.5 instead of 1.6.
        let arithmetic = conduction(FvFaceAveraging::Arithmetic);
        assert_ne!(arithmetic.digest(), harmonic.digest());
        let solution = steady(&arithmetic);
        let error = solution
            .iter()
            .zip(centroids)
            .map(|(u, x)| (u - exact(*x)).abs())
            .fold(0.0, f64::max);
        let fluxes = arithmetic.face_fluxes(0.0, &solution).unwrap();
        // Exact resistance with the arithmetic face: (n/2 − 1/2)h/1 + h/2.5 + (n/2 − 1/2)h/4.
        let resistance = (cells as f64 / 2.0 - 0.5) * h * (1.0 + 0.25) + h / 2.5;
        let arithmetic_flux = 100.0 / resistance;
        assert!((fluxes.interior[face] - arithmetic_flux).abs() <= bound);
        assert!(error > 0.1, "{cells} cells: arithmetic error {error}");
        arithmetic_errors.push((cells, error, arithmetic_flux));
    }
    println!("arithmetic averaging (cells, max error K, interface flux): {arithmetic_errors:?}");
    // First-order convergence of the arithmetic error, recorded not required: halving h
    // between 10 and 50 cells (factor 5) must shrink the error at least threefold.
    let error_at = |cells: usize| arithmetic_errors.iter().find(|(c, _, _)| *c == cells).unwrap().1;
    assert!(error_at(50) * 3.0 <= error_at(10));
}

#[test]
fn per_region_source_and_face_data_match_hand_assembled_residuals_bitwise() {
    // 2-D periodic plane, per-region source q, uniform speeds.
    let spec = plane_spec(true);
    let topology = Arc::new(FvTopology::from_partitioned_tensor_box(&spec, &plane_partition()).unwrap());
    let (ax, ay) = (1.0, -0.5);
    let q = [("solid", 2.5), ("fluid", -0.75)];
    let mut data = uniform(&[("ax", ax), ("ay", ay)]);
    data.insert("q".into(), per_region(&q));
    let operator = bind(
        REGIONS,
        "SourcedAdvection2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &data,
    )
    .unwrap();
    let state = values(12, 17);
    let rate = values(12, 19);
    let mut observed = vec![0.0; 12];
    DaeOperator::residual(&operator, &cx(), 0.0, &state, &rate, &mut observed).unwrap();
    // Hand scheme in the realization's order: |Ω|u̇, interior faces (axis 0 rows, then axis 1),
    // periodic faces (axis 0 wraps, then axis 1), owner += |f|F*, neighbour −= |f|F*, then
    // −|Ω| q_region per cell.
    let volumes = topology.cell_volumes();
    let mut hand = rate.iter().zip(volumes).map(|(r, v)| v * r).collect::<Vec<_>>();
    let cell = |i: usize, j: usize| i + 4 * j;
    let (hx, hy) = (0.5, 2.0 / 3.0);
    let mut faces = Vec::new();
    for j in 0..3 {
        for i in 1..4 {
            faces.push((cell(i - 1, j), cell(i, j), hy, ax));
        }
    }
    for i in 0..4 {
        for j in 1..3 {
            faces.push((cell(i, j - 1), cell(i, j), hx, ay));
        }
    }
    for j in 0..3 {
        faces.push((cell(3, j), cell(0, j), hy, ax));
    }
    for i in 0..4 {
        faces.push((cell(i, 2), cell(i, 0), hx, ay));
    }
    for (minus, plus, area, speed) in faces {
        let term = area * hand_llf(speed, state[minus], state[plus], 1.0);
        hand[minus] += term;
        hand[plus] -= term;
    }
    for (index, (out, volume)) in hand.iter_mut().zip(volumes).enumerate() {
        let source = if index < 4 { q[0].1 } else { q[1].1 };
        *out -= volume * source;
    }
    assert_eq!(bytes_of(&observed), bytes_of(&hand));
    let receipt = operator.receipt();
    assert_eq!(receipt.region_data.len(), 1);
    assert_eq!(receipt.region_data[0].name, "q");
    assert_eq!(receipt.region_data[0].values, tags(&q));
    assert_eq!(operator.data()["q"], per_region(&q));
    assert_eq!(operator.data()["ax"], FvDatumValue::Uniform(ax));
    // The identity sees the per-region value.
    let mut other = data.clone();
    other.insert("q".into(), per_region(&[("solid", 2.5), ("fluid", -0.5)]));
    let changed = bind(REGIONS, "SourcedAdvection2D", NumericalFluxIntent::LocalLaxFriedrichs, &topology, &other).unwrap();
    assert_ne!(changed.digest(), operator.digest());
    // Mass rate: −Σ_b |f|F_b + Σ |Ω| S with no boundary is the integrated source.
    let mut rates = vec![0.0; 1];
    methodus::ExplicitOperator::invariant_rates(&operator, &cx(), 0.0, &state, &mut rates).unwrap();
    let expected = 4.0 * hx * hy * q[0].1 + 8.0 * hx * hy * q[1].1;
    assert!((rates[0] - expected).abs() <= 16.0 * f64::EPSILON * 10.0, "{rates:?}");

    // 1-D periodic advection with a per-region speed: every face kernel reads the owner
    // (minus) cell's value, so the hand LLF uses a(minus) on each face.
    let (_, ring) = split_line(6, true);
    let a = [("left", 1.5), ("right", -0.25)];
    let mut data = BTreeMap::new();
    data.insert("a".to_string(), per_region(&a));
    let operator = bind(SCALAR, "PeriodicAdvection1D", NumericalFluxIntent::LocalLaxFriedrichs, &ring, &data).unwrap();
    let state = values(6, 29);
    let observed = residual(&operator, &state);
    let speed_of = |cell: usize| if cell < 3 { a[0].1 } else { a[1].1 };
    let mut hand = vec![0.0; 6];
    let mut faces = (0..5).map(|i| (i, i + 1)).collect::<Vec<_>>();
    faces.push((5, 0));
    for (minus, plus) in faces {
        let term = hand_llf(speed_of(minus), state[minus], state[plus], 1.0);
        hand[minus] += term;
        hand[plus] -= term;
    }
    assert_eq!(bytes_of(&observed), bytes_of(&hand));
    // The CFL speed is likewise the owner's |a| on both face sides.
    let cfl = operator.cfl(0.0, &state).unwrap();
    assert_eq!(cfl.max_face_speed, 1.5);
    let mut sums = vec![0.0; 6];
    for (minus, plus) in (0..5).map(|i| (i, i + 1)).chain([(5, 0)]) {
        sums[minus] += speed_of(minus).abs();
        sums[plus] += speed_of(minus).abs();
    }
    let expected = sums.iter().map(|s| (1.0 / 6.0) / s).fold(f64::INFINITY, f64::min);
    assert_eq!(cfl.stable_step, expected);
    assert!(operator.conservation_receipt(0.0, &state).unwrap().balanced);
}

#[test]
fn per_region_binding_refusals() {
    let (_, split) = split_line(4, false);
    let plain = line(4, false);
    let base = || uniform(&[("a", 0.0), ("t_hot", 400.0), ("t_cold", 300.0)]);
    let with = |topology: &Arc<FvTopology>, name: &str, value: FvDatumValue| {
        let mut data = base();
        data.insert(name.into(), value);
        bind(REGIONS, "Conduction1D", NumericalFluxIntent::Central, topology, &data)
    };
    assert_eq!(
        code(&with(&plain, "a", per_region(&[("left", 1.0)])).unwrap_err()),
        "FV_DATUM_REGION_UNPARTITIONED"
    );
    assert_eq!(
        code(&with(&split, "a", per_region(&[("left", 1.0)])).unwrap_err()),
        "FV_DATUM_REGION_MISSING"
    );
    assert_eq!(
        code(&with(&split, "a", per_region(&[("left", 1.0), ("right", f64::NAN)])).unwrap_err()),
        "FV_DATUM_REGION_MISSING"
    );
    assert_eq!(
        code(&with(&split, "a", per_region(&[("left", 1.0), ("right", 1.0), ("gas", 1.0)])).unwrap_err()),
        "FV_DATUM_REGION_UNKNOWN"
    );
    assert_eq!(
        code(&with(&split, "a", FvDatumValue::Uniform(f64::INFINITY)).unwrap_err()),
        "FV_DATUM_MISSING"
    );
    assert_eq!(
        code(&with(&split, "b", FvDatumValue::Uniform(1.0)).unwrap_err()),
        "FV_DATUM_UNKNOWN"
    );
    // A per-region boundary datum binds the owner's value too (both boundaries are left/right).
    let hot = with(&split, "t_hot", per_region(&[("left", 400.0), ("right", 1.0)])).unwrap();
    let ghost = hot.face_fluxes(0.0, &[400.0; 4]).unwrap();
    assert_eq!(ghost.boundary[&RegionTagId::new("x_min")], vec![0.0]);

    let diffusion = |topology: &Arc<FvTopology>, coefficients: &[(&str, f64)], averaging| {
        with(topology, "a", FvDatumValue::Uniform(0.0))
            .unwrap()
            .with_region_diffusion(FvRegionDiffusion {
                coefficients: tags(coefficients),
                averaging,
            })
    };
    let harmonic = FvFaceAveraging::Harmonic;
    assert_eq!(
        code(&diffusion(&plain, &[("left", 1.0)], harmonic).unwrap_err()),
        "FV_DIFFUSION_REGION_UNPARTITIONED"
    );
    assert_eq!(
        code(&diffusion(&split, &[("left", 1.0)], harmonic).unwrap_err()),
        "FV_DIFFUSION_REGION_MISSING"
    );
    assert_eq!(
        code(&diffusion(&split, &[("left", 1.0), ("right", -4.0)], harmonic).unwrap_err()),
        "FV_DIFFUSION_REGION_MISSING"
    );
    assert_eq!(
        code(&diffusion(&split, &[("left", 1.0), ("right", 4.0), ("gas", 1.0)], harmonic).unwrap_err()),
        "FV_DIFFUSION_REGION_UNKNOWN"
    );
    let once = diffusion(&split, &[("left", 1.0), ("right", 4.0)], harmonic).unwrap();
    assert_eq!(
        code(&once.with_diffusion(FvDiffusion { coefficient: 1.0 }).unwrap_err()),
        "FV_DIFFUSION_INVALID"
    );
    let once = with(&split, "a", FvDatumValue::Uniform(0.0))
        .unwrap()
        .with_diffusion(FvDiffusion { coefficient: 1.0 })
        .unwrap();
    assert_eq!(
        code(
            &once
                .with_region_diffusion(FvRegionDiffusion {
                    coefficients: tags(&[("left", 1.0), ("right", 4.0)]),
                    averaging: harmonic,
                })
                .unwrap_err()
        ),
        "FV_DIFFUSION_INVALID"
    );
    // Zero conductivity on both sides of a face is a zero transmissibility, not a NaN.
    let insulated = diffusion(&split, &[("left", 0.0), ("right", 0.0)], harmonic).unwrap();
    let fluxes = insulated.face_fluxes(0.0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(fluxes.interior.iter().all(|flux| *flux == 0.0));
    assert_eq!(FvFaceAveraging::Harmonic.average(1.0, 4.0), 1.6);
    assert_eq!(FvFaceAveraging::Arithmetic.average(1.0, 4.0), 2.5);
}

fn round_trip<T>(value: &T, label: &str)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let text = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&text).unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_eq!(&back, value, "{label}");
    assert_eq!(serde_json::to_string(&back).unwrap(), text, "{label}");
}

#[test]
fn receipts_round_trip_through_serde() {
    // A regionwise realization: receipt with region data and region diffusion.
    let (_, split) = split_line(6, false);
    let mut data = uniform(&[("a", 0.0), ("t_hot", 400.0)]);
    data.insert("t_cold".into(), per_region(&[("left", 300.0), ("right", 300.0)]));
    let regionwise = bind(REGIONS, "Conduction1D", NumericalFluxIntent::Central, &split, &data)
        .unwrap()
        .with_region_diffusion(FvRegionDiffusion {
            coefficients: tags(&[("left", 1.0), ("right", 4.0)]),
            averaging: FvFaceAveraging::Harmonic,
        })
        .unwrap();
    let state = values(6, 3);
    round_trip::<FvRealizationReceipt>(&regionwise.receipt(), "regionwise receipt");
    round_trip::<FvCfl>(&regionwise.cfl(0.0, &state).unwrap(), "cfl");
    round_trip::<FvConservationReceipt>(
        &regionwise.conservation_receipt(0.0, &state).unwrap(),
        "conservation receipt",
    );
    round_trip::<FvTopologyInvariants>(&split.check_invariants().unwrap(), "invariants");
    round_trip(&regionwise.face_fluxes(0.0, &state).unwrap(), "face fluxes");
    // A uniform diffusive MUSCL realization: receipt with `muscl` and `diffusion`.
    let (system, program) = law(
        MUSCL,
        "DirichletAdvection1D",
        NumericalFluxIntent::Central,
        ReconstructionIntent::Muscl {
            variables: ReconstructionVariables::Conserved,
            gradient: GradientScheme::LeastSquares,
            limiter: LimiterFamily::Minmod,
        },
    );
    let muscl = FvRealization::new(
        &system,
        program,
        line(8, false),
        &regions(&system),
        &[("a".to_string(), 1.0)].into_iter().collect(),
    )
    .unwrap()
    .with_diffusion(FvDiffusion { coefficient: 0.3 })
    .unwrap();
    let receipt = muscl.receipt();
    assert!(receipt.muscl.is_some() && receipt.diffusion.is_some());
    round_trip::<FvRealizationReceipt>(&receipt, "muscl receipt");
    round_trip(receipt.muscl.as_ref().unwrap(), "reconstruction receipt");
    round_trip(receipt.diffusion.as_ref().unwrap(), "diffusion receipt");
    round_trip(&receipt.kernels, "kernel receipts");
    // An unconstrained CFL carries `+∞` and no limiting cell: JSON has no infinity.
    let still = bind(
        SCALAR,
        "PeriodicAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &line(4, true),
        &uniform(&[("a", 0.0)]),
    )
    .unwrap();
    let cfl = still.cfl(0.0, &[0.0; 4]).unwrap();
    assert_eq!(cfl.stable_step, f64::INFINITY);
    assert!(serde_json::to_string(&cfl).unwrap().contains("null"));
    // The W10 request: PlanarCoverageReport reads back from its own serialization.
    round_trip(
        &PlanarCoverageReport {
            area: [0.5, 0.5],
            intersection_total: 0.5,
            worst_relative_defect: 1e-15,
            worst_constant_defect: 2e-16,
            relative_tolerance: 1e-9,
            area_floor: 1e-12,
            pairs: 7,
            slivers_classified: 1,
        },
        "planar coverage",
    );
}
