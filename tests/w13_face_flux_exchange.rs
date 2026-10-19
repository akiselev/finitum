//! W13 A4 (SV2C L9, gate 1c precondition): `FaceFluxExchange` between an FV boundary face set
//! and a P1 FE exterior-facet set on a matching interface. The coupled FV slab (k = 1) + P1
//! solid (k = 4) problem is the two-material conduction case, solved monolithically here the
//! way Krasis composes the two leaves: `[R_fv + X_fv ; R_fe + X_fe] = 0`.
use std::collections::BTreeMap;
use std::sync::Arc;

use finitum::{
    BlockLayout, FaceFluxExchange, FaceFluxExchangeLaw, FieldSource, FinitumError, FvRealization,
    FvRegionProperties, FvRegions, FvTopology, MeshProfile, PointEvaluation, RegionMap,
    RegionTagId, SystemConstitutiveInput, SystemOperator, SystemRealizationPlan, TaggedMesh,
    TensorAxis, TensorBox, realize,
};
use methodus::{DaeOperator, EvaluationContext};
use quantitas::UnitRegistry;
use scientia::{
    BoundaryFluxSemantics, DerivativeEvaluation, FvMethodIntent, InputSourceRequirement,
    NumericalFluxIntent, OperatorSystem, PositivityIntent, ReconstructionIntent, SourceTreatment,
    TensorInputRole, compile_conservation_law_system, compile_operator_system, compile_semantics,
    lower_conservation_law,
};

const CHT: &str = include_str!("fixtures/fv-cht.res");
const SOLID: &str = include_str!("fixtures/fe-solid.res");

fn cx() -> EvaluationContext {
    EvaluationContext::reproducible()
}

/// FV slab [0, 0.5]×[0, 1], k = 1 (one region), 400 K at x = 0, adiabatic walls, the
/// interface (x = 0.5) a zero prescribed flux the exchange supplies.
fn fluid(nx: usize, ny: usize) -> FvRealization {
    let module = compile_semantics(CHT, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|d| panic!("{d:#?}"))
        .semantic;
    let system =
        compile_conservation_law_system(&module, "ExchangeFluid2D", "balance", "T").unwrap();
    let intent = FvMethodIntent {
        numerical_flux: NumericalFluxIntent::LocalLaxFriedrichs,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = Arc::new(lower_conservation_law(&module, &system, &intent, &[]).unwrap());
    let mut map = RegionMap::new();
    for boundary in &system.boundaries {
        let (BoundaryFluxSemantics::GhostState { region, .. }
        | BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tags: &[&str] = match boundary.region_name() {
            "west" => &["x_min"],
            "interface" => &["x_max"],
            "walls" => &["y_min", "y_max"],
            other => panic!("{other}"),
        };
        map.insert(*region, tags.iter().map(|t| RegionTagId::new(*t)));
    }
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::uniform(0.0, 0.5, nx),
                TensorAxis::uniform(0.0, 1.0, ny),
            ],
        })
        .unwrap(),
    );
    FvRealization::with_face_fields(
        &system,
        program,
        topology.clone(),
        &map,
        &BTreeMap::from([("t_west".to_string(), 400.0)]),
        &BTreeMap::from([
            ("ux".to_string(), FieldSource::constant([0.0])),
            ("uy".to_string(), FieldSource::constant([0.0])),
        ]),
    )
    .unwrap()
    .with_regions(FvRegions {
        cell_regions: vec![RegionTagId::new("fluid"); topology.cell_count()],
        properties: BTreeMap::from([(
            RegionTagId::new("fluid"),
            FvRegionProperties {
                conductivity: 1.0,
                capacity: 1.0,
                stagnant: true,
            },
        )]),
    })
    .unwrap()
}

/// `q = −k grad T` as the single constitutive input of the P1 solid.
fn constant_flux(model: &OperatorSystem, conductivity: f64) -> Vec<SystemConstitutiveInput> {
    let t_symbol = model.field_order[0];
    let mut built = Vec::new();
    for block in &model.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let grad_t = integral
                    .primal
                    .inputs
                    .iter()
                    .find(|candidate| {
                        candidate.source == InputSourceRequirement::Basis
                            && candidate.role == TensorInputRole::Active
                            && candidate.binding.symbol == t_symbol
                            && candidate.binding.evaluation.derivative
                                == DerivativeEvaluation::Gradient
                    })
                    .unwrap()
                    .id;
                built.push(
                    SystemConstitutiveInput::try_new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        2,
                        "w13_face_flux_exchange/q",
                        move |point: &PointEvaluation| {
                            Ok(point
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|g| -conductivity * g)
                                .collect())
                        },
                        move |_point: &PointEvaluation, direction: &PointEvaluation| {
                            Ok(direction
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|dg| -conductivity * dg)
                                .collect())
                        },
                    )
                    .unwrap(),
                );
            }
        }
    }
    built
}

/// P1 solid on [0.5, 1]×[0, 1], k = 4, natural boundary everywhere (Dirichlet rows are set by
/// the coupled test operator).
fn solid(subdivisions: [usize; 2]) -> (TaggedMesh, SystemOperator) {
    let compiled = compile_semantics(SOLID, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|d| panic!("{d:#?}"));
    let model = compile_operator_system(&compiled.semantic, "Solid", &["energy"]).unwrap();
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.5, 1.0], [0.0, 1.0]],
        subdivisions: subdivisions.to_vec(),
    })
    .unwrap();
    let layout =
        BlockLayout::new(vec![(model.field_order[0], mesh.mesh.vertices().len(), 1)]).unwrap();
    let flux = constant_flux(&model, 4.0);
    let plan = SystemRealizationPlan::new(model, mesh.mesh.clone(), layout).unwrap();
    (mesh, plan.bind_kernels(flux, BTreeMap::new()).unwrap())
}

struct Coupled<'a> {
    fv: &'a FvRealization,
    fe: &'a SystemOperator,
    exchange: &'a FaceFluxExchange,
    /// FE vertices on x = 1 (Dirichlet 300 K rows).
    fixed: Vec<usize>,
}

impl Coupled<'_> {
    fn split(&self) -> usize {
        DaeOperator::dimension(self.fv)
    }

    fn residual(&self, state: &[f64]) -> Vec<f64> {
        let n = self.split();
        let (fv, fe) = state.split_at(n);
        let mut out = vec![0.0; state.len()];
        let (fv_out, fe_out) = out.split_at_mut(n);
        DaeOperator::residual(self.fv, &cx(), 0.0, fv, &vec![0.0; n], fv_out).unwrap();
        self.fe
            .residual(0.0, fe, &vec![0.0; fe.len()], fe_out)
            .unwrap();
        self.exchange.residual(fv, fe, fv_out, fe_out).unwrap();
        for vertex in &self.fixed {
            fe_out[*vertex] = fe[*vertex] - 300.0;
        }
        out
    }

    /// Dense solve of the (linear) coupled problem.
    fn solve(&self) -> Vec<f64> {
        let total = self.split() + self.exchange.fe_dimension();
        let zero = vec![0.0; total];
        let b = self.residual(&zero);
        let mut matrix = vec![vec![0.0; total + 1]; total];
        for column in 0..total {
            let mut e = vec![0.0; total];
            e[column] = 1.0;
            let r = self.residual(&e);
            for row in 0..total {
                matrix[row][column] = r[row] - b[row];
            }
        }
        for row in 0..total {
            matrix[row][total] = -b[row];
        }
        for pivot in 0..total {
            let best = (pivot..total)
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
        let mut solution = vec![0.0; total];
        for row in (0..total).rev() {
            let tail = (row + 1..total)
                .map(|c| matrix[row][c] * solution[c])
                .sum::<f64>();
            solution[row] = (matrix[row][total] - tail) / matrix[row][row];
        }
        solution
    }
}

fn code(error: &FinitumError) -> String {
    let message = error.to_string();
    message
        .split(':')
        .find(|part| part.trim_start().starts_with("FV_"))
        .map(|part| part.trim().to_string())
        .unwrap_or(message)
}

fn interface_facets(mesh: &TaggedMesh) -> Vec<finitum::FacetId> {
    mesh.tags.facet_regions[&RegionTagId::new("x_min")].clone()
}

#[test]
fn dirichlet_neumann_exchange_reproduces_two_material_conduction_with_interface_balance() {
    // Bars (fixed before running): 1e-9 K and 1e-9 W/m². Basis: FV TPFA (half-cell against the
    // P1 trace) and P1 are both exact for the piecewise-linear solution, the matching facets
    // carry the same P1 trace; the dense solve of ~50 unknowns at 400 K with condition ≲ 1e3
    // leaves ≲ 1e-10. InterfaceBalance within its 4·n_v·ε bound.
    let fv = fluid(4, 4);
    let (mesh, fe) = solid([3, 4]);
    let facets = interface_facets(&mesh);
    let exchange = FaceFluxExchange::new(
        &fv,
        &RegionTagId::new("x_max"),
        &mesh.mesh,
        &facets,
        FaceFluxExchangeLaw::DirichletNeumann,
        1e-12,
    )
    .unwrap();
    assert_eq!(exchange.faces(), 4);
    assert_eq!(exchange.receipt().enforcement, "weak_flux");
    let fixed = mesh
        .mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, p)| p[0] == 1.0)
        .map(|(v, _)| v)
        .collect::<Vec<_>>();
    let coupled = Coupled {
        fv: &fv,
        fe: &fe,
        exchange: &exchange,
        fixed,
    };
    let solution = coupled.solve();
    let (fv_state, fe_state) = solution.split_at(coupled.split());
    for (cell, value) in fv_state.iter().enumerate() {
        let x = fv.topology().cell_centroids()[2 * cell];
        assert!(
            (value - (400.0 - 160.0 * x)).abs() <= 1e-9,
            "FV {x}: {value}"
        );
    }
    for (vertex, value) in fe_state.iter().enumerate() {
        let x = mesh.mesh.vertices()[vertex][0];
        assert!(
            (value - (320.0 - 40.0 * (x - 0.5))).abs() <= 1e-9,
            "FE {x}: {value}"
        );
    }
    for trace in exchange.fe_trace(fe_state).unwrap() {
        assert!((trace - 320.0).abs() <= 1e-9, "interface {trace}");
    }
    for flux in exchange.face_fluxes(fv_state, fe_state).unwrap() {
        assert!((flux - 160.0).abs() <= 1e-9, "flux {flux}");
    }
    let balance = exchange.balance(fv_state, fe_state).unwrap();
    assert!(balance.balanced, "{balance:?}");
    assert!((balance.fv_flux_sum - 160.0).abs() <= 1e-9);
    // P1 traces on a 2-node facet: the halves are exact, so the balance is exact here.
    assert_eq!(balance.imbalance, 0.0);
    let residual = coupled.residual(&solution);
    assert!(residual.iter().all(|r| r.abs() <= 1e-9), "{residual:?}");
}

#[test]
fn robin_exchange_matches_the_series_resistance() {
    // q = h (T_P − T_s), h = 10: the discrete solution is exact for the series resistance
    // x_P/k_f + 1/h + 0.5/k_s with x_P the last FV centroid (0.4375). Bar 1e-9 (as above).
    let fv = fluid(4, 2);
    let (mesh, fe) = solid([2, 2]);
    let exchange = FaceFluxExchange::new(
        &fv,
        &RegionTagId::new("x_max"),
        &mesh.mesh,
        &interface_facets(&mesh),
        FaceFluxExchangeLaw::Robin { coefficient: 10.0 },
        1e-12,
    )
    .unwrap();
    let fixed = mesh
        .mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, p)| p[0] == 1.0)
        .map(|(v, _)| v)
        .collect();
    let coupled = Coupled {
        fv: &fv,
        fe: &fe,
        exchange: &exchange,
        fixed,
    };
    let solution = coupled.solve();
    let (fv_state, fe_state) = solution.split_at(coupled.split());
    let q = 100.0 / (0.4375 + 0.1 + 0.125);
    for flux in exchange.face_fluxes(fv_state, fe_state).unwrap() {
        assert!((flux - q).abs() <= 1e-9, "{flux} vs {q}");
    }
    for trace in exchange.fe_trace(fe_state).unwrap() {
        assert!((trace - (300.0 + q * 0.125)).abs() <= 1e-9);
    }
    assert!(exchange.balance(fv_state, fe_state).unwrap().balanced);
}

#[test]
fn exchange_jvp_and_vjp_are_exact_transposes() {
    // Bar 64ε·scale: the VJP is the hand transpose of a linear map.
    let fv = fluid(4, 4);
    let (mesh, _) = solid([3, 4]);
    let exchange = FaceFluxExchange::new(
        &fv,
        &RegionTagId::new("x_max"),
        &mesh.mesh,
        &interface_facets(&mesh),
        FaceFluxExchangeLaw::DirichletNeumann,
        1e-12,
    )
    .unwrap();
    let (n, m) = (exchange.fv_dimension(), exchange.fe_dimension());
    let probe = |len: usize, seed: f64| {
        (0..len)
            .map(|i| ((i as f64 + seed) * 0.618_034).sin())
            .collect::<Vec<_>>()
    };
    let (vf, ve, wf, we) = (probe(n, 0.1), probe(m, 0.7), probe(n, 1.3), probe(m, 2.9));
    let (mut jf, mut je) = (vec![0.0; n], vec![0.0; m]);
    exchange
        .jacobian_vector_product(&vf, &ve, &mut jf, &mut je)
        .unwrap();
    let (mut tf, mut te) = (vec![0.0; n], vec![0.0; m]);
    exchange
        .vector_jacobian_product(&wf, &we, &mut tf, &mut te)
        .unwrap();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let left = dot(&wf, &jf) + dot(&we, &je);
    let right = dot(&tf, &vf) + dot(&te, &ve);
    let scale = wf
        .iter()
        .zip(&jf)
        .chain(we.iter().zip(&je))
        .map(|(a, b)| (a * b).abs())
        .sum::<f64>();
    assert!(
        (left - right).abs() <= 64.0 * f64::EPSILON * scale,
        "{left} {right}"
    );
    // The JVP equals the residual map of the direction (linear exchange).
    let (mut rf, mut re) = (vec![0.0; n], vec![0.0; m]);
    exchange.residual(&vf, &ve, &mut rf, &mut re).unwrap();
    assert_eq!((rf, re), (jf, je));
}

#[test]
fn nonmatching_and_misbound_interfaces_are_refused() {
    let fv = fluid(4, 4);
    let tag = RegionTagId::new("x_max");
    let (coarse, _) = solid([3, 3]);
    let error = FaceFluxExchange::new(
        &fv,
        &tag,
        &coarse.mesh,
        &interface_facets(&coarse),
        FaceFluxExchangeLaw::DirichletNeumann,
        1e-12,
    )
    .unwrap_err();
    assert_eq!(code(&error), "FV_EXCHANGE_NONMATCHING");
    let (mesh, _) = solid([3, 4]);
    let facets = interface_facets(&mesh);
    // Same count, wrong facets (the far side x = 1 has outward normal +x, coincident nowhere).
    let far = mesh.tags.facet_regions[&RegionTagId::new("x_max")].clone();
    let refusals = [
        FaceFluxExchange::new(
            &fv,
            &tag,
            &mesh.mesh,
            &far,
            FaceFluxExchangeLaw::DirichletNeumann,
            1e-12,
        ),
        FaceFluxExchange::new(
            &fv,
            &tag,
            &mesh.mesh,
            &facets[..3],
            FaceFluxExchangeLaw::DirichletNeumann,
            1e-12,
        ),
        FaceFluxExchange::new(
            &fv,
            &tag,
            &mesh.mesh,
            &[facets[0], facets[0], facets[1], facets[2]],
            FaceFluxExchangeLaw::DirichletNeumann,
            1e-12,
        ),
    ];
    for result in refusals {
        assert_eq!(code(&result.unwrap_err()), "FV_EXCHANGE_NONMATCHING");
    }
    // The Dirichlet west boundary is not a prescribed-flux placeholder.
    let error = FaceFluxExchange::new(
        &fv,
        &RegionTagId::new("x_min"),
        &mesh.mesh,
        &facets,
        FaceFluxExchangeLaw::DirichletNeumann,
        1e-12,
    )
    .unwrap_err();
    assert_eq!(code(&error), "FV_EXCHANGE_BOUNDARY");
    let error = FaceFluxExchange::new(
        &fv,
        &tag,
        &mesh.mesh,
        &facets,
        FaceFluxExchangeLaw::Robin { coefficient: 0.0 },
        1e-12,
    )
    .unwrap_err();
    assert_eq!(code(&error), "FV_EXCHANGE_LAW");
    let exchange = FaceFluxExchange::new(
        &fv,
        &tag,
        &mesh.mesh,
        &facets,
        FaceFluxExchangeLaw::DirichletNeumann,
        1e-12,
    )
    .unwrap();
    let error = exchange
        .face_fluxes(&[0.0; 3], &vec![0.0; exchange.fe_dimension()])
        .unwrap_err();
    assert_eq!(code(&error), "FV_EXCHANGE_SHAPE");
}
