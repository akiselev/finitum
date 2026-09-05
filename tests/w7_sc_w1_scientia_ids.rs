//! W7 / HANDOFF §6 Finitum items: `SystemIdMap` built from Scientia's
//! `scientia-operator-system/2` `SystemOperator` (ids by value, artifact digests from
//! `instance_artifacts`), proven equal to Finitum's own one-instance identity map and to its
//! dense `compose` allocation; and the typed `@inf_sup` obligation consumed as an
//! `InfSupPairing`, proven equal to the structural derivation on the corpus Stokes and Darcy
//! models.

use finitum::{FinitumError, InfSupPairing, InstanceId, SysResId, SysVarId, SystemIdMap};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    Registries, SymbolId, VerificationObligationKind, compile_model_system,
    compile_operator_system, compile_semantics, compile_system, compile_system_operator,
    derive_operator_structure_for_system, derive_verification_profiles, resolve_module_closure,
};
use std::collections::BTreeMap;

const POISSON: &str = include_str!("fixtures/corpus/01-poisson.res");
const STOKES: &str = include_str!("fixtures/corpus/25-stokes.res");
const DARCY: &str = include_str!("fixtures/corpus/13-mixed-darcy.res");

const THERMAL: &str = r#"
module physics.thermal;

pub model HeatConduction {
    domain body { dimension = 2; coordinates = cartesian; }
    field T: state scalar H1(order=1) on body {
        quantity = ThermodynamicTemperature;
        unit = K;
        nominal = 300 K;
        time_role = differential;
    };
    input field Q: VolumetricHeatSource on body;
    input value ambient: ThermodynamicTemperature;
    provider density(T: ThermodynamicTemperature) -> Density { differentiability = symbolic; }
    provider specific_heat(T: ThermodynamicTemperature) -> SpecificHeat { differentiability = symbolic; }
    provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity { differentiability = symbolic; }
    property rho = density(T);
    property cp = specific_heat(T);
    property k = thermal_conductivity(T);
    equation thermal on body { rho * cp * dt(T) - div(k * grad(T)) = Q; }
    initial { T = ambient; }
    output temperature: ThermodynamicTemperature on body = T;
}
"#;

const TWO_HEAT: &str = r#"
module systems.two_heat;
use physics.thermal.{HeatConduction};

pub system TwoHeat {
    domain body { dimension = 2; coordinates = cartesian; }
    instance a: HeatConduction(body = body);
    instance b: HeatConduction(body = body);
}
"#;

fn registries() -> (UnitRegistry, QuantityKindRegistry) {
    (
        UnitRegistry::si_bootstrap(),
        QuantityKindRegistry::si_bootstrap(),
    )
}

#[test]
fn implicit_one_instance_system_ids_from_scientia_equal_finitums_identity_map() {
    let closure = resolve_module_closure(POISSON, &BTreeMap::<String, String>::new()).unwrap();
    let (units, kinds) = registries();
    let compilation =
        compile_model_system(&closure, Registries::new(&units, &kinds), "Poisson").unwrap();
    let operator = compile_system_operator(&compilation).unwrap();
    let from_scientia = SystemIdMap::from_scientia(&operator.operator).unwrap();

    let [(_, model_system)] = operator.model_systems.as_slice() else {
        panic!("one instance");
    };
    let one_instance = SystemIdMap::one_instance(model_system).unwrap();
    assert_eq!(from_scientia, one_instance);
    assert_eq!(from_scientia.identity(), one_instance.identity());
    let unknown = model_system.field_order[0];
    assert_eq!(
        from_scientia.variable(InstanceId(0), unknown),
        Some(SysVarId(unknown.0))
    );
    assert_eq!(
        from_scientia.residual(InstanceId(0), "balance"),
        Some(SysResId(0))
    );
    assert_eq!(
        from_scientia.instances()[0].artifact_digest,
        model_system.artifact_digest
    );
    assert_eq!(from_scientia.instances()[0].name, "Poisson");

    // The same `/1` artifact Finitum's single-model tests compile directly.
    let direct = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let direct_system = compile_operator_system(&direct.semantic, "Poisson", &["balance"]).unwrap();
    assert_eq!(direct_system.artifact_digest, model_system.artifact_digest);

    // A missing artifact digest is refused typed.
    let mut stripped = operator.operator.clone();
    stripped.instance_artifacts.clear();
    assert!(matches!(
        SystemIdMap::from_scientia(&stripped),
        Err(FinitumError::ArtifactMismatch(_))
    ));
}

#[test]
fn declared_two_instance_system_ids_from_scientia_equal_finitums_dense_composition() {
    let modules = BTreeMap::from([("physics.thermal".to_string(), THERMAL.to_string())]);
    let closure = resolve_module_closure(TWO_HEAT, &modules).unwrap();
    let (units, kinds) = registries();
    let compilation = compile_system(&closure, Registries::new(&units, &kinds), "TwoHeat").unwrap();
    let operator = compile_system_operator(&compilation).unwrap();
    let from_scientia = SystemIdMap::from_scientia(&operator.operator).unwrap();

    let [(_, a), (_, b)] = operator.model_systems.as_slice() else {
        panic!("two instances");
    };
    let composed = SystemIdMap::compose(&[("a", a), ("b", b)]).unwrap();
    assert_eq!(
        from_scientia, composed,
        "Scientia's dense allocation (instance order, then local SymbolId order) is Finitum's"
    );
    assert_eq!(from_scientia.identity(), composed.identity());
    assert_eq!(from_scientia.instances().len(), 2);
    assert_eq!(from_scientia.instances()[1].name, "b");
    assert_eq!(
        from_scientia.instances()[1].artifact_digest,
        b.artifact_digest
    );
    assert_eq!(from_scientia.variables().len(), 2);
    assert_eq!(from_scientia.residuals().len(), 2);
    let t = a.field_order[0];
    assert_eq!(from_scientia.variable(InstanceId(0), t), Some(SysVarId(0)));
    assert_eq!(from_scientia.variable(InstanceId(1), t), Some(SysVarId(1)));
    assert_eq!(
        from_scientia.residual(InstanceId(1), "thermal"),
        Some(SysResId(1))
    );
    assert_eq!(
        from_scientia.residual_path(SysResId(1)).as_deref(),
        Some("b.thermal")
    );
    // Scientia's own residual rows are the per-model row symbols Finitum keys by.
    for residual in &operator.operator.residuals {
        let origin = from_scientia
            .residual_origin(SysResId(residual.id.0))
            .unwrap();
        assert_eq!(origin.row, residual.row);
    }
}

fn inf_sup_obligation(source: &str, model: &str) -> VerificationObligationKind {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    derive_verification_profiles(&compilation)
        .into_iter()
        .find(|profile| profile.model == model)
        .unwrap_or_else(|| panic!("{model} profile"))
        .obligations
        .into_iter()
        .map(|obligation| obligation.kind)
        .find(|kind| matches!(kind, VerificationObligationKind::InfSup { .. }))
        .unwrap_or_else(|| panic!("{model} declares @inf_sup"))
}

#[test]
fn typed_inf_sup_obligations_bind_the_pairing_finitum_derives_structurally() {
    for (source, model, equations) in [
        (STOKES, "StokesFlow", ["momentum", "incompressibility"]),
        (DARCY, "MixedDarcy", ["darcy_law", "mass_balance"]),
    ] {
        let kind = inf_sup_obligation(source, model);
        let typed = InfSupPairing::from_obligation(&kind).unwrap();
        let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
        let system = compile_operator_system(&compilation.semantic, model, &equations).unwrap();
        let structure = derive_operator_structure_for_system(&system, None).unwrap();
        let structural = InfSupPairing::from_structure(&structure).unwrap();
        assert_eq!(
            typed, structural,
            "{model}: typed pairing = structural pairing"
        );
        let VerificationObligationKind::InfSup { pair, .. } = &kind else {
            unreachable!()
        };
        assert!(!pair.is_empty());
    }
    // Refusals: a non-inf-sup obligation, and an undecided pairing.
    assert!(matches!(
        InfSupPairing::from_obligation(&VerificationObligationKind::DerivativeTaylor {
            block: None,
            active_inputs: vec![SymbolId(0)],
        }),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        InfSupPairing::from_obligation(&VerificationObligationKind::InfSup {
            pair: "P1-P1".into(),
            constrained: Some(SymbolId(0)),
            multiplier: None,
        }),
        Err(FinitumError::UnsupportedRealization(_))
    ));
    assert!(matches!(
        InfSupPairing::from_obligation(&VerificationObligationKind::InfSup {
            pair: "same".into(),
            constrained: Some(SymbolId(3)),
            multiplier: Some(SymbolId(3)),
        }),
        Err(FinitumError::InvalidRealization(_))
    ));
}
