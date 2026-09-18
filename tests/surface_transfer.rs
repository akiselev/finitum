use finitum::SurfaceTransfer;
#[test]
fn nonmatching_surface_preserves_affine_traces_and_dual_heat_or_mechanical_load() {
    // Tilted 3-D surface; target nodes do not coincide with the source triangulation.
    let vertices = vec![[0., 0., 0.], [1., 0., 1.], [1., 1., 2.], [0., 1., 1.]];
    let triangles = vec![[0, 1, 2], [0, 2, 3]];
    let points = vec![[0.2, 0.3, 0.5], [0.7, 0.1, 0.8], [0.5, 0.5, 1.0]];
    let p =
        SurfaceTransfer::p1(vertices.clone(), triangles.clone(), points.clone(), 1e-12).unwrap();
    assert!(p.apply_transpose(&[f64::MAX; 3]).is_err());
    for affine in [[300., 12., -3., 4.], [0.01, 0.2, -0.3, 0.1]] {
        let f = |x: &[f64; 3]| affine[0] + affine[1] * x[0] + affine[2] * x[1] + affine[3] * x[2];
        let u: Vec<_> = vertices.iter().map(f).collect();
        let target = p.apply(&u).unwrap();
        for (value, point) in target.iter().zip(&points) {
            assert!((value - f(point)).abs() < 1e-12);
        }
        let load = [2.0, -0.3, 4.0];
        let dual = p.apply_transpose(&load).unwrap();
        let lhs: f64 = target.iter().zip(load).map(|(a, b)| a * b).sum();
        let rhs: f64 = u.iter().zip(&dual).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-11);
        assert!((dual.iter().sum::<f64>() - load.iter().sum::<f64>()).abs() < 1e-12);
    }
    assert!(
        SurfaceTransfer::p1(
            vertices.clone(),
            triangles.clone(),
            vec![[0.2, 0.3, 0.501]],
            1e-12
        )
        .is_err()
    );
    assert!(
        SurfaceTransfer::p1(
            vertices.clone(),
            triangles.clone(),
            vec![[2., 2., 4.]],
            1e-12
        )
        .is_err()
    );
    assert!(SurfaceTransfer::p1(vertices.clone(), vec![[0, 0, 1]], points.clone(), 1e-12).is_err());
    let mut overlapping = vertices.clone();
    overlapping.extend(vertices);
    assert!(
        SurfaceTransfer::p1(
            overlapping,
            vec![[0, 1, 2], [4, 5, 6]],
            vec![[0.7, 0.1, 0.8]],
            1e-12
        )
        .is_err()
    );
}
