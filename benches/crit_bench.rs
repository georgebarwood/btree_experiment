use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

criterion_group!(
    benches,
    bench_get,
    bench_clone,
    bench_ref_iter,
    bench_into_iter
);
criterion_main!(benches);

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clone");
    for n in [1000, 10000].iter() {
        let mut exp_map = btree_experiment::BTreeMap::new();
        for i in 0..*n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..*n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| b.iter(|| exp_map.clone()));
        group.bench_function(BenchmarkId::new("Std", n), |b| b.iter(|| std_map.clone()));
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("Get");
    for n in [50, 100, 200, 500, 1000].iter() {
        let n = *n;
        let mut exp_map = btree_experiment::BTreeMap::new();
        for i in 0..n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                for i in 0..n {
                    assert!(exp_map.get(&i).unwrap() == &i);
                }
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                for i in 0..n {
                    assert!(std_map.get(&i).unwrap() == &i);
                }
            })
        });
    }
    group.finish();
}

fn bench_ref_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("RefIter");
    for n in [100, 1000, 10000, 100000].iter() {
        let mut exp_map = btree_experiment::BTreeMap::new();
        for i in 0..*n {
            exp_map.insert(i, i);
        }

        let mut std_map = std::collections::BTreeMap::new();
        for i in 0..*n {
            std_map.insert(i, i);
        }

        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                for (k, v) in exp_map.iter() {
                    assert!(k == v);
                }
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                for (k, v) in std_map.iter() {
                    assert!(k == v);
                }
            })
        });
    }
    group.finish();
}

fn exp_into_iter_test(n: usize) {
    let mut m = btree_experiment::BTreeMap::<usize, usize>::default();
    for i in 0..n {
        m.insert(i, i);
    }
    for (k, v) in m {
        assert!(k == v);
    }
}

fn std_into_iter_test(n: usize) {
    let mut m = std::collections::BTreeMap::<usize, usize>::default();
    for i in 0..n {
        m.insert(i, i);
    }
    for (k, v) in m {
        assert!(k == v);
    }
}

fn bench_into_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("IntoIter");
    for n in [100, 1000, 10000].iter() {
        group.bench_function(BenchmarkId::new("Exp", n), |b| {
            b.iter(|| {
                exp_into_iter_test(*n);
            })
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| {
                std_into_iter_test(*n);
            })
        });
    }
    group.finish();
}

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
