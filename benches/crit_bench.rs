use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

criterion_group!(benches, bench_get, bench_clone);
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
            b.iter(|| exp_map.get(&(n - 100)))
        });
        group.bench_function(BenchmarkId::new("Std", n), |b| {
            b.iter(|| std_map.get(&(n - 100)))
        });
    }
    group.finish();
}

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
