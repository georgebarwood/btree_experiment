use crate::*;

#[test]
fn exp_cursor_remove_rev_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = /*std::collections::*/ BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            c.insert_before(i, i).unwrap();
        }
        assert!(m.len() == n);

        let mut c = m.upper_bound_mut(Bound::Unbounded);
        let mut i = n;
        while let Some((k, v)) = c.remove_prev() {
            i -= 1;
            assert_eq!((k, v), (i, i));
            // println!("Ok, i={}", i);
        }
        assert_eq!(i, 0);
    }
}

#[test]
fn std_cursor_remove_rev_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = std::collections::BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            unsafe {
                c.insert_before_unchecked(i, i);
            }
        }
        assert!(m.len() == n);

        let mut c = m.upper_bound_mut(Bound::Unbounded);
        let mut i = n;
        while let Some((k, v)) = c.remove_prev() {
            i -= 1;
            assert_eq!((k, v), (i, i));
        }
        assert_eq!(i, 0);
    }
}

#[test]
fn exp_cursor_remove_fwd_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = /*std::collections::*/ BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            c.insert_before(i, i).unwrap();
        }
        assert!(m.len() == n);

        let mut c = m.lower_bound_mut(Bound::Unbounded);
        let mut i = 0;
        while let Some((k, v)) = c.remove_next() {
            assert_eq!((k, v), (i, i));
            i += 1;
        }
        assert_eq!(i, n);
    }
}

#[test]
fn std_cursor_remove_fwd_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = std::collections::BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            c.insert_before(i, i).unwrap();
        }
        assert!(m.len() == n);

        let mut c = m.lower_bound_mut(Bound::Unbounded);
        let mut i = 0;
        while let Some((k, v)) = c.remove_next() {
            assert_eq!((k, v), (i, i));
            i += 1;
        }
        assert_eq!(i, n);
    }
}

#[test]
fn exp_cursor_insert_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = /*std::collections::*/ BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            c.insert_before(i, i).unwrap();
        }
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            let (k, v) = c.next().unwrap();
            assert_eq!((*k, *v), (i, i));
        }
        let mut c = m.upper_bound_mut(Bound::Unbounded);
        for i in 0..n {
            let (k, v) = c.prev().unwrap();
            assert_eq!((*k, *v), (n - i - 1, n - i - 1));
        }
    }
}

#[test]
fn std_cursor_insert_test() {
    for _rep in 0..1000 {
        let n = 10000;
        let mut m = std::collections::BTreeMap::<usize, usize>::new();
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            c.insert_before(i, i).unwrap();
        }
        let mut c = m.lower_bound_mut(Bound::Unbounded);
        for i in 0..n {
            let (k, v) = c.next().unwrap();
            assert_eq!((*k, *v), (i, i));
        }
    }
}

#[test]
fn mut_cursor_test() {
    let n = 200;
    let mut m = BTreeMap::<usize, usize>::new();
    for i in 0..n {
        m.insert(i, i);
    }

    let mut c = m.lower_bound_mut(Bound::Included(&105));
    for i in 105..n {
        let (k, v) = c.next().unwrap();
        // println!("x={:?}", x);
        assert_eq!((*k, *v), (i, i))
    }

    let mut c = m.lower_bound_mut(Bound::Excluded(&105));
    for i in 106..n {
        let (k, v) = c.next().unwrap();
        // println!("x={:?}", x);
        assert_eq!((*k, *v), (i, i))
    }

    let mut c = m.upper_bound_mut(Bound::Included(&105));
    for i in 106..n {
        let (k, v) = c.next().unwrap();
        // println!("x={:?}", x);
        assert_eq!((*k, *v), (i, i))
    }

    let mut c = m.upper_bound_mut(Bound::Excluded(&105));
    for i in 105..n {
        let (k, v) = c.next().unwrap();
        // println!("x={:?}", x);
        assert_eq!((*k, *v), (i, i))
    }

    let mut c = m.upper_bound_mut(Bound::Unbounded);
    for i in (0..n).rev() {
        let (k, v) = c.prev().unwrap();
        // println!("x={:?}", x);
        assert_eq!((*k, *v), (i, i))
    }

    let mut a = BTreeMap::new();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");
    a.insert(4, "d");
    let cursor = a.lower_bound_mut(Bound::Included(&2));
    assert_eq!(cursor.peek_prev(), Some((&1, &mut "a")));
    assert_eq!(cursor.peek_next(), Some((&2, &mut "b")));
    let cursor = a.lower_bound_mut(Bound::Excluded(&2));
    assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
    assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));

    let mut a = BTreeMap::new();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");
    a.insert(4, "d");
    let cursor = a.upper_bound_mut(Bound::Included(&3));
    assert_eq!(cursor.peek_prev(), Some((&3, &mut "c")));
    assert_eq!(cursor.peek_next(), Some((&4, &mut "d")));
    let cursor = a.upper_bound_mut(Bound::Excluded(&3));
    assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
    assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));
}

#[test]
fn test_is_this_ub() {
    BTreeMap::new().entry(0).or_insert('a');

    let mut m = BTreeMap::new();
    m.insert(0, 'a');
    *m.entry(0).or_insert('a') = 'b';
    match m.entry(0) {
        Entry::Occupied(e) => e.remove(),
        _ => panic!(),
    };
}

#[test]
fn basic_range_test() {
    let mut map = BTreeMap::<usize, usize>::new();
    for i in 0..100 {
        map.insert(i, i);
    }

    for j in 0..100 {
        assert_eq!(map.range(0..=j).count(), j + 1);
    }
}

#[test]
fn test_exp_insert_fwd() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_std_insert_fwd() {
    for _rep in 0..1000 {
        let mut t = std::collections::BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_exp_insert_rev() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in (0..n).rev() {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_std_insert_rev() {
    for _rep in 0..1000 {
        let mut t = std::collections::BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in (0..n).rev() {
            t.insert(i, i);
        }
    }
}

#[test]
fn test_exp_entry() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.entry(i).or_insert(i);
        }
    }
}

#[test]
fn test_std_entry() {
    for _rep in 0..1000 {
        let mut t = std::collections::BTreeMap::<usize, usize>::default();
        let n = 10000;
        for i in 0..n {
            t.entry(i).or_insert(i);
        }
    }
}

#[test]
fn test_exp_iter() {
    let mut m = /*std::collections::*/ BTreeMap::<usize, usize>::default();
    let n = 100000;
    for i in 0..n {
        m.entry(i).or_insert(i);
    }
    for _rep in 0..1000 {
        for (k, v) in m.iter() {
            assert!(k == v);
        }
    }
}

#[test]
fn test_std_iter() {
    let mut m = std::collections::BTreeMap::<usize, usize>::default();
    let n = 100000;
    for i in 0..n {
        m.entry(i).or_insert(i);
    }
    for _rep in 0..1000 {
        for (k, v) in m.iter() {
            assert!(k == v);
        }
    }
}

#[test]
fn test_exp_into_iter() {
    for _rep in 0..100 {
        let mut m = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        let n = 100000;
        for i in 0..n {
            m.insert(i, i);
        }
        for (k, v) in m {
            assert!(k == v);
        }
    }
}

#[test]
fn test_std_into_iter() {
    for _rep in 0..100 {
        let mut m = std::collections::BTreeMap::<usize, usize>::default();
        let n = 100000;
        for i in 0..n {
            m.insert(i, i);
        }
        for (k, v) in m {
            assert!(k == v);
        }
    }
}

#[test]
fn various_tests() {
    for _rep in 0..1000 {
        let mut t = /*std::collections::*/ BTreeMap::<usize, usize>::default();
        t.check();
        let n = 10000;
        for i in 0..n {
            t.insert(i, i);
        }
        if false {
            assert!(t.first_key_value().unwrap().0 == &0);
            assert!(t.last_key_value().unwrap().0 == &(n - 1));

            println!("doing for x in & test");
            for x in &t {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("doing for x in &mut test");
            for x in &mut t {
                *x.1 *= 1;
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("doing range mut test");

            for x in t.range_mut(20..=60000).rev() {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!("done range mut test");

            println!("t.len()={} doing range non-mut test", t.len());

            for x in t.range(20..=60000).rev() {
                if *x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!("done range non-mut test");

            println!("doing get test");
            for i in 0..n {
                assert_eq!(t.get(&i).unwrap(), &i);
            }

            println!("doing get_mut test");
            for i in 0..n {
                assert_eq!(t.get_mut(&i).unwrap(), &i);
            }

            /*
                    println!("t.len()={} doing walk test", t.len());
                    t.walk(&10, &mut |(k, _): &(usize, usize)| {
                        if *k <= 50 {
                            print!("{:?};", k);
                            false
                        } else {
                            true
                        }
                    });
                    println!();
            */

            println!("doing remove evens test");
            for i in 0..n {
                if i % 2 == 0 {
                    assert_eq!(t.remove(&i).unwrap(), i);
                }
            }

            /*
                    println!("t.len()={} re-doing walk test", t.len());
                    t.walk(&10, &mut |(k, _): &(usize, usize)| {
                        if *k <= 50 {
                            print!("{:?};", k);
                            false
                        } else {
                            true
                        }
                    });
                    println!();
            */

            println!("doing retain test - retain only keys divisible by 5");
            t.retain(|k, _v| k % 5 == 0);

            println!("Consuming iterator test");
            for x in t {
                if x.0 < 50 {
                    print!("{:?};", x);
                }
            }
            println!();

            println!("FromIter collect test");
            let a = [1, 2, 3];
            let map: BTreeMap<i32, i32> = a.iter().map(|&x| (x, x * x)).collect();
            for x in map {
                print!("{:?};", x);
            }
            println!();

            println!("From test");
            let map = BTreeMap::from([(1, 2), (3, 4)]);
            for x in map {
                print!("{:?};", x);
            }
            println!();
        }
    }
}
