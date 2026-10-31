//! Candidate batching retains each backend's coefficient accumulators and tail.
#[allow(unused_macros)]
macro_rules! batched_costs {
    ($feature:literal, $lanes:literal, $name:ident, $single:ident, $term:ident,
     $load:ident, $splat:ident, $add:ident, $sum:ident) => {
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $feature)]
        pub(crate) fn $name(
            m: &[f32],
            s: &[f32],
            factors: &[f32],
            rates: &[f32],
            first: usize,
            weight: f32,
            lambda: f32,
            bias: crate::group::DequantBias,
            costs: &mut [f32],
        ) {
            assert_eq!(factors.len(), costs.len());
            assert_eq!(rates.len(), costs.len());
            if factors.is_empty() {
                return;
            }
            assert!(first < factors.len());
            let len = m.len().min(s.len());
            let end = len / $lanes * $lanes;
            let use_batch = factors.len() >= 4
                && len >= 64
                && [0, (len / (2 * $lanes)) * $lanes, end - $lanes]
                    .iter()
                    .any(|&i| m[i..i + $lanes].iter().all(|&v| v == 0.0));
            if use_batch {
                let can_share = weight.is_finite()
                    && lambda.is_finite()
                    && bias.one.is_finite()
                    && bias.numerator.is_finite()
                    && factors.iter().all(|v| v.is_finite());
                let (batches, rest) = factors.as_chunks::<4>();
                let (outputs, tail) = costs.as_chunks_mut::<4>();
                for (factors, output) in batches.iter().zip(outputs) {
                    let mut acc = [$splat(0.0); 4];
                    for x in (0..end).step_by($lanes) {
                        // Loading through arrays also works with Wasm's v128 API.
                        let mv = $load(m[x..x + $lanes].try_into().unwrap());
                        let sv = $load(s[x..x + $lanes].try_into().unwrap());
                        if can_share
                            && m[x..x + $lanes].iter().all(|&v| v == 0.0)
                            && s[x..x + $lanes].iter().all(|v| v.is_finite())
                        {
                            let best = $term(mv, sv, factors[0], weight, lambda, bias);
                            for a in &mut acc {
                                *a = $add(*a, best);
                            }
                        } else {
                            for i in 0..4 {
                                acc[i] =
                                    $add(acc[i], $term(mv, sv, factors[i], weight, lambda, bias));
                            }
                        }
                    }
                    for i in 0..4 {
                        output[i] = $sum(acc[i])
                            + crate::color_correlation::cfl_rd_cost_scalar(
                                &m[end..len],
                                &s[end..len],
                                factors[i],
                                weight,
                                lambda,
                                bias,
                            );
                    }
                }
                for (&factor, out) in rest.iter().zip(tail) {
                    *out = $single(m, s, factor, weight, lambda, bias);
                }
            } else if weight >= 0.0 && lambda >= 0.0 {
                let mut limit = f32::INFINITY;
                for i in std::iter::once(first).chain((0..factors.len()).filter(|&i| i != first)) {
                    let mut acc = $splat(0.0);
                    let mut pruned = false;
                    for x in (0..end).step_by($lanes) {
                        let mv = $load(m[x..x + $lanes].try_into().unwrap());
                        let sv = $load(s[x..x + $lanes].try_into().unwrap());
                        acc = $add(acc, $term(mv, sv, factors[i], weight, lambda, bias));
                        if (x + $lanes).is_multiple_of(64) && $sum(acc) + rates[i] > limit {
                            pruned = true;
                            break;
                        }
                    }
                    let cost = if pruned {
                        f32::INFINITY
                    } else {
                        $sum(acc)
                            + crate::color_correlation::cfl_rd_cost_scalar(
                                &m[end..len],
                                &s[end..len],
                                factors[i],
                                weight,
                                lambda,
                                bias,
                            )
                    };
                    costs[i] = cost;
                    if cost + rates[i] < limit {
                        limit = cost + rates[i];
                    }
                }
            } else {
                for (&factor, out) in factors.iter().zip(costs) {
                    *out = $single(m, s, factor, weight, lambda, bias);
                }
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use batched_costs;

#[cfg(test)]
mod tests;
