/// Polynomial (of degree > 0)
pub(crate) struct Poly {
    /// Coefficients, in order
    coefs: Vec<f64>,
}

impl Poly {
    /// Create new polynomial from coefficients
    pub fn new(coefs: &[f64]) -> Self {
        assert!(coefs.len() > 1, "Cannot create polynomial of degree zero");
        Self {
            coefs: coefs.to_vec(),
        }
    }
    
    /// Get polynomial degree
    fn degree(&self) -> usize {
        self.coefs.len() - 1
    }

    /// Evaluates polynomial at x.
    fn eval(&self, x: f64) -> f64 {
        self.coefs
            .iter()
            .enumerate()
            .map(|(i, coef)| coef * f64::powi(x, i as i32))
            .sum::<f64>()
    }

    fn derivative(&self) -> Poly {
        let mut coefs = Vec::with_capacity(self.degree());
        for i in 0..self.degree() {
            coefs[i] = (i as f64 + 1.) * self.coefs[i + 1];
        }

        Self {
            coefs,
        }
    }

    /// Numerically solve small degree polynomials. This is a customized method. It
    /// ignores roots larger than 1000 and only gives small roots approximately.
    ///
    /// @param p Array of parameters s.t. p(x) = p[0] + p[1]*x + ...
    /// @param degree The degree of p(x).
    /// @return roots
    pub fn solve_approx(&self) -> Vec<f64> {
        const MAX_ROOT: f64 = 1000.;
        let ref p = self.coefs;
        let mut roots = Vec::new();
        match self.degree() {
            0 => panic!("No roots of degree 0"),
            1 => { // Degree 1
                if p[0].abs() > MAX_ROOT * p[1].abs() {
                    // 0 roots
                } else {
                    // 1 root
                    roots.push(-p[0]/p[1]);
                }
                return roots;
            },
            _ => {}
        }

        // Calculate roots of derivative.
        let p_der = self.derivative();
        let der_roots = p_der.solve_approx();

        // Go through all possibilities for roots of the polynomial.
        for i in 0..der_roots.len() {
            let min = if i == 0 {
                -MAX_ROOT
            } else {
                der_roots[i - 1]
            };

            let max = if i == der_roots.len() {
                MAX_ROOT
            } else {
                der_roots[i]
            };

            if self.eval(min) * self.eval(max) < 0. {
                // We have a zero-crossing in this interval, use a combination of Newton' and bisection.
                // Some thanks to Numerical Recipes in C.

                let (mut lower, mut upper) = if self.eval(min) < self.eval(max) {
                    (min, max)
                } else {
                    (max, min)
                };

                let mut root = 0.5*(lower + upper);
                let mut dx_old = upper - lower;
                let mut dx = dx_old;
                let mut f = self.eval(root);
                let mut df = p_der.eval(root);

                for j in 0..100 {
                    if ((f + df*(upper - root))*(f + df*(lower - root)) > 0.) || ((2.*f).abs() > (dx_old*df).abs()) {
                        dx_old = dx;
                        dx = 0.5*(upper - lower);
                        root = lower + dx;
                    } else {
                        dx_old = dx;
                        dx = -f/df;
                        root += dx;
                    }

                    if root == upper || root == lower {
                        break;
                    }

                    f = self.eval(root);
                    df = self.eval(root);

                    if f > 0. {
                        upper = root;
                    } else {
                        lower = root;
                    }
                }

                roots.push(root);
            } else if self.eval(max) == 0. {
                // Double/triple root.
                roots.push(max);
            }
        }

        roots
    }
}

#[cfg(test)]
mod test {
    use super::Poly;

    macro_rules! assert_close {
        ($a: expr, $b: expr) => {
            {
                let (a, b) = ($a, $b);
                const EPS: f64 = 1e-6;
                if f64::abs(a - b) >= EPS {
                    // Delegate 
                    assert_eq!(a, b);
                }
            }
        };
    }

    #[test]
    #[should_panic]
    fn disallow_empty() {
        Poly::new(&[]);
    }

    #[test]
    fn eval_1d() {
        let line = Poly::new(&[1., 2.]);
        assert_close!(line.eval(0.), 1.);
        assert_close!(line.eval(1.), 3.);
        assert_close!(line.eval(-1.), -1.);
    }

    #[test]
    fn eval_2d() {
        let parabola = Poly::new(&[1., 2., 3.,]);
        assert_close!(parabola.eval(0.), 1.);
        assert_close!(parabola.eval(1.), 6.);
        assert_close!(parabola.eval(2.), 17.);
        assert_close!(parabola.eval(-1.), 2.);
        assert_close!(parabola.eval(-2.), 9.);
    }
}