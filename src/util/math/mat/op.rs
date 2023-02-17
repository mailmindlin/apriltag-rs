use std::{str::Chars, iter::Peekable, ops::Deref};

use super::Mat;

fn count_args(expr: &str) -> usize {
    let mut nargs = 0;
    for p in expr.chars() {
        match p {
            'M' | 'F' => {
                nargs += 1;
            }
        }
    }
    nargs
}

enum UnaryOpKind {
    Transpose,
    Inverse,
    UnaryMinus,
}

enum BinaryOpKind {
    Add,
    Subtract,
    Multiply,
}
enum MatdExpression {
    NextArg,
    Literal {
        value: f64,
    },
    UnaryOp {
        kind: UnaryOpKind,
        inner: Box<MatdExpression>,
    },
    BinaryOp {
        kind: BinaryOpKind,
        lhs: Box<MatdExpression>,
        rhs: Box<MatdExpression>,
    },
}

enum MaybeRef<'a, T> {
    Ref(&'a T),
    Val(T),
}

impl<'a, T> Deref for MaybeRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &'a Self::Target {
        match self {
            MaybeRef::Ref(r) => *r,
            MaybeRef::Val(v) => v,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EvalError {
    NotEnoughArguments,
    InvalidOperation,
}

impl MatdExpression {
    pub fn next_arg() -> Box<MatdExpression> {
        box MatdExpression::NextArg
    }

    pub fn nargs(&self) -> usize {
        //TODO: non-recursive version?
        match self {
            MatdExpression::NextArg => 1,
            MatdExpression::Literal { value } => 0,
            MatdExpression::UnaryOp { inner, .. } => inner.nargs(),
            MatdExpression::BinaryOp { lhs, rhs, .. } => lhs.nargs() + rhs.nargs(),
        }
    }

    fn eval_inner<'a>(&self, args: &mut impl Iterator<Item=&'a Mat>) -> Result<MaybeRef<'a, Mat>, EvalError> {
        #[inline]
        fn scale_inplace<'a>(lhs: MaybeRef<'a, Mat>, scalar: f64) -> Mat {
            match lhs {
                MaybeRef::Ref(lhs) => lhs.scale(scalar),
                // Optimization: skip copying
                MaybeRef::Val(lhs) => {
                    lhs.scale_inplace(scalar);
                    lhs
                }
            }
        }
        
        match self {
            MatdExpression::NextArg => {
                match args.next() {
                    Some(arg) => Ok(MaybeRef::Ref(arg)),
                    None => Err(EvalError::NotEnoughArguments),
                }
            },
            MatdExpression::Literal { value } => {
                Ok(MaybeRef::Val(Mat::scalar(*value)))
            },
            MatdExpression::UnaryOp { kind, inner } => {
                let inner = inner.eval_inner(args)?;

                let result = match kind {
                    UnaryOpKind::Transpose =>
                        match inner {
                            MaybeRef::Ref(inner) => inner.transpose(),
                            // Optimization: skip copying
                            MaybeRef::Val(inner) => inner.transpose_inplace(),
                        },
                    UnaryOpKind::Inverse =>
                        match inner.inv() {
                            Some(inv) => inv,
                            // Singular matrix
                            None => return Err(EvalError::InvalidOperation),
                        },
                    UnaryOpKind::UnaryMinus => scale_inplace(inner, -1.),
                };

                Ok(MaybeRef::Val(result))
            }
            MatdExpression::BinaryOp { kind, lhs, rhs } => {
                let lhs = lhs.eval_inner(args)?;
                let rhs = rhs.eval_inner(args)?;
                let res = match kind {
                    BinaryOpKind::Add =>
                        // Try to reuse arguments
                        match lhs {
                            MaybeRef::Val(lhs) => lhs + &*rhs,
                            MaybeRef::Ref(lhs) => match rhs {
                                // Matrix addition is commutative
                                MaybeRef::Val(rhs) => rhs + lhs,
                                MaybeRef::Ref(rhs) => lhs + rhs, // Sometimes we just need to copy
                            },
                        },
                    BinaryOpKind::Subtract =>
                        match lhs {
                            MaybeRef::Val(lhs) => lhs - &*rhs, // Reuse lhs
                            MaybeRef::Ref(lhs) => lhs - &*rhs,
                        },
                    BinaryOpKind::Multiply => {
                        if let Some(scalar) = lhs.as_scalar() {
                            scale_inplace(rhs, scalar)
                        } else if let Some(scalar) = rhs.as_scalar() {
                            scale_inplace(lhs, scalar)
                        } else {
                            lhs.matmul(&*rhs)
                        }
                    },
                };

                Ok(MaybeRef::Val(res))
            }
        }
    }
    pub fn eval(&self, args: &[&Mat]) -> Result<Mat, EvalError> {
        let mut iter = args.iter().copied();
        let res = self.eval_inner(&mut iter)?;
        Ok(match res {
            MaybeRef::Ref(res) => *res,
            MaybeRef::Val(res) => res,
        })
    }
}

struct MatdExpressionParser<'a> {
    expr: Peekable<Chars<'a>>,
}

impl<'a> MatdExpressionParser<'a> {
    fn new(expr: &'a str) -> Self {
        Self {
            expr: expr.chars().peekable(),
        }
    }

    pub fn parse(expr: &'a str) -> Box<MatdExpression> {
        let parser = Self::new(expr);
        parser.parse_recurse(None, false)
    }

    fn peek(&mut self) -> Option<&char> {
        self.expr.peek()
    }
    fn expect(&mut self, expected: char) {
        let actual = self.expr.next().expect("Unexpected EOF");
        assert_eq!(expected, actual, "Unexpected character");
    }

    fn parse_transpose(&mut self, acc: Option<Box<MatdExpression>>) -> Box<MatdExpression> {
        self.expect('\'');
        box MatdExpression::UnaryOp {
            kind: UnaryOpKind::Transpose,
            // either a syntax error or a math op failed, producing null
            inner: acc.expect("missing acc")
        }
    }

    fn parse_inverse(&mut self, acc: Option<Box<MatdExpression>>) -> Box<MatdExpression> {
        let acc = acc.unwrap();
        // handle inverse ^-1. No other exponents are allowed.
        self.expect('^');
        self.expect('-');
        self.expect('1');

        box MatdExpression::UnaryOp {
            kind: UnaryOpKind::Inverse,
            inner: acc,
        }
    }

    fn parse_scalar(&mut self) -> Box<MatdExpression> {
        //TODO: this is probably not quite right
        let text = {
            let mut text = String::new();
            
            while let Some(cd) = self.peek() {
                match *cd {
                    '0'..='9' | '.' => {
                        text.push(*cd);
                    },
                    _ => break,
                }
            }
            text
        };

        assert_ne!(text.len(), 0, "Empty scalar literal");

        let value = text.parse::<f64>().expect("Unable to parse literal");
        
        box MatdExpression::Literal { value }
    }

    /// handle right-associative operators, greedily consuming them. These
    /// include transpose and inverse. This is called by the main recursion
    /// method.
    fn gobble_right(&mut self, mut acc: Option<Box<MatdExpression>>) -> Box<MatdExpression> {
        while let Some(c) = self.peek() {
            acc = Some(match c {
                '\'' => self.parse_transpose(acc),
                '^' => self.parse_inverse(acc),
                _ => break,
            })
        }
        acc.unwrap()
    }
    
    fn implicit_mul(acc: Option<Box<MatdExpression>>, rhs: Box<MatdExpression>) -> Box<MatdExpression> {
        if let Some(acc) = acc {
            box MatdExpression::BinaryOp { kind: BinaryOpKind::Multiply, lhs: acc, rhs }
        } else {
            rhs
        }
    }

    /// @garb, garbpos  A list of every matrix allocated during evaluation... used to assist cleanup.
    /// @oneterm: we should return at the end of this term (i.e., stop at a PLUS, MINUS, LPAREN).
    fn parse_recurse(&mut self, mut acc: Option<Box<MatdExpression>>, oneterm: bool) -> Box<MatdExpression> {
        while let Some(c) = self.peek() {
            match *c {
                '(' => {
                    if oneterm {
                        if let Some(acc) = acc {
                            return acc;
                        }
                    }
                    self.expect('(');
                    let rhs = self.parse_recurse(None, false);
                    let rhs = self.gobble_right(Some(rhs));
                    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                ')' => {
                    if !oneterm {
                        self.expect(')');
                    }
                    return acc.unwrap();
                },
                '*' => {
                    self.expect('*');
                    let rhs = self.parse_recurse(None, true);
                    let rhs = self.gobble_right(Some(rhs));
    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                'F' => {
                    self.expect('F');
                    let rhs = box MatdExpression::NextArg;
                    let rhs = self.gobble_right(Some(rhs));
    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                'M' => {
                    self.expect('M');
                    let rhs = box MatdExpression::NextArg;
                    let rhs = self.gobble_right(Some(rhs));

                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                /*'D' => {
                    int rows = expr[*pos+1]-'0';
                    int cols = expr[*pos+2]-'0';
    
                    matd_t *rhs = matd_create(rows, cols);
    
                } */
                // a constant (SCALAR) defined inline. Treat just like M, creating a matd_t on the fly.
                '0'..='9' | '.' => {
                    let rhs = self.parse_scalar();
                    let rhs = self.gobble_right(Some(rhs));
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                '+' => {
                    self.expect('+');
                    // don't support unary plus
                    let _acc = acc.expect("Unary plus not supported");
                    if oneterm {
                        return _acc;
                    }

                    let rhs = self.parse_recurse(None, true);
                    let rhs = self.gobble_right(Some(rhs));

                    acc = Some(box MatdExpression::BinaryOp { kind: BinaryOpKind::Add, lhs: _acc, rhs });
                },
                '-' => {
                    if let Some(_acc) = acc {
                        if oneterm {
                            return _acc;
                        }

                        // subtract
                        self.expect('-');
                        let rhs = self.parse_recurse(None, true);
                        let rhs = self.gobble_right(Some(rhs));

                        acc = Some(box MatdExpression::BinaryOp { kind: BinaryOpKind::Subtract, lhs: _acc, rhs });
                    } else {

                        self.expect('-');

                        // unary minus
                        let rhs = self.parse_recurse(None, true);
                        let rhs = self.gobble_right(Some(rhs));

                        acc = Some(box MatdExpression::UnaryOp { kind: UnaryOpKind::UnaryMinus, inner: rhs });
                    }
                }
                ' ' => {
                    // nothing to do. spaces are meaningless.
                },
                ch => {
                    panic!("Unknown character: '{ch}'");
                },
            }
        }

        acc.unwrap()
    }
}

// TODO Optimization: Some operations we could perform in-place,
// saving some memory allocation work. E.g., ADD, SUBTRACT. Just need
// to make sure that we don't do an in-place modification on a matrix
// that was an input argument!

impl Mat {
    pub fn op(expr: &str, args: &[&Mat]) -> Result<Mat, EvalError> {
        let expr = MatdExpressionParser::parse(expr);

        let nargs = expr.nargs();
        assert_ne!(nargs, 0, "Empty expression");
        assert_eq!(nargs, args.len(), "Arity mismatch");

        expr.eval(args)
    }
}