use std::{str::Chars, iter::Peekable, borrow::Cow, num::ParseFloatError};

use super::Mat;

pub(crate) enum UnaryOpKind {
    Transpose,
    Inverse,
    UnaryMinus,
}

pub(crate) enum BinaryOpKind {
    Add,
    Subtract,
    Multiply,
}

pub(crate) enum MatdExpression {
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

#[derive(Debug, Clone)]
pub(crate) enum ParseOrEvalError {
    Parse(MatdParseError),
    Eval(EvalError),
}

impl From<MatdParseError> for ParseOrEvalError {
    fn from(value: MatdParseError) -> Self {
        Self::Parse(value)
    }
}

impl From<EvalError> for ParseOrEvalError {
    fn from(value: EvalError) -> Self {
        Self::Eval(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EvalError {
    ArityMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidOperation,
}

impl MatdExpression {
    pub const fn nargs(&self) -> usize {
        //TODO: non-recursive version?
        match self {
            MatdExpression::NextArg => 1,
            MatdExpression::Literal { .. } => 0,
            MatdExpression::UnaryOp { inner, .. } => inner.nargs(),
            MatdExpression::BinaryOp { lhs, rhs, .. } => lhs.nargs() + rhs.nargs(),
        }
    }

    fn eval_inner<'a>(&self, args: &mut impl Iterator<Item=&'a Mat>) -> Result<Cow<'a, Mat>, EvalError> {
        #[inline]
        fn scale_inplace<'a>(lhs: Cow<'a, Mat>, scalar: f64) -> Mat {
            match lhs {
                Cow::Borrowed(lhs) => lhs.scale(scalar),
                // Optimization: skip copying
                Cow::Owned(mut lhs) => {
                    lhs.scale_inplace(scalar);
                    lhs
                }
            }
        }
        
        match self {
            MatdExpression::NextArg => {
                match args.next() {
                    Some(arg) => Ok(Cow::Borrowed(arg)),
                    None => Err(EvalError::ArityMismatch { expected: 1, actual: 0 }),
                }
            },
            MatdExpression::Literal { value } => {
                Ok(Cow::Owned(Mat::scalar(*value)))
            },
            MatdExpression::UnaryOp { kind, inner } => {
                let inner = inner.eval_inner(args)?;

                let result = match kind {
                    UnaryOpKind::Transpose =>
                        match inner {
                            Cow::Borrowed(inner) => inner.transpose(),
                            // Optimization: skip copying
                            Cow::Owned(inner) => inner.transpose_inplace(),
                        },
                    UnaryOpKind::Inverse =>
                        match inner.as_ref().inv() {
                            Some(inv) => inv,
                            // Singular matrix
                            None => return Err(EvalError::InvalidOperation),
                        },
                    UnaryOpKind::UnaryMinus => scale_inplace(inner, -1.),
                };

                Ok(Cow::Owned(result))
            }
            MatdExpression::BinaryOp { kind, lhs, rhs } => {
                let lhs = lhs.eval_inner(args)?;
                let rhs = rhs.eval_inner(args)?;
                let res = match kind {
                    BinaryOpKind::Add =>
                        // Try to reuse arguments
                        match lhs {
                            Cow::Owned(lhs) => lhs + rhs.as_ref(),
                            Cow::Borrowed(lhs) => match rhs {
                                // Matrix addition is commutative
                                Cow::Owned(rhs) => rhs + lhs,
                                Cow::Borrowed(rhs) => lhs + rhs, // Sometimes we just need to copy
                            },
                        },
                    BinaryOpKind::Subtract =>
                        match lhs {
                            Cow::Owned(lhs) => lhs - rhs.as_ref(), // Reuse lhs
                            Cow::Borrowed(lhs) => lhs - rhs.as_ref(),
                        },
                    BinaryOpKind::Multiply => {
                        if let Some(scalar) = lhs.as_ref().as_scalar() {
                            scale_inplace(rhs, scalar)
                        } else if let Some(scalar) = rhs.as_ref().as_scalar() {
                            scale_inplace(lhs, scalar)
                        } else {
                            lhs.as_ref().matmul(rhs.as_ref())
                        }
                    },
                };

                Ok(Cow::Owned(res))
            }
        }
    }
    pub fn eval(&self, args: &[&Mat]) -> Result<Mat, EvalError> {
        let arity = self.nargs();
        if args.len() != arity {
            return Err(EvalError::ArityMismatch { expected: arity, actual: args.len() })
        }
        let mut iter = args.iter().copied();
        let res = self.eval_inner(&mut iter)?;
        Ok(match res {
            Cow::Borrowed(res) => res.clone(),
            Cow::Owned(res) => res,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) enum MatdParseError {
    UnexpectedEOF,
    UnexpectedChar {
        expected: char,
        actual: char,
    },
    InvalidScalarLiteral(ParseFloatError),
    Empty,
    ConstantExpression,
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

    pub fn parse(expr: &'a str) -> Result<Box<MatdExpression>, MatdParseError> {
        let mut parser = Self::new(expr);
        let res = parser.parse_recurse(None, false)?;
        if res.nargs() == 0 {
            return Err(MatdParseError::ConstantExpression);
        }
        Ok(res)
    }

    fn peek(&mut self) -> Option<&char> {
        self.expr.peek()
    }

    fn expect(&mut self, expected: char) -> Result<(), MatdParseError> {
        let actual = self.expr.next()
            .ok_or(MatdParseError::UnexpectedEOF)?;
        if expected != actual {
            return Err(MatdParseError::UnexpectedChar { expected, actual });
        }
        Ok(())
    }

    fn parse_transpose(&mut self, acc: Option<Box<MatdExpression>>) -> Result<Box<MatdExpression>, MatdParseError> {
        self.expect('\'')?;
        Ok(Box::new(MatdExpression::UnaryOp {
            kind: UnaryOpKind::Transpose,
            // either a syntax error or a math op failed, producing null
            inner: acc.expect("missing acc")
        }))
    }

    fn parse_inverse(&mut self, acc: Option<Box<MatdExpression>>) -> Result<Box<MatdExpression>, MatdParseError> {
        let acc = acc.unwrap();
        // handle inverse ^-1. No other exponents are allowed.
        self.expect('^')?;
        self.expect('-')?;
        self.expect('1')?;

        Ok(Box::new(MatdExpression::UnaryOp {
            kind: UnaryOpKind::Inverse,
            inner: acc,
        }))
    }

    fn parse_scalar(&mut self) -> Result<Box<MatdExpression>, MatdParseError> {
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

        match text.parse::<f64>() {
            Ok(value) => Ok(Box::new(MatdExpression::Literal { value })),
            Err(e) => Err(MatdParseError::InvalidScalarLiteral(e)),
        }
    }

    /// handle right-associative operators, greedily consuming them. These
    /// include transpose and inverse. This is called by the main recursion
    /// method.
    fn gobble_right(&mut self, mut acc: Option<Box<MatdExpression>>) -> Result<Box<MatdExpression>, MatdParseError> {
        while let Some(c) = self.peek() {
            acc = Some(match c {
                '\'' => self.parse_transpose(acc)?,
                '^' => self.parse_inverse(acc)?,
                _ => break,
            })
        }
        Ok(acc.unwrap())
    }
    
    fn implicit_mul(acc: Option<Box<MatdExpression>>, rhs: Box<MatdExpression>) -> Box<MatdExpression> {
        if let Some(acc) = acc {
            Box::new(MatdExpression::BinaryOp { kind: BinaryOpKind::Multiply, lhs: acc, rhs })
        } else {
            rhs
        }
    }

    /// @garb, garbpos  A list of every matrix allocated during evaluation... used to assist cleanup.
    /// @oneterm: we should return at the end of this term (i.e., stop at a PLUS, MINUS, LPAREN).
    fn parse_recurse(&mut self, mut acc: Option<Box<MatdExpression>>, oneterm: bool) -> Result<Box<MatdExpression>, MatdParseError> {
        while let Some(c) = self.peek() {
            match *c {
                '(' => {
                    if oneterm {
                        if let Some(acc) = acc {
                            return Ok(acc);
                        }
                    }
                    self.expect('(')?;
                    let rhs = self.parse_recurse(None, false)?;
                    let rhs = self.gobble_right(Some(rhs))?;
                    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                ')' => {
                    if !oneterm {
                        self.expect(')')?;
                    }
                    return Ok(acc.unwrap());
                },
                '*' => {
                    self.expect('*')?;
                    let rhs = self.parse_recurse(None, true)?;
                    let rhs = self.gobble_right(Some(rhs))?;
    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                'F' => {
                    self.expect('F')?;
                    let rhs = Box::new(MatdExpression::NextArg);
                    let rhs = self.gobble_right(Some(rhs))?;
    
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                'M' => {
                    self.expect('M')?;
                    let rhs = Box::new(MatdExpression::NextArg);
                    let rhs = self.gobble_right(Some(rhs))?;

                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                /*'D' => {
                    int rows = expr[*pos+1]-'0';
                    int cols = expr[*pos+2]-'0';
    
                    matd_t *rhs = matd_create(rows, cols);
    
                } */
                // a constant (SCALAR) defined inline. Treat just like M, creating a matd_t on the fly.
                '0'..='9' | '.' => {
                    let rhs = self.parse_scalar()?;
                    let rhs = self.gobble_right(Some(rhs))?;
                    acc = Some(Self::implicit_mul(acc, rhs));
                },
                '+' => {
                    self.expect('+')?;
                    // don't support unary plus
                    let _acc = acc.expect("Unary plus not supported");
                    if oneterm {
                        return Ok(_acc);
                    }

                    let rhs = self.parse_recurse(None, true)?;
                    let rhs = self.gobble_right(Some(rhs))?;

                    acc = Some(Box::new(MatdExpression::BinaryOp { kind: BinaryOpKind::Add, lhs: _acc, rhs }));
                },
                '-' => {
                    if let Some(_acc) = acc {
                        if oneterm {
                            return Ok(_acc);
                        }

                        // subtract
                        self.expect('-')?;
                        let rhs = self.parse_recurse(None, true)?;
                        let rhs = self.gobble_right(Some(rhs))?;

                        acc = Some(Box::new(MatdExpression::BinaryOp { kind: BinaryOpKind::Subtract, lhs: _acc, rhs }));
                    } else {
                        self.expect('-')?;

                        // unary minus
                        let rhs = self.parse_recurse(None, true)?;
                        let rhs = self.gobble_right(Some(rhs))?;

                        acc = Some(Box::new(MatdExpression::UnaryOp { kind: UnaryOpKind::UnaryMinus, inner: rhs }));
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

        acc.ok_or(MatdParseError::Empty)
    }
}

// TODO Optimization: Some operations we could perform in-place,
// saving some memory allocation work. E.g., ADD, SUBTRACT. Just need
// to make sure that we don't do an in-place modification on a matrix
// that was an input argument!

impl Mat {
    pub(crate) fn compile_op(expr: &str) -> Result<Box<MatdExpression>, MatdParseError> {
        MatdExpressionParser::parse(expr)
    }

    pub(crate) fn op(expr: &str, args: &[&Mat]) -> Result<Mat, ParseOrEvalError> {
        let expr = MatdExpressionParser::parse(expr)?;

        let res = expr.eval(args)?;
        Ok(res)
    }
}