package com.mindlin.apriltagrs.util;

import java.util.Objects;

public final class Matrix33 {
    private final double[] elements;

    public Matrix33(double e11, double e12, double e13, double e21, double e22, double e23, double e31, double e32, double e33) {
        this.elements = new double[] {
            e11, e12, e13,
            e21, e22, e23,
            e31, e32, e33,
        };
    }
    
    public Matrix33(double[] elements) {
        Objects.requireNonNull(elements);
        if (elements.length != 9)
            throw new IllegalArgumentException();
        this.elements = elements.clone();
    }
}
