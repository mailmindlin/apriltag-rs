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

    public double get(int row, int col) {
        if (col < 0 || row < 0 || col >= 3 || row >= 3)
            throw new IllegalArgumentException();
        return this.elements[row * 3 + col];
    }

    /**
     * Converts a rotation matrix into a quaternion
     * @return
     */
    public double[] as_quaternion() {
        double[] decision = new double[] {
            get(0, 0),
            get(1, 1),
            get(2, 2),
            get(0, 0) + get(1, 1) + get(2, 2),
        };

        // Argmax
        double max = decision[0];
        int choice = 0;
        for (int i = 1; i < decision.length; i++) {
            if (decision[i] > max) {
                max = decision[i];
                choice = i;
            }
        }

        double[] quat = new double[4];
        if (choice == 3) {
            quat[0] = get(2, 1) - get(1, 2);
            quat[1] = get(0, 2) - get(2, 0);
            quat[2] = get(1, 0) - get(0, 1);
            quat[3] = 1 + decision[3];
        } else {
            int i = choice;
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;

            quat[i] = 1 - decision[3] + 2 * get(i, i);
            quat[j] = get(j, i) + get(i, j);
            quat[k] = get(k, i) + get(i, k);
            quat[3] = get(k, j) - get(j, k);
        }
        
        // Normalize
        var norm = Math.sqrt((quat[0] * quat[0]) + (quat[1] * quat[1]) + (quat[2] * quat[2]) + (quat[3] * quat[3]));

        double scale = (norm == 0.0) ? Double.NaN : 1. / norm;
        for (int i = 0; i < quat.length; i++)
            quat[i] *= scale;

        return quat;
    }
}
