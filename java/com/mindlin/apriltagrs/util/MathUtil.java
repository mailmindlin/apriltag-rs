package com.mindlin.apriltagrs.util;

public final class MathUtil {
    private MathUtil() {}

    public static int argmax(double[] values) {
        assert values.length > 0;
        int choice = 0;
        var max = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i] > max) {
                choice = i;
                max = values[i];
            }
        }
        return choice;
    }
}
