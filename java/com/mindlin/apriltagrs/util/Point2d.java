package com.mindlin.apriltagrs.util;

import java.util.Objects;

public final class Point2d {
    public final double x;
    public final double y;
    public Point2d(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Point2d(Point2d src) {
        Objects.requireNonNull(src);
        this.x = src.x;
        this.y = src.y;
    }
}