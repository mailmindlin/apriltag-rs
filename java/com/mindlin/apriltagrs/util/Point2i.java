package com.mindlin.apriltagrs.util;

import java.util.Objects;

public final class Point2i {
    public final int x;
    public final int y;
    public Point2i(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Point2i(Point2i src) {
        Objects.requireNonNull(src);
        this.x = src.x;
        this.y = src.y;
    }
}