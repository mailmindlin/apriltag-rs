package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.util.Collections;
import java.util.List;

public final class AprilTagFamily implements Closeable {
    /**
     * Create new detector object
     * 
     * @return Handle to detector
     */
    private static native AprilTagFamily getForName(String name) throws IllegalArgumentException;

    private long ptr;

    private AprilTagFamily(long ptr) {
        this.ptr = ptr;
    }

    @Override
    public void close() {
        AprilTagDetector.destroy(ptr);
        this.ptr = 0;
    }
}