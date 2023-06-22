package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.mindlin.apriltagrs.AprilTagLibrary.NativeObject;

public final class AprilTagFamily extends NativeObject {
    private static native long nativeForName(String name) throws IllegalArgumentException;
    private static native long[] nativeGetCodes(long ptr);
    private static native int[] nativeGetBits(long ptr);
    private static native String nativeGetName(long ptr);
    private static native void nativeDestroy(long ptr);

    public static AprilTagFamily forName(String name) throws IllegalArgumentException {
        AprilTagLibrary.loadLibrary();
        Objects.requireNonNull(name);

        long ptr = AprilTagFamily.nativeForName(name);
        return new AprilTagFamily(ptr);
    }

    private AprilTagFamily(long ptr) {
        super(ptr);
    }

    public long[] getCodes() throws IllegalStateException {
        return this.nativeRead(AprilTagFamily::nativeGetCodes);
    }

    public List<int[]> getBits() throws IllegalStateException {
        return this.nativeRead(AprilTagFamily::nativeGetBits);
    }

    public int getWidthAtBorder() {
        
    }

    public int getTotalWidth() {

    }

    public boolean hasReversedBorder() {

    }

    public int getMinHamming() {

    }

    public String getName() throws IllegalStateException {
        return this.nativeRead(AprilTagFamily::nativeGetName);
    }

    @Override
    protected void destroy(long ptr) {
        AprilTagFamily.nativeDestroy(ptr);
    }
}