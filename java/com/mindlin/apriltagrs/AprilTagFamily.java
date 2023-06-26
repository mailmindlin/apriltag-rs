package com.mindlin.apriltagrs;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.mindlin.apriltagrs.AprilTagLibrary.NativeObject;
import com.mindlin.apriltagrs.AprilTagLibrary.NativeObjectReleasedException;
import com.mindlin.apriltagrs.util.Point2i;

public final class AprilTagFamily extends NativeObject {
    /**
     * Default number of bits to correct
     */
    public static final int DEFAULT_HAMMING = 2;
    /**
     * Maximum value for maxHamming
     */
    public static final int MAX_HAMMING = 3;

    private static native long nativeForName(String name) throws IllegalArgumentException;
    private static native long[] nativeGetCodes(long ptr);
    private static native int[] nativeGetBits(long ptr);
    private static native int nativeGetWidthAtBorder(long ptr);
    private static native int nativeGetTotalWidth(long ptr);
    private static native boolean nativeHasReversedBorder(long ptr);
    private static native int nativeGetMinHamming(long ptr);
    private static native String nativeGetName(long ptr);
    private static native void nativeDestroy(long ptr);

    /**
     * Look up named AprilTagFamily
     * @param name Family name
     * @return
     * @throws IllegalArgumentException
     */
    public static AprilTagFamily forName(String name) throws IllegalArgumentException {
        AprilTagLibrary.loadLibrary();
        Objects.requireNonNull(name);

        long ptr = AprilTagFamily.nativeForName(name);
        return new AprilTagFamily(ptr);
    }

    private AprilTagFamily(long ptr) {
        super(ptr);
    }

    public long[] getCodes() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeGetCodes);
    }

    public List<? extends Point2i> getBits() throws NativeObjectReleasedException {
        var arr = this.nativeRead(AprilTagFamily::nativeGetBits);
        assert arr.length % 2 == 0;
        var result = new ArrayList<Point2i>(arr.length / 2);
        for (int i = 0; i < arr.length / 2; i++) {
            int x = arr[i * 2 + 0];
            int y = arr[i * 2 + 1];
            result.add(new Point2i(x, y));
        }
        return Collections.unmodifiableList(result);
    }

    public int getWidthAtBorder() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeGetWidthAtBorder);
    }

    public int getTotalWidth() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeGetTotalWidth);
    }

    public boolean hasReversedBorder() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeHasReversedBorder);
    }

    public int getMinHamming() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeGetMinHamming);
    }

    void validateBitsCorrected(int bitsCorrected) throws IllegalArgumentException {
        if (bitsCorrected < 0)
            throw new IllegalArgumentException("maxHamming negative for family " + this.getName());
        int maxMaxHamming = Math.min(this.getMinHamming(), AprilTagFamily.MAX_HAMMING);
        if (bitsCorrected > maxMaxHamming)
            throw new IllegalArgumentException("maxHamming too large for family " + this.getName());
    }

    public String getName() throws NativeObjectReleasedException {
        return this.nativeRead(AprilTagFamily::nativeGetName);
    }

    @Override
    protected void destroy(long ptr) {
        AprilTagFamily.nativeDestroy(ptr);
    }
}