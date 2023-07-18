package com.mindlin.apriltagrs;

import java.util.Map;

import com.mindlin.apriltagrs.AprilTagLibrary.NativeObject;

public final class AprilTagDetection extends NativeObject {
    private static native long nativeGetFamilyPointer(long ptr) throws NullPointerException;
    private static native int nativeGetTagId(long ptr);
    private static native int nativeGetHammingDistance(long ptr);
    private static native float nativeGetDecisionMargin(long ptr);
    private static native double[] nativeGetHomogrophy(long ptr);
    
    private final Map<Long, AprilTagFamily> familyLookup;
    protected AprilTagDetection(long ptr, Map<Long, AprilTagFamily> familyLookup) {
        super(ptr);
        this.familyLookup = familyLookup;
    }

    public AprilTagFamily getFamily() {
        var familyPtr = this.nativeRead(AprilTagDetection::nativeGetFamilyPointer);
        var family = this.familyLookup.get(familyPtr);
        //TODO: Check not null?
        return family;
    }

    /**
     * Get the decoded tag ID
     */
    public int getTagId() {
        return this.nativeRead(AprilTagDetection::nativeGetTagId);
    }

    /**
     * Get the number of error bits that were corrected
     */
    public int getHammingDistance() {
        return this.nativeRead(AprilTagDetection::nativeGetHammingDistance);
    }

    public float getDecisionMargin() {
        return this.nativeRead(AprilTagDetection::nativeGetDecisionMargin);
    }

    public double[] getHomogrophy() {
        return this.nativeRead(AprilTagDetection::nativeGetHomogrophy);
    }

    @Override
    protected native void destroy(long ptr);
}