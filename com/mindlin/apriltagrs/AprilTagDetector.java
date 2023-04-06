package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.util.Collections;
import java.util.List;

public final class AprilTagDetector implements Closeable {
    /**
     * Create new detector object
     * 
     * @return Handle to detector
     */
    private static native long create();
    private static native void setParams(long ptr, int nthreads, float quadDecimate, float quadSigma, boolean refineEdges, float decodeSharpening, boolean debug);
    private static native void addFamily(long ptr, String familyName, short maxHamming);
    private static native void removeFamily(long ptr, String familyName);
    private static native long clearFamilies(long ptr);
    private static native void destroy(long ptr);

    private AprilTagParams params;
    private long ptr;

    public AprilTagDetector() {
        this(Collections.emptyList(), new AprilTagParams());
    }

    public AprilTagDetector(List<String> families, AprilTagParams params) {
        this.ptr = AprilTagDetector.create();
        this.params = new AprilTagParams(params);
    }

    private void updateParams() {
        AprilTagDetector.setParams(
            this.ptr,
            this.params.nthreads,
            this.params.quadDecimate,
            this.params.quadSigma,
            this.params.refineEdges,
            this.params.decodeSharpening,
            this.params.debug);
    }

    public void setThreads(int nthreads) {
        this.params.nthreads = nthreads;
        this.updateParams();
    }

    public int getThreads() {
        return this.params.nthreads;
    }

    public void setQuadDecimate(float quadDecimate) {
        this.params.quadDecimate = quadDecimate;
        this.updateParams();
    }

    public void addFamily(String familyName) {
        
    }

    public void addFamily(String familyName, int maxHamming) {

    }

    public void clearFamilies() {
        AprilTagDetector.clearFamilies(this.ptr);
    }

    @Override
    public void close() {
        AprilTagDetector.destroy(ptr);
        this.ptr = 0;
    }
}