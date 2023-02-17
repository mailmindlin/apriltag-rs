package com.mindlin.apriltagrs;

public final class AprilTagParams {
    int nthreads = 1;
    float quadDecimate = 2.0f;
    float quadSigma = 0.0f;
    boolean refineEdges = true;
    float decodeSharpening = 0.25f;
    boolean debug = false;

    public AprilTagParams() {
    }

    public AprilTagParams(AprilTagParams params) {
        this.nthreads = params.nthreads;
        this.quadDecimate = params.quadDecimate;
        this.quadSigma = params.quadSigma;
        this.refineEdges = params.refineEdges;
        this.decodeSharpening = params.decodeSharpening;
        this.debug = params.debug;
    }
}